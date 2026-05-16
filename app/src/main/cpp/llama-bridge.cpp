/**
 * llama-bridge.cpp
 *
 * JNI bridge between Kotlin and the llama.cpp C API.
 *
 * Design: matches the Engine/Session mental model from LiteRT.
 *   - nativeLoadModel()    → long (model pointer)   — call once at app start
 *   - nativeCreateSession() → long (context pointer)  — call once per chat/query
 *   - nativeGenerate()     — streams tokens via callback into Kotlin
 *   - nativeFreeSession()  — frees context + sampler
 *   - nativeFreeModel()    — frees model weights
 *
 * Thread safety: every JNI call that touches a context pointer must be
 * dispatched on a single-thread Kotlin coroutine dispatcher (see LlamaEngine.kt).
 * The bridge itself does no locking.
 *
 * Tested against llama.cpp master / tag b5028.
 * API surface: llama_model_load_from_file, llama_init_from_model,
 *              llama_sampler_chain, llama_decode, llama_token_to_piece.
 */

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <cstring>

// llama.cpp public C API — relative path from CMakeLists.txt submodule location
#include "llama.cpp/include/llama.h"

#define TAG "LlamaBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Internal session state — one per InferenceSession on the Kotlin side
// ---------------------------------------------------------------------------
struct LlamaSession {
    llama_context * ctx     = nullptr;
    llama_sampler * sampler = nullptr;
    int             n_past  = 0;   // tokens already decoded into the KV cache
};

// ---------------------------------------------------------------------------
// Helper: convert jstring → std::string
// ---------------------------------------------------------------------------
static std::string jstring_to_std(JNIEnv * env, jstring js) {
    if (!js) return {};
    const char * chars = env->GetStringUTFChars(js, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(js, chars);
    return result;
}

// ---------------------------------------------------------------------------
// Helper: tokenize a UTF-8 string.
// Returns false if the token buffer is too large for the context.
// ---------------------------------------------------------------------------
static bool tokenize(
        const llama_model * model,
        const std::string & text,
        std::vector<llama_token> & out_tokens,
        bool add_special,
        bool parse_special)
{
    // First pass: measure
    int n = llama_tokenize(
            llama_model_get_vocab(model),
            text.c_str(), (int32_t) text.size(),
            nullptr, 0,
            add_special, parse_special);

    if (n < 0) {
        // Negative n means the buffer was too small; retry with correct size
        out_tokens.resize(-n);
        n = llama_tokenize(
                llama_model_get_vocab(model),
                text.c_str(), (int32_t) text.size(),
                out_tokens.data(), (int32_t) out_tokens.size(),
                add_special, parse_special);
    } else {
        out_tokens.resize(n);
        n = llama_tokenize(
                llama_model_get_vocab(model),
                text.c_str(), (int32_t) text.size(),
                out_tokens.data(), (int32_t) out_tokens.size(),
                add_special, parse_special);
    }

    if (n < 0) {
        LOGE("tokenize: failed (n=%d)", n);
        return false;
    }
    out_tokens.resize(n);
    return true;
}

// ---------------------------------------------------------------------------
// nativeLoadModel
//
// Loads GGUF weights from disk. Returns a pointer cast to jlong, or 0 on failure.
// Call once; keep alive for the app lifetime.
// ---------------------------------------------------------------------------
extern "C"
JNIEXPORT jlong JNICALL
Java_dev_yourname_graphrag_engine_LlamaEngine_nativeLoadModel(
        JNIEnv * env,
        jobject  /* this */,
        jstring  j_model_path)
{
    llama_backend_init();   // idempotent — safe to call multiple times

    std::string path = jstring_to_std(env, j_model_path);
    LOGI("Loading model: %s", path.c_str());

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU-only on Android; set >0 if Vulkan backend lands
    mparams.use_mmap     = true;

    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) {
        LOGE("Failed to load model from: %s", path.c_str());
        return 0L;
    }

    LOGI("Model loaded OK");
    return reinterpret_cast<jlong>(model);
}

// ---------------------------------------------------------------------------
// nativeCreateSession
//
// Creates a context + sampler for one chat/query lifecycle.
// system_prompt is prepended and decoded into the KV cache here, so the
// per-query generate() call only pays for new tokens.
// Returns a pointer to a heap-allocated LlamaSession, cast to jlong.
// ---------------------------------------------------------------------------
extern "C"
JNIEXPORT jlong JNICALL
Java_dev_yourname_graphrag_engine_LlamaEngine_nativeCreateSession(
        JNIEnv * env,
        jobject  /* this */,
        jlong    j_model_ptr,
        jstring  j_system_prompt,
        jint     j_n_ctx,
        jint     j_n_threads)
{
    auto * model = reinterpret_cast<llama_model *>(j_model_ptr);
    if (!model) {
        LOGE("nativeCreateSession: null model pointer");
        return 0L;
    }

    // --- Context ---
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = (uint32_t) j_n_ctx;      // e.g. 4096
    cparams.n_batch    = 512;
    cparams.n_threads  = (uint32_t) j_n_threads;  // e.g. 4

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOGE("nativeCreateSession: failed to create context");
        return 0L;
    }

    // --- Sampler chain ---
    // Order matters: temperature → top-k → top-p → min-p → dist (final sampler)
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // --- Pre-fill system prompt into KV cache ---
    std::string sys = jstring_to_std(env, j_system_prompt);
    int n_past = 0;

    if (!sys.empty()) {
        std::vector<llama_token> sys_tokens;
        if (!tokenize(model, sys, sys_tokens, /*add_special=*/true, /*parse_special=*/true)) {
            LOGE("nativeCreateSession: failed to tokenize system prompt");
            llama_sampler_free(sampler);
            llama_free(ctx);
            return 0L;
        }

        // Decode system prompt tokens in one batch
        llama_batch batch = llama_batch_get_one(sys_tokens.data(), (int32_t) sys_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            LOGE("nativeCreateSession: failed to decode system prompt");
            llama_sampler_free(sampler);
            llama_free(ctx);
            return 0L;
        }
        n_past = (int) sys_tokens.size();
        LOGI("System prompt pre-filled: %d tokens", n_past);
    }

    // --- Heap-allocate session ---
    auto * session   = new LlamaSession();
    session->ctx     = ctx;
    session->sampler = sampler;
    session->n_past  = n_past;

    return reinterpret_cast<jlong>(session);
}

// ---------------------------------------------------------------------------
// nativeGenerate
//
// Tokenizes `prompt`, appends to KV cache, then generates up to max_new_tokens,
// calling back into Kotlin with each decoded token piece.
//
// Callback signature (Kotlin):
//   interface TokenCallback { fun onToken(piece: String): Boolean }
//   returning false cancels generation (e.g. user tapped stop).
// ---------------------------------------------------------------------------
extern "C"
JNIEXPORT void JNICALL
Java_dev_yourname_graphrag_engine_LlamaEngine_nativeGenerate(
        JNIEnv * env,
jobject  /* this */,
jlong    j_session_ptr,
        jstring  j_prompt,
jint     j_max_new_tokens,
        jobject  j_callback)
{
auto * session = reinterpret_cast<LlamaSession *>(j_session_ptr);
if (!session || !session->ctx) {
LOGE("nativeGenerate: invalid session pointer");
return;
}

// Resolve callback method once
jclass   cb_class  = env->GetObjectClass(j_callback);
jmethodID cb_method = env->GetMethodID(
        cb_class, "onToken", "(Ljava/lang/String;)Z");
if (!cb_method) {
LOGE("nativeGenerate: onToken method not found");
return;
}

// Get the model from context
const llama_model * model = llama_get_model(session->ctx);
const llama_vocab * vocab  = llama_model_get_vocab(model);

// Tokenize prompt (no BOS — system prompt already consumed it)
std::string prompt = jstring_to_std(env, j_prompt);
std::vector<llama_token> prompt_tokens;
if (!tokenize(model, prompt, prompt_tokens, /*add_special=*/false, /*parse_special=*/false)) {
LOGE("nativeGenerate: failed to tokenize prompt");
return;
}

// Check context headroom
int n_ctx    = (int) llama_n_ctx(session->ctx);
int n_used   = session->n_past + (int) prompt_tokens.size();
if (n_used >= n_ctx) {
LOGE("nativeGenerate: context full (%d / %d)", n_used, n_ctx);
// Signal overflow to Kotlin side
jstring overflow_msg = env->NewStringUTF("[context full]");
env->CallBooleanMethod(j_callback, cb_method, overflow_msg);
env->DeleteLocalRef(overflow_msg);
return;
}

// Decode prompt tokens
llama_batch prompt_batch = llama_batch_get_one(
        prompt_tokens.data(), (int32_t) prompt_tokens.size());
if (llama_decode(session->ctx, prompt_batch) != 0) {
LOGE("nativeGenerate: failed to decode prompt");
return;
}
session->n_past += (int) prompt_tokens.size();

// --- Token generation loop ---
int       max_new   = j_max_new_tokens;
char      piece_buf[256];
bool      cancelled = false;

for (int i = 0; i < max_new && !cancelled; ++i) {

// Sample next token
llama_token token = llama_sampler_sample(session->sampler, session->ctx, -1);

// Check for end-of-generation tokens
if (llama_vocab_is_eog(vocab, token)) {
LOGI("nativeGenerate: EOG at step %d", i);
break;
}

// Decode token ID → UTF-8 piece
int piece_len = llama_token_to_piece(
        vocab, token,
        piece_buf, sizeof(piece_buf) - 1,
        /*special=*/false, /*lstrip=*/false);

if (piece_len < 0) {
// Buffer too small — allocate exact size
std::vector<char> big_buf(-piece_len + 1);
piece_len = llama_token_to_piece(
        vocab, token,
        big_buf.data(), (int32_t) big_buf.size(),
        /*special=*/false, /*lstrip=*/false);
big_buf[piece_len] = '\0';

jstring j_piece = env->NewStringUTF(big_buf.data());
jboolean cont   = env->CallBooleanMethod(j_callback, cb_method, j_piece);
env->DeleteLocalRef(j_piece);
cancelled = !cont;
} else if (piece_len > 0) {
piece_buf[piece_len] = '\0';
jstring j_piece = env->NewStringUTF(piece_buf);
jboolean cont   = env->CallBooleanMethod(j_callback, cb_method, j_piece);
env->DeleteLocalRef(j_piece);
cancelled = !cont;
}

if (cancelled) break;

// Feed token back for next step
llama_batch next_batch = llama_batch_get_one(&token, 1);
if (llama_decode(session->ctx, next_batch) != 0) {
LOGE("nativeGenerate: decode failed at step %d", i);
break;
}
session->n_past += 1;
}

LOGI("nativeGenerate: done (n_past=%d, cancelled=%d)", session->n_past, (int) cancelled);
}

// ---------------------------------------------------------------------------
// nativeFreeSession
//
// Frees context + sampler. Model weights are NOT freed here.
// Call this at the end of every query (InferenceSession.close()).
// ---------------------------------------------------------------------------
extern "C"
JNIEXPORT void JNICALL
Java_dev_yourname_graphrag_engine_LlamaEngine_nativeFreeSession(
        JNIEnv * /* env */,
jobject  /* this */,
jlong    j_session_ptr)
{
auto * session = reinterpret_cast<LlamaSession *>(j_session_ptr);
if (!session) return;

if (session->sampler) { llama_sampler_free(session->sampler); session->sampler = nullptr; }
if (session->ctx)     { llama_free(session->ctx);             session->ctx     = nullptr; }
delete session;

LOGI("Session freed");
}

// ---------------------------------------------------------------------------
// nativeFreeModel
//
// Frees model weights. Call from Application.onTerminate() only.
// ---------------------------------------------------------------------------
extern "C"
JNIEXPORT void JNICALL
Java_dev_yourname_graphrag_engine_LlamaEngine_nativeFreeModel(
        JNIEnv * /* env */,
jobject  /* this */,
jlong    j_model_ptr)
{
auto * model = reinterpret_cast<llama_model *>(j_model_ptr);
if (!model) return;

llama_model_free(model);
llama_backend_free();
LOGI("Model and backend freed");
}

// ---------------------------------------------------------------------------
// nativeContextUsage
//
// Returns the number of tokens currently consumed in the KV cache.
// Useful for showing a context-fill progress bar in the UI.
// ---------------------------------------------------------------------------
extern "C"
JNIEXPORT jint JNICALL
        Java_dev_yourname_graphrag_engine_LlamaEngine_nativeContextUsage(
        JNIEnv * /* env */,
        jobject  /* this */,
        jlong    j_session_ptr)
{
auto * session = reinterpret_cast<LlamaSession *>(j_session_ptr);
if (!session) return 0;
return (jint) session->n_past;
}