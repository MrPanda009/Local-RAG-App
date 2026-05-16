package com.example.local_rag_app.engine

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * Functional interface for token generation callbacks.
 * The callback is invoked for each generated token.
 * Return true to continue generation, false to cancel.
 */
fun interface TokenCallback {
    /**
     * Called when a token is generated.
     * @param token The generated token as a string.
     * @return true to continue generation, false to cancel.
     */
    suspend fun onToken(token: String): Boolean
}

/**
 * LlamaEngine is a singleton object that manages the lifecycle of a GGUF model
 * loaded via JNI. It operates on a dedicated single-thread executor to ensure
 * thread-safe access to the native model and sessions.
 */
object LlamaEngine {
    // Static initialization of the JNI library
    init {
        System.loadLibrary("graphrag-jni")
    }

    // Native model pointer
    private var modelPtr: Long = 0L

    // Single-thread dispatcher for all model operations
    private val singleThreadDispatcher: CoroutineDispatcher =
        Executors.newSingleThreadExecutor { thread ->
            Thread(thread).apply {
                name = "LlamaEngine-SingleThread"
                isDaemon = false
            }
        }.asCoroutineDispatcher()

    /**
     * Loads a GGUF model from the specified path.
     * This should be called once before creating any sessions.
     *
     * @param path The file path to the GGUF model.
     * @throws RuntimeException if the model fails to load.
     */
    suspend fun loadModel(path: String) {
        withContext(singleThreadDispatcher) {
            if (modelPtr != 0L) {
                throw IllegalStateException("Model already loaded. Call destroy() first.")
            }
            modelPtr = nativeLoadModel(path)
            if (modelPtr == 0L) {
                throw RuntimeException("Failed to load model from path: $path")
            }
        }
    }

    /**
     * Creates a new inference session with the given system prompt.
     *
     * @param systemPrompt The system prompt to initialize the session.
     * @return An InferenceSession ready for token generation.
     * @throws IllegalStateException if no model is loaded.
     */
    fun createSession(systemPrompt: String): InferenceSession {
        if (modelPtr == 0L) {
            throw IllegalStateException("Model not loaded. Call loadModel() first.")
        }
        val sessionPtr = nativeCreateSession(modelPtr, systemPrompt)
        if (sessionPtr == 0L) {
            throw RuntimeException("Failed to create inference session")
        }
        return InferenceSession(sessionPtr, singleThreadDispatcher)
    }

    /**
     * Frees an inference session.
     *
     * @param sessionPtr The native session pointer to free.
     */
    internal fun freeSession(sessionPtr: Long) {
        nativeFreeSession(sessionPtr)
    }

    /**
     * Gets the context usage of the model (e.g., actual tokens used / max context).
     *
     * @return A pair of (used, total) context tokens.
     */
    fun getContextUsage(): Pair<Int, Int> {
        if (modelPtr == 0L) {
            return 0 to 0
        }
        val usage = nativeContextUsage(modelPtr)
        // Assuming the native function packs usage as: (used shl 32 or total)
        val used = (usage shr 32).toInt()
        val total = usage.toInt()
        return used to total
    }

    /**
     * Cleans up resources. Should be called from Application.onTerminate().
     */
    fun destroy() {
        if (modelPtr != 0L) {
            nativeFreeModel(modelPtr)
            modelPtr = 0L
        }
    }

    // ==================== JNI Declarations ====================

    /**
     * Loads a GGUF model from the given file path.
     *
     * @param modelPath The file path to the GGUF model.
     * @return A native pointer to the loaded model, or 0 on failure.
     */
    private external fun nativeLoadModel(modelPath: String): Long

    /**
     * Creates an inference session for the given model.
     *
     * @param modelPtr The native model pointer.
     * @param systemPrompt The system prompt for the session.
     * @return A native pointer to the created session, or 0 on failure.
     */
    private external fun nativeCreateSession(modelPtr: Long, systemPrompt: String): Long

    /**
     * Generates tokens for the given session and prompt.
     * Calls the provided callback for each generated token.
     *
     * @param sessionPtr The native session pointer.
     * @param prompt The user prompt.
     * @param maxNewTokens The maximum number of tokens to generate.
     * @param callback The callback to invoke for each token.
     * @return true if generation completed successfully, false if cancelled.
     */
    internal external fun nativeGenerate(
        sessionPtr: Long,
        prompt: String,
        maxNewTokens: Int,
        callback: TokenCallback
    ): Boolean

    /**
     * Frees an inference session.
     *
     * @param sessionPtr The native session pointer to free.
     */
    private external fun nativeFreeSession(sessionPtr: Long)

    /**
     * Frees a model.
     *
     * @param modelPtr The native model pointer to free.
     */
    private external fun nativeFreeModel(modelPtr: Long)

    /**
     * Gets the context usage of a model.
     *
     * @param modelPtr The native model pointer.
     * @return Context usage encoded as a Long (format: used in upper 32 bits, total in lower 32 bits).
     */
    private external fun nativeContextUsage(modelPtr: Long): Long
}

