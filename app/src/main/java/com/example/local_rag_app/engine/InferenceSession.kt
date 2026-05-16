package com.example.local_rag_app.engine

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * InferenceSession wraps a native session pointer and provides a suspend-friendly API
 * for generating tokens using Kotlin coroutines Flow.
 *
 * @param sessionPtr The native session pointer obtained from the JNI layer.
 * @param dispatcher The CoroutineDispatcher on which to execute native calls.
 */
class InferenceSession(
    private val sessionPtr: Long,
    private val dispatcher: CoroutineDispatcher
) : Closeable {

    private var isClosed = false

    /**
     * Generates tokens for the given prompt.
     * Returns a Flow that emits each generated token as it becomes available.
     *
     * @param prompt The user prompt.
     * @param maxNewTokens The maximum number of tokens to generate (default: 512).
     * @return A Flow<String> that emits generated tokens.
     */
    fun generate(prompt: String, maxNewTokens: Int = 512): Flow<String> = callbackFlow {
        if (isClosed) {
            close(IllegalStateException("Session is closed"))
            return@callbackFlow
        }

        withContext(dispatcher) {
            // Create a callback that sends tokens to the flow
            val callback = TokenCallback { token ->
                val shouldContinue = try {
                    trySend(token).isSuccess
                } catch (_: Exception) {
                    // Channel might be closed if consumer cancelled
                    false
                }
                shouldContinue
            }

            try {
                val success = LlamaEngine.nativeGenerate(
                    sessionPtr,
                    prompt,
                    maxNewTokens,
                    callback
                )

                if (!success) {
                    // Generation was likely cancelled by the callback returning false
                    close()
                }
            } catch (e: Exception) {
                close(e)
            }
        }
    }

    /**
     * Closes the session and frees native resources.
     */
    override fun close() {
        if (!isClosed) {
            isClosed = true
            LlamaEngine.freeSession(sessionPtr)
        }
    }
}



