package com.klischa.llmnotes

import android.util.Log

/**
 * JNI-мост для взаимодействия с C++ библиотекой llama.cpp.
 * Обеспечивает загрузку GGUF-моделей, потоковую генерацию токенов
 * и аппаратную оптимизацию под процессор Helio G99.
 */
object LlamaBridge {
    private const val TAG = "LlamaBridge"

    init {
        try {
            System.loadLibrary("llama-android")
            Log.i(TAG, "Библиотека llama-android успешно загружена")
            nativeInit()
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Не удалось загрузить библиотеку llama-android: ${e.message}")
        }
    }

    fun interface TokenCallback {
        fun onToken(token: String)
    }

    // Внешние нативные C++ функции
    external fun nativeInit(): Boolean
    external fun nativeLoadModel(modelPath: String, nCtx: Int, nThreads: Int): Boolean
    external fun nativeGenerate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        callback: TokenCallback?
    ): String
    external fun nativeStop()
    external fun nativeUnload()
}
