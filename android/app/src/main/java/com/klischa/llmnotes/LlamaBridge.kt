package com.klischa.llmnotes

import android.util.Log

/**
 * JNI-мост для взаимодействия с C++ библиотекой llama.cpp.
 * Поддерживает автоопределение встроенных шаблонов чата моделей GGUF
 * (Llama-3, Qwen, Gemma, Mistral, Phi и др.) и потоковую генерацию.
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
    // Возвращает пустую строку при успехе, либо текст ошибки от llama.cpp
    external fun nativeLoadModel(modelPath: String, nCtx: Int, nThreads: Int): String
    external fun nativeFormatPrompt(systemPrompt: String, userPrompt: String): String
    external fun nativeFormatChat(roles: Array<String>, contents: Array<String>): String
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
