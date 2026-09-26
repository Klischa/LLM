package com.klischa.llmnotes.api

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.URL
import java.util.UUID
import javax.net.ssl.HttpsURLConnection

/**
 * Типы провайдеров LLM:
 * - LOCAL_GGUF: локальный автономный запуск на процессоре Helio G99 через llama.cpp (100% бесплатно, без интернета).
 * - OPENROUTER: облачный API OpenRouter с десятками бесплатных моделей (:free).
 * - GROQ: облачный высокоскоростной API Groq с бесплатным доступом к LLaMA 3.3 70B.
 * - OPENCODE_ZEN: облачный API OpenCode подписки ZEN (https://opencode.ai/zen/v1).
 * - OPENCODE_GO: облачный API OpenCode подписки GO (https://opencode.ai/zen/go/v1).
 */
enum class LLMProviderType(val displayName: String, val baseUrl: String) {
    LOCAL_GGUF("Офлайн GGUF", ""),
    OPENROUTER("OpenRouter (Free)", "https://openrouter.ai/api/v1"),
    GROQ("Groq Cloud (Free)", "https://api.groq.com/openai/v1"),
    OPENCODE_ZEN("OpenCode ZEN", "https://opencode.ai/zen/v1"),
    OPENCODE_GO("OpenCode GO", "https://opencode.ai/zen/go/v1")
}

data class ApiChatMessage(
    val role: String,
    val content: String
)

/**
 * Клиент взаимодействия с внешними OpenAI-совместимыми API:
 * OpenRouter, Groq Cloud, OpenCode ZEN, OpenCode GO.
 */
object OpenCodeClient {
    private const val TAG = "OpenCodeClient"
    const val OPENCODE_USER_AGENT = "opencode/1.18.31 ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14"

    init {
        try {
            System.setProperty("http.agent", OPENCODE_USER_AGENT)
        } catch (_: Exception) {}
    }
    private const val BASE62_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    private val secureRandom = java.security.SecureRandom()
    private var lastTimestamp = 0L
    private var counter = 0L

    /**
     * Генератор идентификаторов OpenCode CLI.
     * Шлюз OpenCode проверяет временную метку и алгоритм инверсии битов:
     * - для сессий (ses_): инвертированное нисходящее время (desc = true)
     * - для сообщений/запросов (msg_): прямое восходящее время (desc = false)
     * Первые 12 hex-символов кодируют миллисекунды, затем следуют 14 символов Base62.
     */
    @Synchronized
    fun generateOpenCodeId(prefix: String, desc: Boolean): String {
        val now = System.currentTimeMillis()
        if (now != lastTimestamp) {
            counter = 1L
            lastTimestamp = now
        } else {
            counter++
        }
        var v = now * 4096L + counter
        if (desc) {
            v = v.inv()
        }
        val hexBuilder = StringBuilder(12)
        for (i in 0 until 6) {
            val shift = 40 - 8 * i
            val byteVal = ((v ushr shift) and 0xFFL).toInt()
            hexBuilder.append(String.format(java.util.Locale.US, "%02x", byteVal))
        }
        val rndBuilder = StringBuilder(14)
        for (i in 0 until 14) {
            val idx = secureRandom.nextInt(BASE62_CHARS.length)
            rndBuilder.append(BASE62_CHARS[idx])
        }
        return prefix + hexBuilder.toString() + rndBuilder.toString()
    }

    fun generateSessionId(): String = generateOpenCodeId("ses_", desc = true)
    fun generateRequestId(): String = generateOpenCodeId("msg_", desc = false)

    // Рекомендуемые бесплатные модели OpenRouter (суффикс :free гарантирует бесплатность)
    val OPENROUTER_DEFAULT_MODELS = listOf(
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-chat:free",
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemini-flash-1.5:free"
    )

    // Рекомендуемые бесплатные модели Groq Cloud
    val GROQ_DEFAULT_MODELS = listOf(
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    )

    // Рекомендуемые модели для подписки OpenCode GO
    val GO_DEFAULT_MODELS = listOf(
        "deepseek-v4-pro",
        "kimi-k2.6",
        "qwen3.6-plus",
        "glm-5.1",
        "minimax-m3",
        "gpt-5.6-luna",
        "grok-4.6",
        "qwen2.5-coder-32b",
        "deepseek-v3"
    )

    // Рекомендуемые модели для подписки OpenCode ZEN
    val ZEN_DEFAULT_MODELS = listOf(
        "qwen3.6-plus",
        "qwen3.8-flash",
        "qwen3.8-max",
        "claude-3-7-sonnet",
        "claude-3-5-sonnet",
        "claude-haiku-4-5",
        "claude-fable-5-1",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5.6-luna",
        "deepseek-r1",
        "deepseek-v3",
        "minimax-m3",
        "glm-5.1",
        "big-pickle",
        "space-bunny-free",
        "nemotron-3.5-lightning-free"
    )

    fun isFreeModel(provider: LLMProviderType, model: String): Boolean {
        return when (provider) {
            LLMProviderType.OPENROUTER -> model.contains(":free")
            LLMProviderType.GROQ -> true
            LLMProviderType.OPENCODE_ZEN -> model == "big-pickle" || model.endsWith("-free")
            else -> false
        }
    }

    fun getDefaultModels(provider: LLMProviderType): List<String> {
        return when (provider) {
            LLMProviderType.OPENROUTER -> OPENROUTER_DEFAULT_MODELS
            LLMProviderType.GROQ -> GROQ_DEFAULT_MODELS
            LLMProviderType.OPENCODE_GO -> GO_DEFAULT_MODELS
            LLMProviderType.OPENCODE_ZEN -> ZEN_DEFAULT_MODELS
            else -> emptyList()
        }
    }

    /**
     * Запрос актуального каталога моделей с сервера провайдера (GET /v1/models).
     */
    fun fetchModels(provider: LLMProviderType, apiKey: String): List<String> {
        if (apiKey.isBlank() || provider == LLMProviderType.LOCAL_GGUF) {
            return getDefaultModels(provider)
        }

        val endpoint = "${provider.baseUrl}/models"
        var conn: HttpsURLConnection? = null
        try {
            val url = URL(endpoint)
            conn = url.openConnection() as HttpsURLConnection
            conn.requestMethod = "GET"

            val rawKey = apiKey.removePrefix("Bearer ").removePrefix("bearer ").trim()
            conn.setRequestProperty("Authorization", "Bearer $rawKey")
            conn.setRequestProperty("Accept", "application/json")

            when (provider) {
                LLMProviderType.OPENROUTER -> {
                    conn.setRequestProperty("User-Agent", "LLM-Android/1.2.6 (Infinix Note 30)")
                    conn.setRequestProperty("HTTP-Referer", "https://github.com/Klischa/LLM")
                    conn.setRequestProperty("X-Title", "LLM Android Assistant")
                }
                LLMProviderType.GROQ -> {
                    conn.setRequestProperty("User-Agent", "LLM-Android/1.2.6 (Infinix Note 30)")
                }
                LLMProviderType.OPENCODE_GO, LLMProviderType.OPENCODE_ZEN -> {
                    val sessionId = generateSessionId()
                    val requestId = generateRequestId()
                    conn.setRequestProperty("x-api-key", rawKey)
                    conn.setRequestProperty("User-Agent", OPENCODE_USER_AGENT)
                    conn.setRequestProperty("x-opencode-client", "cli")
                    conn.setRequestProperty("x-opencode-project", "global")
                    conn.setRequestProperty("x-opencode-session", sessionId)
                    conn.setRequestProperty("x-opencode-request", requestId)
                }
                else -> {}
            }
            conn.connectTimeout = 10000
            conn.readTimeout = 15000

            val responseCode = conn.responseCode
            if (responseCode !in 200..299) {
                val err = conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "HTTP $responseCode"
                Log.w(TAG, "Ошибка fetchModels ($responseCode): $err")
                return getDefaultModels(provider)
            }

            val body = conn.inputStream.bufferedReader().use { it.readText() }
            val list = mutableListOf<String>()
            val trimmed = body.trim()
            if (trimmed.startsWith("[")) {
                val array = JSONArray(trimmed)
                for (i in 0 until array.length()) {
                    val item = array.optJSONObject(i)
                    val id = item?.optString("id") ?: array.optString(i)
                    if (id.isNotBlank()) list.add(id)
                }
            } else if (trimmed.startsWith("{")) {
                val json = JSONObject(trimmed)
                val array = json.optJSONArray("data") ?: json.optJSONArray("models")
                if (array != null) {
                    for (i in 0 until array.length()) {
                        val item = array.optJSONObject(i)
                        val id = item?.optString("id") ?: array.optString(i)
                        if (id.isNotBlank()) list.add(id)
                    }
                }
            }

            if (provider == LLMProviderType.OPENROUTER && list.isNotEmpty()) {
                val freeModels = list.filter { it.contains(":free") }.sorted()
                val otherModels = list.filter { !it.contains(":free") }.sorted()
                return (freeModels + otherModels).distinct()
            }

            return if (list.isNotEmpty()) (getDefaultModels(provider) + list).distinct().sorted() else getDefaultModels(provider)
        } catch (e: Exception) {
            Log.w(TAG, "Ошибка получения моделей: ${e.message}")
            return getDefaultModels(provider)
        } finally {
            conn?.disconnect()
        }
    }

    /**
     * Потоковая генерация ответа через Server-Sent Events (SSE).
     */
    fun streamChatCompletions(
        provider: LLMProviderType,
        apiKey: String,
        model: String,
        messages: List<ApiChatMessage>,
        conversationId: String = "",
        temperature: Float = 0.7f,
        onToken: (String) -> Unit,
        isCancelled: () -> Boolean
    ) {
        val endpoint = "${provider.baseUrl}/chat/completions"
        var conn: HttpsURLConnection? = null
        try {
            val url = URL(endpoint)
            conn = url.openConnection() as HttpsURLConnection
            conn.requestMethod = "POST"

            val rawKey = apiKey.removePrefix("Bearer ").removePrefix("bearer ").trim()

            conn.setRequestProperty("Authorization", "Bearer $rawKey")
            conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            conn.setRequestProperty("Accept", "text/event-stream")

            when (provider) {
                LLMProviderType.OPENCODE_GO, LLMProviderType.OPENCODE_ZEN -> {
                    val sessionId = generateSessionId()
                    val requestId = generateRequestId()
                    conn.setRequestProperty("x-api-key", rawKey)
                    conn.setRequestProperty("User-Agent", OPENCODE_USER_AGENT)
                    conn.setRequestProperty("x-opencode-client", "cli")
                    conn.setRequestProperty("x-opencode-project", "global")
                    conn.setRequestProperty("x-opencode-session", sessionId)
                    conn.setRequestProperty("x-opencode-request", requestId)
                    conn.setRequestProperty("x-session-id", sessionId)
                    conn.setRequestProperty("x-session-affinity", sessionId)
                }
                LLMProviderType.OPENROUTER -> {
                    conn.setRequestProperty("User-Agent", "LLM-Android/1.2.6 (Infinix Note 30)")
                    conn.setRequestProperty("HTTP-Referer", "https://github.com/Klischa/LLM")
                    conn.setRequestProperty("X-Title", "LLM Android Assistant")
                }
                LLMProviderType.GROQ -> {
                    conn.setRequestProperty("User-Agent", "LLM-Android/1.2.6 (Infinix Note 30)")
                }
                else -> {}
            }

            conn.doOutput = true
            conn.connectTimeout = 15000
            conn.readTimeout = 60000

            val rootJson = JSONObject().apply {
                put("model", model)
                put("stream", true)
                put("temperature", temperature)
                val msgsArr = JSONArray()
                for (m in messages) {
                    msgsArr.put(JSONObject().apply {
                        put("role", m.role)
                        put("content", m.content)
                    })
                }
                put("messages", msgsArr)
            }

            conn.outputStream.use { os ->
                os.write(rootJson.toString().toByteArray(Charsets.UTF_8))
            }

            val responseCode = conn.responseCode
            if (responseCode !in 200..299) {
                val err = conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "HTTP $responseCode"
                val cleanErr = try {
                    val errJson = JSONObject(err)
                    val errorObj = errJson.optJSONObject("error")
                    errorObj?.optString("message")
                        ?: errJson.optString("message").takeIf { it.isNotBlank() }
                        ?: err
                } catch (_: Exception) {
                    err
                }

                val tip = when (responseCode) {
                    401 -> "Неверный API-ключ для ${provider.displayName}."
                    403 -> when (provider) {
                        LLMProviderType.OPENROUTER -> "Доступ отклонен OpenRouter ($cleanErr). Проверьте ключ на openrouter.ai/keys."
                        LLMProviderType.GROQ -> "Доступ отклонен Groq Cloud ($cleanErr). Проверьте ключ на console.groq.com."
                        else -> if (isFreeModel(provider, model)) {
                            "Бесплатная промо-модель '$model' заблокирована шлюзом OpenCode для вашего платного API-ключа (HTTP 403).\n💡 Выберите стандартную модель из каталога (например, 'qwen3.6-plus', 'claude-3-7-sonnet', 'gpt-4o')."
                        } else {
                            "Доступ отклонен ($cleanErr). Проверьте активность подписки ${provider.displayName}."
                        }
                    }
                    404 -> if (provider == LLMProviderType.OPENCODE_ZEN && !isFreeModel(provider, model)) {
                        "Модель '$model' недоступна или маршрут не найден (HTTP 404).\n💡 Рекомендуем использовать проверенные рабочие модели: 'qwen3.6-plus', 'qwen3.8-flash', 'claude-3-7-sonnet', 'gpt-4o'."
                    } else {
                        "Модель '$model' не найдена или временно недоступна: $cleanErr"
                    }
                    429 -> "Превышен лимит запросов ${provider.displayName}. Попробуйте чуть позже."
                    else -> cleanErr
                }
                throw Exception("Ошибка ${provider.displayName} ($responseCode): $tip")
            }

            val reader = BufferedReader(InputStreamReader(conn.inputStream, Charsets.UTF_8))
            var line: String?
            var isThinking = false
            while (reader.readLine().also { line = it } != null) {
                if (isCancelled()) break
                val l = line?.trim() ?: continue
                if (l.startsWith("data: ")) {
                    val payload = l.removePrefix("data: ").trim()
                    if (payload == "[DONE]") break
                    try {
                        val chunkJson = JSONObject(payload)
                        val choices = chunkJson.optJSONArray("choices")
                        if (choices != null && choices.length() > 0) {
                            val delta = choices.getJSONObject(0).optJSONObject("delta")
                            val content = if (delta != null && !delta.isNull("content")) {
                                val s = delta.optString("content", "")
                                if (s == "null") "" else s
                            } else ""

                            val reasoning = if (delta != null && !delta.isNull("reasoning_content")) {
                                val s = delta.optString("reasoning_content", "")
                                if (s == "null") "" else s
                            } else ""

                            if (reasoning.isNotEmpty()) {
                                if (!isThinking) {
                                    isThinking = true
                                    onToken("💭 *Размышления:*\n")
                                }
                                onToken(reasoning)
                            } else if (content.isNotEmpty()) {
                                if (isThinking) {
                                    isThinking = false
                                    onToken("\n\n---\n\n")
                                }
                                onToken(content)
                            }
                        }
                    } catch (_: Exception) {}
                }
            }
        } finally {
            conn?.disconnect()
        }
    }
}
