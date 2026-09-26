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
 * - LOCAL_GGUF: локальный автономный запуск на процессоре Helio G99 через llama.cpp.
 * - OPENCODE_GO: облачный API OpenCode подписки GO (https://opencode.ai/zen/go/v1).
 * - OPENCODE_ZEN: облачный API OpenCode подписки ZEN (https://opencode.ai/zen/v1).
 */
enum class LLMProviderType(val displayName: String, val baseUrl: String) {
    LOCAL_GGUF("Офлайн GGUF", ""),
    OPENCODE_GO("OpenCode GO", "https://opencode.ai/zen/go/v1"),
    OPENCODE_ZEN("OpenCode ZEN", "https://opencode.ai/zen/v1")
}

data class ApiChatMessage(
    val role: String,
    val content: String
)

/**
 * Клиент взаимодействия с внешним OpenAI-совместимым API платформы OpenCode.
 * Отправляет обязательные клиентские заголовки OpenCode (User-Agent, x-opencode-client,
 * x-opencode-session, x-session-id), необходимые для авторизации сессий и бесплатного тарифа.
 */
object OpenCodeClient {
    private const val TAG = "OpenCodeClient"
    private const val OPENCODE_USER_AGENT = "opencode/1.18.31 ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14"
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
        "big-pickle",
        "claude-fable-5-1",
        "claude-haiku-4-5",
        "claude-3-7-sonnet",
        "claude-3-5-sonnet",
        "gpt-4o",
        "gpt-4o-mini",
        "deepseek-r1",
        "deepseek-v4-pro",
        "gemini-2.0-flash",
        "kimi-k2.6",
        "qwen-2.5-max"
    )

    fun getDefaultModels(provider: LLMProviderType): List<String> {
        return when (provider) {
            LLMProviderType.OPENCODE_GO -> GO_DEFAULT_MODELS
            LLMProviderType.OPENCODE_ZEN -> ZEN_DEFAULT_MODELS
            else -> emptyList()
        }
    }

    /**
     * Запрос актуального каталога моделей с сервера OpenCode (GET /v1/models).
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
            val sessionId = generateSessionId()
            val requestId = generateRequestId()

            conn.setRequestProperty("Authorization", "Bearer $rawKey")
            conn.setRequestProperty("x-api-key", rawKey)
            conn.setRequestProperty("Accept", "application/json")
            conn.setRequestProperty("User-Agent", OPENCODE_USER_AGENT)
            conn.setRequestProperty("x-opencode-client", "cli")
            conn.setRequestProperty("x-opencode-project", "global")
            conn.setRequestProperty("x-opencode-session", sessionId)
            conn.setRequestProperty("x-opencode-request", requestId)
            conn.connectTimeout = 10000
            conn.readTimeout = 15000

            val responseCode = conn.responseCode
            if (responseCode !in 200..299) {
                val err = conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "HTTP $responseCode"
                Log.w(TAG, "Ошибка fetchModels ($responseCode): $err")
                return getDefaultModels(provider)
            }

            val body = conn.inputStream.bufferedReader().use { it.readText() }
            val json = JSONObject(body)
            val dataArray = json.optJSONArray("data")
            val list = mutableListOf<String>()

            if (dataArray != null) {
                for (i in 0 until dataArray.length()) {
                    val item = dataArray.optJSONObject(i)
                    val id = item?.optString("id") ?: ""
                    if (id.isNotBlank()) {
                        list.add(id)
                    }
                }
            }
            return if (list.isNotEmpty()) list.sorted() else getDefaultModels(provider)
        } catch (e: Exception) {
            Log.w(TAG, "Ошибка получения моделей OpenCode: ${e.message}")
            return getDefaultModels(provider)
        } finally {
            conn?.disconnect()
        }
    }

    /**
     * Потоковая генерация ответа через Server-Sent Events (SSE).
     * Эмулирует официальный OpenCode CLI для снятия ограничений Free Tier.
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

            val sessionId = generateSessionId()
            val requestId = generateRequestId()
            val rawKey = apiKey.removePrefix("Bearer ").removePrefix("bearer ").trim()

            conn.setRequestProperty("Authorization", "Bearer $rawKey")
            conn.setRequestProperty("x-api-key", rawKey)
            conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            conn.setRequestProperty("Accept", "text/event-stream")

            // Точные заголовки OpenCode CLI для авторизации шлюза и Free Tier
            conn.setRequestProperty("User-Agent", OPENCODE_USER_AGENT)
            conn.setRequestProperty("x-opencode-client", "cli")
            conn.setRequestProperty("x-opencode-project", "global")
            conn.setRequestProperty("x-opencode-session", sessionId)
            conn.setRequestProperty("x-opencode-request", requestId)
            conn.setRequestProperty("x-session-id", sessionId)
            conn.setRequestProperty("x-session-affinity", sessionId)

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
                    errJson.optJSONObject("error")?.optString("message", err) ?: err
                } catch (_: Exception) {
                    err
                }

                val tip = when (responseCode) {
                    401 -> "Неверный API-ключ. Проверьте, что ключ подходит для выбранной подписки (${provider.displayName})."
                    403 -> "Доступ отклонен ($cleanErr). Проверьте активность подписки ${provider.displayName}."
                    else -> cleanErr
                }
                throw Exception("Ошибка ${provider.displayName} ($responseCode): $tip")
            }

            val reader = BufferedReader(InputStreamReader(conn.inputStream, Charsets.UTF_8))
            var line: String?
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
                            val content = delta?.optString("content", "") ?: ""
                            val reasoning = delta?.optString("reasoning_content", "") ?: ""
                            val tokenPiece = if (content.isNotEmpty()) content else reasoning
                            if (tokenPiece.isNotEmpty()) {
                                onToken(tokenPiece)
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
