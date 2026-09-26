package com.klischa.llmnotes.api

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.URL
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
 * Поддерживает подписки OpenCode GO и OpenCode ZEN, потоковый вывод ответов (SSE)
 * и получение списка доступных моделей.
 */
object OpenCodeClient {
    private const val TAG = "OpenCodeClient"

    // Рекомендуемые модели для подписки OpenCode GO (включает открытые и кодинг-модели)
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

    // Рекомендуемые модели для подписки OpenCode ZEN (включает флагманские модели frontier)
    val ZEN_DEFAULT_MODELS = listOf(
        "claude-3-7-sonnet",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
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
            conn.setRequestProperty("Authorization", "Bearer $apiKey")
            conn.setRequestProperty("Accept", "application/json")
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
     */
    fun streamChatCompletions(
        provider: LLMProviderType,
        apiKey: String,
        model: String,
        messages: List<ApiChatMessage>,
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
            conn.setRequestProperty("Authorization", "Bearer $apiKey")
            conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            conn.setRequestProperty("Accept", "text/event-stream")
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
                throw Exception("Ошибка ${provider.displayName} ($responseCode): $cleanErr")
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
                            if (content.isNotEmpty()) {
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
