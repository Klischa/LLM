package com.klischa.llmnotes

import android.app.Application
import android.content.ContentResolver
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

enum class MessageRole {
    USER,
    ASSISTANT
}

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: MessageRole,
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val tokensPerSec: Float? = null,
    val isStreaming: Boolean = false
)

enum class SystemPromptPreset(val title: String, val prompt: String) {
    UNIVERSAL(
        "Универсальный",
        "Ты — умный, точный и лаконичный русскоязычный персональный ассистент. " +
        "Твоя задача — отвечать на вопросы понятно и по существу, помогать работать с заметками, " +
        "делать емкие пересказы и структурировать информацию без лишней 'воды'."
    ),
    CONCISE(
        "Кратко",
        "Ты — редактор-аналитик. Отвечай предельно кратко, тезисно и строго по делу. " +
        "Выделяй только главные факты, даты и выводы. Никаких пустых приветствий и вводных слов."
    ),
    TASKS(
        "Задачи",
        "Ты — менеджер задач. Анализируй текст и формируй структурированный список конкретных действий " +
        "(Action Items) с чекбоксами [ ] и дедлайнами."
    ),
    DIALOG(
        "Диалог",
        "Ты — эрудированный, дружелюбный и внимательный собеседник. Отвечай развернуто, " +
        "живым языком, приводи примеры и рассуждай логично."
    )
}

data class UiState(
    val isModelLoaded: Boolean = false,
    val isLoadingModel: Boolean = false,
    val modelPath: String = "",
    val isGenerating: Boolean = false,
    val inputText: String = "",
    val messages: List<ChatMessage> = emptyList(),
    val systemPrompt: String = SystemPromptPreset.UNIVERSAL.prompt,
    val isSystemPromptExpanded: Boolean = false,
    val statusMessage: String = "Выберите модель GGUF в памяти смартфона",
    val tokensPerSecond: Float = 0.0f,
    val totalTokensGenerated: Int = 0,
    val generationTimeSeconds: Float = 0.0f,
    val allocatedRamMb: Long = 0,
    val threadCount: Int = 2, // Оптимум 2 ядра Cortex-A76 для Helio G99
    val temperature: Float = 0.3f,
    val topP: Float = 0.85f
)

class LLMViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var openPfd: ParcelFileDescriptor? = null

    companion object {
        private const val TAG = "LLMViewModel"
    }

    override fun onCleared() {
        super.onCleared()
        openPfd?.close()
        openPfd = null
        LlamaBridge.nativeUnload()
    }

    fun updateInputText(text: String) {
        _uiState.update { it.copy(inputText = text) }
    }

    fun updateSystemPrompt(newPrompt: String) {
        _uiState.update { it.copy(systemPrompt = newPrompt) }
    }

    fun applyPreset(preset: SystemPromptPreset) {
        _uiState.update { it.copy(systemPrompt = preset.prompt) }
    }

    fun toggleSystemPromptExpanded() {
        _uiState.update { it.copy(isSystemPromptExpanded = !it.isSystemPromptExpanded) }
    }

    fun setThreadCount(threads: Int) {
        _uiState.update { it.copy(threadCount = threads) }
    }

    fun setTemperature(temp: Float) {
        _uiState.update { it.copy(temperature = temp) }
    }

    /**
     * Загрузка модели из URI Android Storage Access Framework (SAF).
     * Приоритет отдается прямому чтению без копирования.
     */
    fun loadModelUri(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val contentResolver = getApplication<Application>().contentResolver
            val (displayName, fileSize) = queryUriMetadata(contentResolver, uri)

            _uiState.update {
                it.copy(
                    isLoadingModel = true,
                    statusMessage = "Подготовка файла модели ($displayName)..."
                )
            }

            openPfd?.close()
            openPfd = null

            // Способ 1: Прямой путь к файлу (если схема uri = file)
            if (uri.scheme == "file") {
                val directPath = uri.path
                if (directPath != null && File(directPath).canRead()) {
                    loadModelInternal(directPath, displayName)
                    return@launch
                }
            }

            // Способ 2: Открытие через File Descriptor (SAF proc/self/fd)
            try {
                val pfd = contentResolver.openFileDescriptor(uri, "r")
                if (pfd != null) {
                    val fd = pfd.fd
                    val fdPath = "/proc/self/fd/$fd"

                    val realPath = try {
                        android.system.Os.readlink(fdPath)
                    } catch (e: Exception) {
                        null
                    }

                    val pathToUse = if (realPath != null && File(realPath).canRead() && File(realPath).length() > 0) {
                        Log.i(TAG, "Определен прямой путь к файлу: $realPath")
                        pfd.close()
                        realPath
                    } else {
                        Log.i(TAG, "Используем дескриптор SAF: $fdPath")
                        openPfd = pfd
                        fdPath
                    }

                    _uiState.update {
                        it.copy(statusMessage = "Загрузка модели в память ($displayName)...")
                    }

                    val error = LlamaBridge.nativeLoadModel(
                        pathToUse,
                        2048,
                        _uiState.value.threadCount
                    )

                    if (error.isEmpty()) {
                        _uiState.update {
                            it.copy(
                                isModelLoaded = true,
                                isLoadingModel = false,
                                modelPath = displayName,
                                statusMessage = "Модель готова: $displayName"
                            )
                        }
                        updateRamUsage()
                        return@launch
                    } else {
                        Log.w(TAG, "Не удалось загрузить напрямую ($pathToUse): $error")
                        openPfd?.close()
                        openPfd = null

                        val isFatalModelError = error.contains("unknown model architecture") ||
                                                error.contains("unsupported") ||
                                                error.contains("нехватка памяти") ||
                                                error.contains("out of memory") ||
                                                error.contains("cannot allocate")

                        if (isFatalModelError) {
                            val friendlyError = formatLoadError(error)
                            _uiState.update {
                                it.copy(
                                    isModelLoaded = false,
                                    isLoadingModel = false,
                                    statusMessage = friendlyError
                                )
                            }
                            updateRamUsage()
                            return@launch
                        }

                        _uiState.update {
                            it.copy(
                                statusMessage = "Прямой доступ не удался. Проверка места для локального кэша..."
                            )
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Ошибка открытия FileDescriptor: ${e.message}")
            }

            // Способ 3: Кэширование во внутреннее хранилище приложения при необходимости
            val appFilesDir = getApplication<Application>().filesDir
            val freeBytes = appFilesDir.freeSpace
            val neededBytes = if (fileSize > 0) fileSize else 1500L * 1024 * 1024

            if (freeBytes < neededBytes + 150L * 1024 * 1024) {
                val freeMb = freeBytes / (1024 * 1024)
                val needMb = neededBytes / (1024 * 1024)
                _uiState.update {
                    it.copy(
                        isModelLoaded = false,
                        isLoadingModel = false,
                        statusMessage = "Прямой доступ отклонен ОС. Для кэша свободно $freeMb МБ, нужно $needMb МБ."
                    )
                }
                return@launch
            }

            val cacheFile = File(appFilesDir, "model.gguf")
            try {
                _uiState.update {
                    it.copy(statusMessage = "Копирование модели во внутренний кэш...")
                }

                contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(cacheFile).use { output ->
                        val buffer = ByteArray(1024 * 1024)
                        var bytesCopied = 0L
                        var lastReportTime = System.currentTimeMillis()

                        var read: Int
                        while (input.read(buffer).also { read = it } != -1) {
                            output.write(buffer, 0, read)
                            bytesCopied += read
                            val now = System.currentTimeMillis()
                            if (now - lastReportTime > 600) {
                                lastReportTime = now
                                val progress = if (fileSize > 0) " (${(bytesCopied * 100 / fileSize)}%)" else ""
                                _uiState.update {
                                    it.copy(statusMessage = "Копирование во внутренний кэш$progress...")
                                }
                            }
                        }
                    }
                }

                loadModelInternal(cacheFile.absolutePath, displayName)

            } catch (e: Exception) {
                if (cacheFile.exists()) cacheFile.delete()
                _uiState.update {
                    it.copy(
                        isModelLoaded = false,
                        isLoadingModel = false,
                        statusMessage = "Ошибка кэширования модели: ${e.message}"
                    )
                }
            }
        }
    }

    private fun loadModelInternal(path: String, displayName: String) {
        _uiState.update {
            it.copy(
                isLoadingModel = true,
                statusMessage = "Инициализация llama.cpp ($displayName)..."
            )
        }
        updateRamUsage()

        val error = LlamaBridge.nativeLoadModel(
            path,
            2048,
            _uiState.value.threadCount
        )

        if (error.isEmpty()) {
            _uiState.update {
                it.copy(
                    isModelLoaded = true,
                    isLoadingModel = false,
                    modelPath = displayName,
                    statusMessage = "Модель готова: $displayName"
                )
            }
        } else {
            val friendlyError = formatLoadError(error)
            _uiState.update {
                it.copy(
                    isModelLoaded = false,
                    isLoadingModel = false,
                    statusMessage = friendlyError
                )
            }
        }
        updateRamUsage()
    }

    private fun formatLoadError(error: String): String {
        return when {
            error.contains("unknown model architecture: 'qwen35'") || error.contains("'qwen35'") ->
                "Архитектура 'qwen35' (Qwen 3.5) содержит гибридные слои DeltaNet и не поддерживается мобильным llama.cpp. Используйте официальные модели линейки Qwen 2.5 (архитектура 'qwen2') или Llama-3.2."
            error.contains("unknown model architecture") ->
                "Неподдерживаемая архитектура ($error). Поддерживаются архитектуры: Qwen 2.5 (qwen2), LLaMA 3/3.2 (llama), Gemma 2 (gemma2), Mistral, Phi-3."
            error.contains("недостаточно памяти") || error.contains("out of memory") ->
                "Недостаточно RAM для модели. Выберите модель 1.5B или закройте тяжелые приложения."
            else -> error
        }
    }

    fun loadModel(filePath: String) {
        val fileName = File(filePath).name
        viewModelScope.launch(Dispatchers.IO) {
            loadModelInternal(filePath, fileName)
        }
    }

    private fun queryUriMetadata(contentResolver: ContentResolver, uri: Uri): Pair<String, Long> {
        var name = "model.gguf"
        var size = 0L
        try {
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (cursor.moveToFirst()) {
                    if (nameIndex != -1) cursor.getString(nameIndex)?.let { name = it }
                    if (sizeIndex != -1) size = cursor.getLong(sizeIndex)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Ошибка чтения метаданных: ${e.message}")
        }
        return Pair(name, size)
    }

    /**
     * Отправка сообщения в чат с сохранением многошаговой истории диалога (Multi-turn chat).
     */
    fun sendMessage(userText: String = _uiState.value.inputText) {
        val prompt = userText.trim()
        if (prompt.isEmpty() || !_uiState.value.isModelLoaded || _uiState.value.isGenerating) return

        val userMessage = ChatMessage(
            role = MessageRole.USER,
            text = prompt
        )
        val assistantMessage = ChatMessage(
            role = MessageRole.ASSISTANT,
            text = "",
            isStreaming = true
        )

        val previousMessages = _uiState.value.messages
        val updatedMessages = previousMessages + userMessage + assistantMessage
        val assistantMsgId = assistantMessage.id

        _uiState.update {
            it.copy(
                inputText = "",
                messages = updatedMessages,
                isGenerating = true,
                statusMessage = "Генерация ответа...",
                tokensPerSecond = 0.0f,
                totalTokensGenerated = 0
            )
        }

        // Собираем историю диалога для модели (последние 10 сообщений для экономии контекста)
        val historyLimit = 10
        val historyForPrompt = previousMessages.takeLast(historyLimit)

        val rolesList = mutableListOf<String>()
        val contentsList = mutableListOf<String>()

        // 1. Системный промпт
        if (_uiState.value.systemPrompt.isNotBlank()) {
            rolesList.add("system")
            contentsList.add(_uiState.value.systemPrompt.trim())
        }

        // 2. История предыдущих реплик
        for (msg in historyForPrompt) {
            if (msg.role == MessageRole.USER) {
                rolesList.add("user")
                contentsList.add(msg.text)
            } else if (msg.role == MessageRole.ASSISTANT && msg.text.isNotBlank()) {
                rolesList.add("assistant")
                contentsList.add(msg.text)
            }
        }

        // 3. Текущее сообщение пользователя
        rolesList.add("user")
        contentsList.add(prompt)

        val formattedPrompt = LlamaBridge.nativeFormatChat(
            rolesList.toTypedArray(),
            contentsList.toTypedArray()
        )

        viewModelScope.launch(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()
            var tokenCount = 0
            val responseBuilder = StringBuilder()

            val callback = LlamaBridge.TokenCallback { tokenPiece ->
                tokenCount++
                responseBuilder.append(tokenPiece)
                val currentText = responseBuilder.toString()
                val elapsedSec = (System.currentTimeMillis() - startTime) / 1000f
                val speed = if (elapsedSec > 0.05f) tokenCount / elapsedSec else 0.0f

                _uiState.update { current ->
                    val msgs = current.messages.map { m ->
                        if (m.id == assistantMsgId) {
                            m.copy(text = currentText, tokensPerSec = speed, isStreaming = true)
                        } else {
                            m
                        }
                    }
                    current.copy(
                        messages = msgs,
                        totalTokensGenerated = tokenCount,
                        tokensPerSecond = speed,
                        generationTimeSeconds = elapsedSec
                    )
                }
            }

            LlamaBridge.nativeGenerate(
                formattedPrompt,
                1024,
                _uiState.value.temperature,
                _uiState.value.topP,
                callback
            )

            val totalElapsed = (System.currentTimeMillis() - startTime) / 1000f
            val finalSpeed = if (totalElapsed > 0.05f) tokenCount / totalElapsed else 0.0f

            _uiState.update { current ->
                val msgs = current.messages.map { m ->
                    if (m.id == assistantMsgId) {
                        m.copy(
                            text = responseBuilder.toString(),
                            tokensPerSec = finalSpeed,
                            isStreaming = false
                        )
                    } else {
                        m
                    }
                }
                current.copy(
                    messages = msgs,
                    isGenerating = false,
                    statusMessage = "Готово. Токенов: $tokenCount (~${String.format("%.1f", finalSpeed)} tok/s)",
                    tokensPerSecond = finalSpeed,
                    generationTimeSeconds = totalElapsed
                )
            }
            updateRamUsage()
        }
    }

    fun clearChat() {
        if (_uiState.value.isGenerating) {
            stopGeneration()
        }
        _uiState.update {
            it.copy(
                messages = emptyList(),
                statusMessage = if (it.isModelLoaded) "Диалог очищен" else "Выберите модель GGUF в памяти смартфона"
            )
        }
    }

    fun insertNotePrompt(type: String) {
        val prefix = when (type) {
            "summary" -> "Сделай краткое структурированное резюме следующего текста:\n\n"
            "tasks" -> "Выдели четкий список задач (Action Items) с чекбоксами [ ] из следующего текста:\n\n"
            "format" -> "Отредактируй и красиво структурируй текст в формате Markdown:\n\n"
            "explain" -> "Объясни простыми словами следующую тему:\n\n"
            else -> ""
        }
        _uiState.update {
            it.copy(inputText = prefix + it.inputText)
        }
    }

    fun stopGeneration() {
        LlamaBridge.nativeStop()
        _uiState.update { current ->
            val msgs = current.messages.map { m ->
                if (m.isStreaming) m.copy(isStreaming = false) else m
            }
            current.copy(
                messages = msgs,
                isGenerating = false,
                statusMessage = "Генерация остановлена пользователем"
            )
        }
    }

    fun unloadModel() {
        LlamaBridge.nativeUnload()
        openPfd?.close()
        openPfd = null
        _uiState.update {
            it.copy(
                isModelLoaded = false,
                isLoadingModel = false,
                modelPath = "",
                statusMessage = "Модель выгружена из памяти"
            )
        }
        updateRamUsage()
    }

    private fun updateRamUsage() {
        val runtime = Runtime.getRuntime()
        val usedMem = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        _uiState.update { it.copy(allocatedRamMb = usedMem) }
    }
}
