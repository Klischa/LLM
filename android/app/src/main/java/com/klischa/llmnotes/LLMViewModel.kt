package com.klischa.llmnotes

import android.app.Application
import android.content.ContentResolver
import android.content.Context
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
    val currentChatId: String = "",
    val chatSessions: List<com.klischa.llmnotes.data.ChatSession> = emptyList(),
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

    private val dbHelper = com.klischa.llmnotes.data.ChatDatabaseHelper(application)
    private var openPfd: ParcelFileDescriptor? = null

    companion object {
        private const val TAG = "LLMViewModel"
        private const val PREFS_NAME = "llm_app_prefs"
        private const val PREF_LAST_MODEL_PATH = "last_model_path"
        private const val PREF_LAST_MODEL_URI = "last_model_uri"
        private const val PREF_LAST_MODEL_NAME = "last_model_name"
        private const val PREF_AUTO_LOAD_ENABLED = "auto_load_enabled"
    }

    fun saveLastModelInfo(directPath: String?, uriStr: String?, displayName: String) {
        val prefs = getApplication<Application>().getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().apply {
            if (directPath != null) {
                putString(PREF_LAST_MODEL_PATH, directPath)
            }
            if (uriStr != null) {
                putString(PREF_LAST_MODEL_URI, uriStr)
            }
            putString(PREF_LAST_MODEL_NAME, displayName)
            putBoolean(PREF_AUTO_LOAD_ENABLED, true)
            apply()
        }
    }

    fun tryAutoLoadLastModel() {
        if (_uiState.value.isModelLoaded || _uiState.value.isLoadingModel) return

        val prefs = getApplication<Application>().getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val autoLoad = prefs.getBoolean(PREF_AUTO_LOAD_ENABLED, true)
        if (!autoLoad) return

        val lastPath = prefs.getString(PREF_LAST_MODEL_PATH, null)
        val lastUriStr = prefs.getString(PREF_LAST_MODEL_URI, null)
        val lastName = prefs.getString(PREF_LAST_MODEL_NAME, "модели") ?: "модели"

        // Приоритет 1: сохраненный прямой путь к файлу (например /sdcard/Download/...)
        if (!lastPath.isNullOrBlank()) {
            val file = File(lastPath)
            if (file.exists() && file.canRead() && file.length() > 0) {
                Log.i(TAG, "Автозагрузка сохраненной модели по пути: $lastPath")
                _uiState.update {
                    it.copy(
                        isLoadingModel = true,
                        statusMessage = "Автозагрузка модели ($lastName)..."
                    )
                }
                loadModel(lastPath)
                return
            }
        }

        // Приоритет 2: сохраненный URI SAF
        if (!lastUriStr.isNullOrBlank()) {
            try {
                val uri = Uri.parse(lastUriStr)
                Log.i(TAG, "Автозагрузка сохраненной модели по URI: $lastUriStr")
                _uiState.update {
                    it.copy(
                        isLoadingModel = true,
                        statusMessage = "Автозагрузка модели ($lastName)..."
                    )
                }
                loadModelUri(uri)
                return
            } catch (e: Exception) {
                Log.w(TAG, "Не удалось выполнить автозагрузку по URI: ${e.message}")
            }
        }

        // Приоритет 3: поиск во внутренней папке models
        val modelsDir = File(getApplication<Application>().filesDir, "models")
        if (modelsDir.exists()) {
            val cachedFile = modelsDir.listFiles()?.firstOrNull { it.extension.lowercase() == "gguf" && it.length() > 0 }
            if (cachedFile != null) {
                Log.i(TAG, "Автозагрузка модели из внутренней папки models: ${cachedFile.absolutePath}")
                _uiState.update {
                    it.copy(
                        isLoadingModel = true,
                        statusMessage = "Автозагрузка модели (${cachedFile.name})..."
                    )
                }
                loadModel(cachedFile.absolutePath)
            }
        }
    }

    init {
        loadChatSessions()
    }

    private fun loadChatSessions() {
        val sessions = dbHelper.getAllSessions()
        if (sessions.isEmpty()) {
            val newId = UUID.randomUUID().toString()
            val newSession = dbHelper.createSession(newId, "Новый чат")
            _uiState.update {
                it.copy(
                    currentChatId = newId,
                    chatSessions = listOf(newSession),
                    messages = emptyList()
                )
            }
        } else {
            val current = sessions.first()
            val msgs = dbHelper.getMessages(current.id)
            _uiState.update {
                it.copy(
                    currentChatId = current.id,
                    chatSessions = sessions,
                    messages = msgs
                )
            }
        }
    }

    fun createNewChat() {
        if (_uiState.value.isGenerating) {
            stopGeneration()
        }
        val newId = UUID.randomUUID().toString()
        val newSession = dbHelper.createSession(newId, "Новый чат")
        val updatedList = listOf(newSession) + _uiState.value.chatSessions.filter { it.id != newId }
        _uiState.update {
            it.copy(
                currentChatId = newId,
                chatSessions = updatedList,
                messages = emptyList(),
                statusMessage = if (it.isModelLoaded) "Новый диалог создан" else it.statusMessage
            )
        }
    }

    fun selectChat(chatId: String) {
        if (chatId == _uiState.value.currentChatId) return
        if (_uiState.value.isGenerating) {
            stopGeneration()
        }
        val msgs = dbHelper.getMessages(chatId)
        _uiState.update {
            it.copy(
                currentChatId = chatId,
                messages = msgs,
                statusMessage = if (it.isModelLoaded) "Диалог загружен" else it.statusMessage
            )
        }
    }

    fun deleteChat(chatId: String) {
        if (_uiState.value.isGenerating && chatId == _uiState.value.currentChatId) {
            stopGeneration()
        }
        dbHelper.deleteSession(chatId)
        val sessions = dbHelper.getAllSessions()
        if (sessions.isEmpty()) {
            createNewChat()
        } else {
            val nextSession = sessions.first()
            val msgs = dbHelper.getMessages(nextSession.id)
            _uiState.update {
                it.copy(
                    currentChatId = nextSession.id,
                    chatSessions = sessions,
                    messages = msgs
                )
            }
        }
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
                        val direct = if (File(pathToUse).exists() && File(pathToUse).length() > 0) pathToUse else null
                        saveLastModelInfo(direct, uri.toString(), displayName)
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

            val modelsDir = File(appFilesDir, "models").apply { mkdirs() }
            val cacheFile = File(modelsDir, displayName.ifBlank { "model.gguf" })
            try {
                _uiState.update {
                    it.copy(statusMessage = "Копирование модели в папку models...")
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
            saveLastModelInfo(path, null, displayName)
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

        val currentChatId = _uiState.value.currentChatId.ifBlank {
            val newId = UUID.randomUUID().toString()
            dbHelper.createSession(newId, "Новый чат")
            newId
        }

        // Сохраняем сообщение пользователя в БД
        dbHelper.saveMessage(currentChatId, userMessage)

        // Обновляем заголовок чата, если он еще стандартный
        val currentSession = _uiState.value.chatSessions.find { it.id == currentChatId }
        if (currentSession == null || currentSession.title == "Новый чат") {
            val newTitle = prompt.replace("\n", " ").trim().take(30) + if (prompt.length > 30) "..." else ""
            dbHelper.updateSessionTitle(currentChatId, newTitle)
            val updatedSessions = dbHelper.getAllSessions()
            _uiState.update { it.copy(chatSessions = updatedSessions) }
        }

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

        // 1. Системный промпт и контекст телеметрии устройства (Device Tools)
        val baseSysPrompt = _uiState.value.systemPrompt.trim()
        val telemetryContext = if (DeviceTools.isDeviceQuery(prompt)) {
            try {
                DeviceTools.buildTelemetryContext(getApplication())
            } catch (e: Exception) {
                Log.w(TAG, "Ошибка сбора телеметрии: ${e.message}")
                ""
            }
        } else {
            ""
        }

        val effectiveSysPrompt = when {
            baseSysPrompt.isNotEmpty() && telemetryContext.isNotEmpty() -> "$baseSysPrompt\n\n$telemetryContext"
            telemetryContext.isNotEmpty() -> telemetryContext
            else -> baseSysPrompt
        }

        if (effectiveSysPrompt.isNotBlank()) {
            rolesList.add("system")
            contentsList.add(effectiveSysPrompt)
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
            val finalText = responseBuilder.toString()

            val completedMsg = ChatMessage(
                id = assistantMsgId,
                role = MessageRole.ASSISTANT,
                text = finalText,
                tokensPerSec = finalSpeed,
                isStreaming = false
            )
            dbHelper.saveMessage(currentChatId, completedMsg)
            val refreshedSessions = dbHelper.getAllSessions()

            _uiState.update { current ->
                val msgs = current.messages.map { m ->
                    if (m.id == assistantMsgId) {
                        completedMsg
                    } else {
                        m
                    }
                }
                current.copy(
                    messages = msgs,
                    chatSessions = refreshedSessions,
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
        val currentId = _uiState.value.currentChatId
        if (currentId.isNotBlank()) {
            dbHelper.deleteSession(currentId)
            val newSession = dbHelper.createSession(currentId, "Новый чат")
            val refreshed = dbHelper.getAllSessions()
            _uiState.update {
                it.copy(
                    chatSessions = refreshed,
                    messages = emptyList(),
                    statusMessage = if (it.isModelLoaded) "Диалог очищен" else "Выберите модель GGUF в памяти смартфона"
                )
            }
        } else {
            _uiState.update {
                it.copy(
                    messages = emptyList(),
                    statusMessage = if (it.isModelLoaded) "Диалог очищен" else "Выберите модель GGUF в памяти смартфона"
                )
            }
        }
    }

    fun insertNotePrompt(type: String) {
        val query = when (type) {
            "photos" -> "Сколько фото и видео сохранено на моем телефоне?"
            "storage" -> "Сколько свободно памяти на моем телефоне?"
            "battery" -> "Какой текущий уровень заряда аккумулятора и температура?"
            "device" -> "Какие характеристики, модель и параметры у моего телефона?"
            "summary" -> "Сделай краткое структурированное резюме следующего текста:\n\n"
            "tasks" -> "Выдели четкий список задач (Action Items) с чекбоксами [ ] из следующего текста:\n\n"
            "format" -> "Отредактируй и красиво структурируй текст в формате Markdown:\n\n"
            "explain" -> "Объясни простыми словами следующую тему:\n\n"
            else -> ""
        }

        if (type in listOf("photos", "storage", "battery", "device")) {
            if (_uiState.value.inputText.isBlank() && _uiState.value.isModelLoaded && !_uiState.value.isGenerating) {
                sendMessage(query)
            } else {
                _uiState.update { it.copy(inputText = query) }
            }
        } else {
            _uiState.update {
                it.copy(inputText = query + it.inputText)
            }
        }
    }

    fun stopGeneration() {
        LlamaBridge.nativeStop()
        val currentId = _uiState.value.currentChatId
        _uiState.update { current ->
            val msgs = current.messages.map { m ->
                if (m.isStreaming) {
                    val stopped = m.copy(isStreaming = false)
                    if (currentId.isNotBlank() && stopped.text.isNotBlank()) {
                        dbHelper.saveMessage(currentId, stopped)
                    }
                    stopped
                } else {
                    m
                }
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

        val prefs = getApplication<Application>().getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putBoolean(PREF_AUTO_LOAD_ENABLED, false).apply()

        _uiState.update {
            it.copy(
                isModelLoaded = false,
                isLoadingModel = false,
                modelPath = "",
                statusMessage = "Модель выгружена из памяти",
                tokensPerSecond = 0.0f,
                allocatedRamMb = 0
            )
        }
        updateRamUsage()
    }

    private fun updateRamUsage() {
        val rssMb = try {
            File("/proc/self/status").useLines { lines ->
                val line = lines.firstOrNull { it.startsWith("VmRSS:") }
                line?.split("\\s+".toRegex())?.getOrNull(1)?.toLongOrNull()?.div(1024)
            }
        } catch (e: Exception) {
            null
        }

        val fallbackMem = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024)
        val finalMb = rssMb ?: fallbackMem
        _uiState.update { it.copy(allocatedRamMb = finalMb) }
    }
}
