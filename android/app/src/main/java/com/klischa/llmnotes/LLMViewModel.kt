package com.klischa.llmnotes

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class UiState(
    val isModelLoaded: Boolean = false,
    val modelPath: String = "",
    val isGenerating: Boolean = false,
    val inputText: String = "",
    val outputText: String = "",
    val statusMessage: String = "Выберите модель GGUF в памяти смартфона",
    val tokensPerSecond: Float = 0.0f,
    val totalTokensGenerated: Int = 0,
    val generationTimeSeconds: Float = 0.0f,
    val allocatedRamMb: Long = 0,
    val threadCount: Int = 2, // Оптимум 2 ядра A76 для Helio G99
    val temperature: Float = 0.3f,
    val topP: Float = 0.85f,
    val selectedTab: Int = 0 // 0: Заметки и саммари, 1: Вопрос-ответ
)

class LLMViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val systemPrompt = (
        "Ты — умный и лаконичный русскоязычный офлайн-ассистент для смартфона Infinix Note 30. " +
        "Твоя задача — помогать пользователю работать с заметками, отвечать на вопросы точно и без " +
        "лишней 'воды', делать краткие пересказы и структурировать информацию."
    )

    fun updateInputText(text: String) {
        _uiState.update { it.copy(inputText = text) }
    }

    fun setSelectedTab(index: Int) {
        _uiState.update { it.copy(selectedTab = index) }
    }

    fun setThreadCount(threads: Int) {
        _uiState.update { it.copy(threadCount = threads) }
    }

    fun setTemperature(temp: Float) {
        _uiState.update { it.copy(temperature = temp) }
    }

    fun loadModel(filePath: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.update { it.copy(statusMessage = "Загрузка модели в память...") }
            updateRamUsage()

            val success = LlamaBridge.nativeLoadModel(
                filePath,
                2048,
                _uiState.value.threadCount
            )

            if (success) {
                _uiState.update {
                    it.copy(
                        isModelLoaded = true,
                        modelPath = filePath,
                        statusMessage = "Модель готова к работе (потоков: ${it.threadCount})"
                    )
                }
            } else {
                _uiState.update {
                    it.copy(
                        isModelLoaded = false,
                        statusMessage = "Ошибка при загрузке GGUF модели"
                    )
                }
            }
            updateRamUsage()
        }
    }

    fun generateSummary() {
        val note = _uiState.value.inputText.trim()
        if (note.isEmpty()) return
        val userPrompt = "Сделай краткое резюме заметки:\n\n$note"
        runPrompt(userPrompt)
    }

    fun extractActionItems() {
        val note = _uiState.value.inputText.trim()
        if (note.isEmpty()) return
        val userPrompt = "Выдели четкий список задач (Action Items) с чекбоксами из следующего текста:\n\n$note"
        runPrompt(userPrompt)
    }

    fun formatAndClean() {
        val note = _uiState.value.inputText.trim()
        if (note.isEmpty()) return
        val userPrompt = "Отредактируй и красиво структурируй текст заметки в Markdown:\n\n$note"
        runPrompt(userPrompt)
    }

    fun askQuestion() {
        val q = _uiState.value.inputText.trim()
        if (q.isEmpty()) return
        runPrompt(q)
    }

    private fun runPrompt(userPrompt: String) {
        if (!_uiState.value.isModelLoaded || _uiState.value.isGenerating) return

        // Формирование ChatML промпта под формат Qwen2.5
        val chatMlPrompt = buildString {
            append("<|im_start|>system\n")
            append(systemPrompt)
            append("<|im_end|>\n<|im_start|>user\n")
            append(userPrompt)
            append("<|im_end|>\n<|im_start|>assistant\n")
        }

        viewModelScope.launch(Dispatchers.Default) {
            _uiState.update {
                it.copy(
                    isGenerating = true,
                    outputText = "",
                    statusMessage = "Генерация ответа...",
                    tokensPerSecond = 0.0f,
                    totalTokensGenerated = 0
                )
            }

            val startTime = System.currentTimeMillis()
            var tokenCount = 0

            val callback = LlamaBridge.TokenCallback { tokenPiece ->
                tokenCount++
                val elapsedSec = (System.currentTimeMillis() - startTime) / 1000f
                val speed = if (elapsedSec > 0.05f) tokenCount / elapsedSec else 0.0f

                _uiState.update { current ->
                    current.copy(
                        outputText = current.outputText + tokenPiece,
                        totalTokensGenerated = tokenCount,
                        tokensPerSecond = speed,
                        generationTimeSeconds = elapsedSec
                    )
                }
            }

            LlamaBridge.nativeGenerate(
                chatMlPrompt,
                512,
                _uiState.value.temperature,
                _uiState.value.topP,
                callback
            )

            val totalElapsed = (System.currentTimeMillis() - startTime) / 1000f
            val finalSpeed = if (totalElapsed > 0.05f) tokenCount / totalElapsed else 0.0f

            _uiState.update {
                it.copy(
                    isGenerating = false,
                    statusMessage = "Готово. Токенов: $tokenCount (~${String.format("%.1f", finalSpeed)} tok/s)",
                    tokensPerSecond = finalSpeed,
                    generationTimeSeconds = totalElapsed
                )
            }
            updateRamUsage()
        }
    }

    fun stopGeneration() {
        LlamaBridge.nativeStop()
        _uiState.update {
            it.copy(
                isGenerating = false,
                statusMessage = "Генерация остановлена пользователем"
            )
        }
    }

    fun unloadModel() {
        LlamaBridge.nativeUnload()
        _uiState.update {
            it.copy(
                isModelLoaded = false,
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
