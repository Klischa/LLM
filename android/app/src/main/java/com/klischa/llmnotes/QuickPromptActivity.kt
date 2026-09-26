@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.klischa.llmnotes

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.lifecycleScope
import com.klischa.llmnotes.api.ApiChatMessage
import com.klischa.llmnotes.api.LLMProviderType
import com.klischa.llmnotes.api.OpenCodeClient
import com.klischa.llmnotes.ui.LLMNotesTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Компактное окно быстрого ввода запроса (Quick Prompt) поверх рабочего стола.
 * Вызывается по клику на виджет, генерирует ответ и обновляет виджет на рабочем столе.
 */
class QuickPromptActivity : ComponentActivity() {

    private var attachedFileName by mutableStateOf<String?>(null)
    private var attachedFileContent by mutableStateOf<String?>(null)
    private var attachedFileSize by mutableStateOf<Long>(0L)

    private val openAttachmentLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let {
            try {
                var displayName = "file.txt"
                var size = 0L
                contentResolver.query(it, null, null, null, null)?.use { cursor ->
                    val nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    val sizeIdx = cursor.getColumnIndex(OpenableColumns.SIZE)
                    if (cursor.moveToFirst()) {
                        if (nameIdx != -1) displayName = cursor.getString(nameIdx) ?: displayName
                        if (sizeIdx != -1) size = cursor.getLong(sizeIdx)
                    }
                }

                val bytes = contentResolver.openInputStream(it)?.use { input -> input.readBytes() } ?: ByteArray(0)
                val isBinary = bytes.take(1024).any { b -> b == 0.toByte() }
                if (isBinary) {
                    Toast.makeText(this, "⚠️ Бинарный файл не поддерживается для анализа", Toast.LENGTH_LONG).show()
                } else {
                    val text = String(bytes, Charsets.UTF_8).take(25000)
                    attachedFileName = displayName
                    attachedFileSize = size
                    attachedFileContent = text
                    Toast.makeText(this, "📎 Файл '$displayName' прикреплен", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Toast.makeText(this, "Ошибка чтения файла: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            LLMNotesTheme {
                QuickPromptDialogContent(
                    onDismiss = { finish() },
                    onAttachClick = {
                        openAttachmentLauncher.launch(arrayOf("*/*"))
                    },
                    attachedName = attachedFileName,
                    attachedSize = attachedFileSize,
                    onRemoveAttachment = {
                        attachedFileName = null
                        attachedFileContent = null
                        attachedFileSize = 0L
                    },
                    getAttachedContent = { attachedFileContent },
                    onOpenFullApp = { query ->
                        val intent = Intent(this, MainActivity::class.java).apply {
                            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
                        }
                        startActivity(intent)
                        finish()
                    }
                )
            }
        }
    }
}

@Composable
fun QuickPromptDialogContent(
    onDismiss: () -> Unit,
    onAttachClick: () -> Unit,
    attachedName: String?,
    attachedSize: Long,
    onRemoveAttachment: () -> Unit,
    getAttachedContent: () -> String?,
    onOpenFullApp: (String) -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var inputText by remember { mutableStateOf("") }
    var responseText by remember { mutableStateOf("") }
    var isGenerating by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("") }
    var currentJob by remember { mutableStateOf<Job?>(null) }

    // Чтение настроек провайдера из SharedPreferences
    val prefs = remember { context.getSharedPreferences("llm_app_prefs", Context.MODE_PRIVATE) }
    val providerStr = prefs.getString("provider_type", LLMProviderType.LOCAL_GGUF.name)
    val provider = try { LLMProviderType.valueOf(providerStr ?: "") } catch (_: Exception) { LLMProviderType.LOCAL_GGUF }

    val modelName = when (provider) {
        LLMProviderType.OPENROUTER -> prefs.getString("openrouter_model", "google/gemini-2.0-flash-exp:free") ?: "OpenRouter Free"
        LLMProviderType.GROQ -> prefs.getString("groq_model", "llama-3.3-70b-versatile") ?: "Groq Free"
        LLMProviderType.OPENCODE_ZEN -> prefs.getString("opencode_model_zen", "qwen3.6-plus") ?: "OpenCode ZEN"
        LLMProviderType.OPENCODE_GO -> prefs.getString("opencode_model_go", "deepseek-v4-pro") ?: "OpenCode GO"
        LLMProviderType.LOCAL_GGUF -> prefs.getString("last_model_name", "Офлайн GGUF") ?: "Офлайн GGUF"
    }

    val apiKey = when (provider) {
        LLMProviderType.OPENROUTER -> prefs.getString("openrouter_key", "") ?: ""
        LLMProviderType.GROQ -> prefs.getString("groq_key", "") ?: ""
        LLMProviderType.OPENCODE_ZEN -> prefs.getString("opencode_zen_key", "") ?: ""
        LLMProviderType.OPENCODE_GO -> prefs.getString("opencode_go_key", "") ?: ""
        LLMProviderType.LOCAL_GGUF -> ""
    }

    // Извлеченные блоки кода для кнопок скачивания и копирования
    val codeBlocks = remember(responseText) {
        FileStorageManager.extractCodeBlocks(responseText)
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth(0.94f)
                .padding(vertical = 24.dp),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                // Шапка диалога
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_widget_sparkle),
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Column {
                            Text(
                                text = "Быстрый запрос",
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                            )
                            Text(
                                text = "${provider.displayName} • $modelName",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary,
                                maxLines = 1
                            )
                        }
                    }
                    IconButton(onClick = onDismiss, modifier = Modifier.size(32.dp)) {
                        Icon(Icons.Default.Close, contentDescription = "Закрыть")
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Быстрые чипы телеметрии
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    SuggestionChip(
                        onClick = { inputText = "Сколько фото сохранено на моем телефоне?" },
                        label = { Text("📸 Фото", fontSize = 11.sp) }
                    )
                    SuggestionChip(
                        onClick = { inputText = "Сколько свободно памяти на телефоне?" },
                        label = { Text("💾 Память", fontSize = 11.sp) }
                    )
                    SuggestionChip(
                        onClick = { inputText = "Какой заряд аккумулятора?" },
                        label = { Text("🔋 Батарея", fontSize = 11.sp) }
                    )
                }

                // Превью прикрепленного файла
                if (attachedName != null) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Surface(
                        color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.6f),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    painter = painterResource(id = R.drawable.ic_attach),
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(16.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = "$attachedName (${attachedSize / 1024} КБ)",
                                    style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium),
                                    maxLines = 1
                                )
                            }
                            IconButton(onClick = onRemoveAttachment, modifier = Modifier.size(20.dp)) {
                                Icon(Icons.Default.Close, contentDescription = "Удалить", modifier = Modifier.size(14.dp))
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Поле ввода запроса со скрепкой
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    placeholder = { Text("Введите запрос или путь к файлу...", fontSize = 13.sp) },
                    trailingIcon = {
                        IconButton(onClick = onAttachClick) {
                            Icon(
                                painter = painterResource(id = R.drawable.ic_attach),
                                contentDescription = "Прикрепить файл",
                                tint = if (attachedName != null) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    maxLines = 3
                )

                Spacer(modifier = Modifier.height(10.dp))

                // Кнопка отправки
                Button(
                    onClick = {
                        if (inputText.isBlank() && attachedName == null) return@Button
                        if (provider != LLMProviderType.LOCAL_GGUF && apiKey.isBlank()) {
                            Toast.makeText(context, "Укажите API-ключ в настройках приложения", Toast.LENGTH_LONG).show()
                            return@Button
                        }

                        val queryText = inputText
                        isGenerating = true
                        responseText = ""
                        statusText = "Генерация ответа..."

                        currentJob = coroutineScope.launch(Dispatchers.IO) {
                            try {
                                // 1. Проверка вложенного файла или путей в памяти
                                val attachedContent = getAttachedContent()
                                val filePrefix = if (attachedName != null && attachedContent != null) {
                                    "[Прикрепленный файл: $attachedName]:\n```\n$attachedContent\n```\n\n"
                                } else ""

                                // Обогащение запроса путями из памяти
                                val (enrichedQuery, loadedFiles) = FileStorageManager.enrichPromptWithLocalFiles(queryText)
                                val finalPrompt = filePrefix + enrichedQuery

                                if (loadedFiles.isNotEmpty()) {
                                    withContext(Dispatchers.Main) {
                                        Toast.makeText(context, "Прочитан файл: ${loadedFiles.joinToString()}", Toast.LENGTH_SHORT).show()
                                    }
                                }

                                // 2. Обогащение системной телеметрией
                                val systemContext = if (DeviceTools.isDeviceQuery(queryText)) {
                                    DeviceTools.buildTelemetryContext(context)
                                } else {
                                    "Ты — краткий, умный и точный русскоязычный персональный ассистент."
                                }

                                // 3. Инференс
                                val responseBuilder = StringBuilder()
                                if (provider != LLMProviderType.LOCAL_GGUF) {
                                    val messages = listOf(
                                        ApiChatMessage("system", systemContext),
                                        ApiChatMessage("user", finalPrompt)
                                    )
                                    OpenCodeClient.streamChatCompletions(
                                        provider = provider,
                                        apiKey = apiKey,
                                        model = modelName,
                                        messages = messages,
                                        onToken = { token ->
                                            responseBuilder.append(token)
                                            val cur = responseBuilder.toString()
                                            coroutineScope.launch(Dispatchers.Main) {
                                                responseText = cur
                                            }
                                        },
                                        isCancelled = { !isGenerating }
                                    )
                                } else {
                                    // Локальный инференс через LlamaBridge
                                    if (LlamaBridge.isModelLoaded) {
                                        val formatted = LlamaBridge.nativeFormatChat(
                                            arrayOf("system", "user"),
                                            arrayOf(systemContext, finalPrompt)
                                        )
                                        val callback = LlamaBridge.TokenCallback { token ->
                                            responseBuilder.append(token)
                                            val cur = responseBuilder.toString()
                                            coroutineScope.launch(Dispatchers.Main) {
                                                responseText = cur
                                            }
                                        }
                                        LlamaBridge.nativeGenerate(
                                            formatted,
                                            1024,
                                            0.3f,
                                            0.85f,
                                            callback
                                        )
                                    } else {
                                        withContext(Dispatchers.Main) {
                                            responseText = "⚠️ Локальная модель GGUF не загружена в память.\nОткройте приложение LLM и выберите модель GGUF, либо переключитесь на бесплатные облачные провайдеры (Groq / OpenRouter)."
                                        }
                                    }
                                }

                                val finalResp = responseBuilder.toString()
                                withContext(Dispatchers.Main) {
                                    isGenerating = false
                                    statusText = "Готово"
                                    // Обновляем виджет на рабочем столе полученным ответом
                                    LLMWidgetProvider.updateAllWidgets(
                                        context = context,
                                        query = queryText,
                                        response = finalResp,
                                        modelName = "${provider.displayName} • $modelName"
                                    )
                                }
                            } catch (e: Exception) {
                                withContext(Dispatchers.Main) {
                                    isGenerating = false
                                    statusText = "Ошибка: ${e.message}"
                                    responseText = "⚠️ Ошибка: ${e.message}"
                                }
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    enabled = !isGenerating
                ) {
                    if (isGenerating) {
                        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Остановить")
                    } else {
                        Icon(Icons.Default.Send, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Отправить")
                    }
                }

                // Область вывода ответа модели
                if (responseText.isNotBlank() || isGenerating) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 90.dp, max = 280.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                        ),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(12.dp)
                                .verticalScroll(rememberScrollState())
                        ) {
                            SelectionContainer {
                                Text(
                                    text = responseText.ifBlank { statusText },
                                    style = MaterialTheme.typography.bodyMedium,
                                    lineHeight = 20.sp
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    // Панель действий: Скопировать, Скачать код, Открыть в приложении
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Кнопка копирования
                        OutlinedButton(
                            onClick = {
                                FileStorageManager.copyToClipboard(context, responseText)
                                Toast.makeText(context, "Текст скопирован", Toast.LENGTH_SHORT).show()
                            },
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Icon(
                                painter = painterResource(id = R.drawable.ic_copy),
                                contentDescription = "Скопировать",
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Скопировать", fontSize = 11.sp)
                        }

                        // Кнопка скачивания кода (если есть код)
                        if (codeBlocks.isNotEmpty()) {
                            Button(
                                onClick = {
                                    val block = codeBlocks.first()
                                    val saved = FileStorageManager.saveCodeToDownloads(
                                        context,
                                        block.code,
                                        block.suggestedFilename,
                                        block.language
                                    )
                                    Toast.makeText(context, "📥 Код сохранен: Download/${saved.name}", Toast.LENGTH_LONG).show()
                                },
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Icon(
                                    painter = painterResource(id = R.drawable.ic_download),
                                    contentDescription = "Скачать код",
                                    modifier = Modifier.size(14.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Скачать код", fontSize = 11.sp)
                            }
                        }

                        Spacer(modifier = Modifier.weight(1f))

                        // Открыть в основном чате
                        TextButton(
                            onClick = { onOpenFullApp(inputText) },
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Text("В чат ➔", fontSize = 11.sp)
                        }
                    }
                }
            }
        }
    }
}
