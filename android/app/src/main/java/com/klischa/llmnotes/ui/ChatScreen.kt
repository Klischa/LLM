@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.klischa.llmnotes.ui

import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.klischa.llmnotes.ChatMessage
import com.klischa.llmnotes.LLMViewModel
import com.klischa.llmnotes.MessageRole
import com.klischa.llmnotes.SystemPromptPreset
import com.klischa.llmnotes.UiState
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun ChatScreen(
    viewModel: LLMViewModel,
    onSelectModelClick: () -> Unit
) {
    val uiState = viewModel.uiState.collectAsState().value
    val listState = rememberLazyListState()
    val clipboardManager = LocalClipboardManager.current
    val context = LocalContext.current

    // Автоматическая прокрутка к последнему сообщению при получении новых токенов
    LaunchedEffect(uiState.messages.size, uiState.messages.lastOrNull()?.text) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "LLM",
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        val modelDisplayName = if (uiState.isModelLoaded) {
                            java.io.File(uiState.modelPath).name.ifBlank { uiState.modelPath }
                        } else {
                            "Офлайн ассистент • Helio G99"
                        }
                        Text(
                            text = modelDisplayName,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                        )
                    }
                },
                actions = {
                    // Кнопка настройки системного промпта
                    IconButton(onClick = { viewModel.toggleSystemPromptExpanded() }) {
                        Icon(
                            imageVector = Icons.Default.Edit,
                            contentDescription = "Системный промпт",
                            tint = if (uiState.isSystemPromptExpanded) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                        )
                    }

                    // Выбор GGUF модели
                    IconButton(onClick = onSelectModelClick) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = "Выбрать GGUF"
                        )
                    }

                    // Очистить историю диалога
                    if (uiState.messages.isNotEmpty()) {
                        IconButton(onClick = { viewModel.clearChat() }) {
                            Icon(
                                imageVector = Icons.Default.Delete,
                                contentDescription = "Очистить чат"
                            )
                        }
                    }

                    // Выгрузить модель
                    if (uiState.isModelLoaded) {
                        IconButton(onClick = { viewModel.unloadModel() }) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = "Выгрузить модель",
                                tint = MaterialTheme.colorScheme.error
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .imePadding()
        ) {
            // 1. Панель параметров системного промпта
            AnimatedVisibility(visible = uiState.isSystemPromptExpanded) {
                SystemPromptCard(uiState, viewModel)
            }

            // 2. Информационная плашка статуса модели и производительности
            DeviceStatusStrip(uiState, viewModel)

            // 3. Область сообщений чата
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                if (uiState.messages.isEmpty()) {
                    EmptyChatPlaceholder(uiState, viewModel, onSelectModelClick)
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 12.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        items(uiState.messages, key = { it.id }) { message ->
                            ChatMessageBubble(
                                message = message,
                                onCopyClick = {
                                    clipboardManager.setText(AnnotatedString(message.text))
                                    Toast.makeText(context, "Текст скопирован", Toast.LENGTH_SHORT).show()
                                }
                            )
                        }
                    }
                }
            }

            // 4. Панель быстрых подсказок (саммари, задачи, markdown)
            QuickPromptChips(
                isModelLoaded = uiState.isModelLoaded,
                onChipClick = { type -> viewModel.insertNotePrompt(type) }
            )

            // 5. Поле ввода сообщения и кнопка отправки/остановки
            ChatInputBar(
                uiState = uiState,
                onInputTextChange = { viewModel.updateInputText(it) },
                onSendClick = { viewModel.sendMessage() },
                onStopClick = { viewModel.stopGeneration() }
            )
        }
    }
}

@Composable
fun DeviceStatusStrip(uiState: UiState, viewModel: LLMViewModel) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.35f)
        )
    ) {
        Column(modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(9.dp)
                            .background(
                                when {
                                    uiState.isModelLoaded -> Color(0xFF4CAF50)
                                    uiState.isLoadingModel -> Color(0xFF2196F3)
                                    else -> Color(0xFFFF9800)
                                },
                                shape = CircleShape
                            )
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = when {
                            uiState.isModelLoaded -> "Модель активна"
                            uiState.isLoadingModel -> "Загрузка модели..."
                            else -> "Модель не выбрана"
                        },
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                    )
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "RAM: ~${uiState.allocatedRamMb} МБ",
                        style = MaterialTheme.typography.labelSmall
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = "${String.format("%.1f", uiState.tokensPerSecond)} tok/s",
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }

            if (uiState.isLoadingModel) {
                Spacer(modifier = Modifier.height(4.dp))
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }

            // Текст статуса или ошибки
            val isError = !uiState.isModelLoaded && !uiState.isLoadingModel &&
                          (uiState.statusMessage.contains("Ошибка") ||
                           uiState.statusMessage.contains("не найден") ||
                           uiState.statusMessage.contains("не поддерживается") ||
                           uiState.statusMessage.contains("Недостаточно"))
            if (uiState.statusMessage.isNotBlank()) {
                Text(
                    text = uiState.statusMessage,
                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                    color = when {
                        isError -> MaterialTheme.colorScheme.error
                        uiState.isLoadingModel -> MaterialTheme.colorScheme.primary
                        else -> MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    modifier = Modifier.padding(top = 2.dp)
                )
            }
        }
    }
}

@Composable
fun SystemPromptCard(uiState: UiState, viewModel: LLMViewModel) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f)
        )
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Text(
                text = "Системный промпт диалога:",
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
            )

            Spacer(modifier = Modifier.height(4.dp))

            // Пресеты
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                SystemPromptPreset.values().forEach { preset ->
                    FilterChip(
                        selected = uiState.systemPrompt == preset.prompt,
                        onClick = { viewModel.applyPreset(preset) },
                        label = { Text(preset.title, fontSize = 11.sp) },
                        modifier = Modifier.padding(vertical = 1.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            OutlinedTextField(
                value = uiState.systemPrompt,
                onValueChange = { viewModel.updateSystemPrompt(it) },
                label = { Text("Пользовательская инструкция") },
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 70.dp, max = 130.dp),
                shape = RoundedCornerShape(10.dp)
            )

            // Переключатель потоков
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 6.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Потоки (Helio G99 оптимум: 2):",
                    style = MaterialTheme.typography.labelSmall
                )
                Row {
                    listOf(2, 3, 4).forEach { count ->
                        FilterChip(
                            selected = uiState.threadCount == count,
                            onClick = { viewModel.setThreadCount(count) },
                            label = { Text("$count", fontSize = 11.sp) },
                            modifier = Modifier.padding(horizontal = 2.dp)
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun EmptyChatPlaceholder(
    uiState: UiState,
    viewModel: LLMViewModel,
    onSelectModelClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Офлайн-чат с LLM",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = if (uiState.isModelLoaded)
                            "Модель готова к диалогу. Введите сообщение ниже или выберите готовую тему."
                        else
                            "Загрузите GGUF-модель (Qwen 2.5 1.5B/3B, Llama 3.2), чтобы начать общение.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    if (!uiState.isModelLoaded) {
                        Spacer(modifier = Modifier.height(12.dp))
                        Button(
                            onClick = onSelectModelClick,
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Выбрать файл .gguf")
                        }
                    }
                }
            }

            // Быстрые карточки для старта
            if (uiState.isModelLoaded) {
                Text(
                    text = "Идеи для запросов:",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                val suggestions = listOf(
                    "📸 Сколько фото сохранено на моем телефоне?",
                    "💾 Сколько свободно памяти на устройстве?",
                    "🔋 Какой текущий уровень заряда аккумулятора?",
                    "📝 Сделай краткое резюме заметки...",
                    "✅ Составь список задач (Action Items)...",
                    "💡 Объясни простыми словами квантовую физику"
                )

                suggestions.forEach { suggestion ->
                    OutlinedButton(
                        onClick = { viewModel.updateInputText(suggestion) },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(10.dp),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp)
                    ) {
                        Text(
                            text = suggestion,
                            fontSize = 12.sp,
                            maxLines = 1
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun ChatMessageBubble(
    message: ChatMessage,
    onCopyClick: () -> Unit
) {
    val isUser = message.role == MessageRole.USER
    val timeFormat = SimpleDateFormat("HH:mm", Locale.getDefault())
    val timeStr = timeFormat.format(Date(message.timestamp))

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Card(
            shape = if (isUser) {
                RoundedCornerShape(16.dp, 16.dp, 4.dp, 16.dp)
            } else {
                RoundedCornerShape(16.dp, 16.dp, 16.dp, 4.dp)
            },
            colors = CardDefaults.cardColors(
                containerColor = if (isUser) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceVariant
                }
            ),
            modifier = Modifier.widthIn(min = 80.dp, max = 320.dp)
        ) {
            Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
                // Роль собеседника
                Text(
                    text = if (isUser) "Вы" else "Ассистент",
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                    color = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary
                )

                Spacer(modifier = Modifier.height(4.dp))

                // Текст сообщения
                SelectionContainer {
                    Text(
                        text = if (message.text.isEmpty() && message.isStreaming) "▌" else message.text,
                        style = MaterialTheme.typography.bodyMedium.copy(lineHeight = 21.sp),
                        color = if (isUser) {
                            MaterialTheme.colorScheme.onPrimaryContainer
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                }

                Spacer(modifier = Modifier.height(4.dp))

                // Нижняя строка: скорость, кнопка копирования и время
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (!isUser && message.tokensPerSec != null && message.tokensPerSec > 0) {
                        Text(
                            text = "${String.format("%.1f", message.tokensPerSec)} tok/s",
                            fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.primary
                        )
                    } else {
                        Spacer(modifier = Modifier.width(1.dp))
                    }

                    Row(verticalAlignment = Alignment.CenterVertically) {
                        if (!isUser && message.text.isNotBlank()) {
                            Icon(
                                imageVector = Icons.Default.Share,
                                contentDescription = "Копировать",
                                modifier = Modifier
                                    .size(14.dp)
                                    .clickable { onCopyClick() },
                                tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                        }

                        Text(
                            text = timeStr,
                            fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun QuickPromptChips(
    isModelLoaded: Boolean,
    onChipClick: (String) -> Unit
) {
    LazyRow(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        item {
            SuggestionChip(
                onClick = { onChipClick("photos") },
                label = { Text("📸 Сколько фото?", fontSize = 11.sp) },
                enabled = isModelLoaded
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("storage") },
                label = { Text("💾 Память", fontSize = 11.sp) },
                enabled = isModelLoaded
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("battery") },
                label = { Text("🔋 Батарея", fontSize = 11.sp) },
                enabled = isModelLoaded
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("device") },
                label = { Text("📱 Телефон", fontSize = 11.sp) },
                enabled = isModelLoaded
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("summary") },
                label = { Text("📝 Саммари", fontSize = 11.sp) },
                enabled = isModelLoaded
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("tasks") },
                label = { Text("✅ Задачи", fontSize = 11.sp) },
                enabled = isModelLoaded
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("format") },
                label = { Text("✨ Markdown", fontSize = 11.sp) },
                enabled = isModelLoaded
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("explain") },
                label = { Text("💡 Объясни", fontSize = 11.sp) },
                enabled = isModelLoaded
            )
        }
    }
}

@Composable
fun ChatInputBar(
    uiState: UiState,
    onInputTextChange: (String) -> Unit,
    onSendClick: () -> Unit,
    onStopClick: () -> Unit
) {
    Surface(
        tonalElevation = 2.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.Bottom,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = uiState.inputText,
                onValueChange = onInputTextChange,
                placeholder = {
                    Text(
                        if (uiState.isModelLoaded) "Сообщение ассистенту..."
                        else "Сначала выберите модель GGUF..."
                    )
                },
                modifier = Modifier
                    .weight(1f)
                    .heightIn(min = 52.dp, max = 130.dp),
                shape = RoundedCornerShape(22.dp),
                enabled = uiState.isModelLoaded && !uiState.isGenerating
            )

            if (uiState.isGenerating) {
                IconButton(
                    onClick = onStopClick,
                    modifier = Modifier
                        .size(48.dp)
                        .background(MaterialTheme.colorScheme.errorContainer, CircleShape)
                ) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Остановить",
                        tint = MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            } else {
                IconButton(
                    onClick = onSendClick,
                    enabled = uiState.isModelLoaded && uiState.inputText.isNotBlank(),
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            if (uiState.isModelLoaded && uiState.inputText.isNotBlank()) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.surfaceVariant
                            },
                            CircleShape
                        )
                ) {
                    Icon(
                        imageVector = Icons.Default.Send,
                        contentDescription = "Отправить",
                        tint = if (uiState.isModelLoaded && uiState.inputText.isNotBlank()) {
                            MaterialTheme.colorScheme.onPrimary
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                        }
                    )
                }
            }
        }
    }
}
