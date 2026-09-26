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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.klischa.llmnotes.ChatMessage
import com.klischa.llmnotes.LLMViewModel
import com.klischa.llmnotes.MessageRole
import com.klischa.llmnotes.SystemPromptPreset
import com.klischa.llmnotes.UiState
import com.klischa.llmnotes.api.LLMProviderType
import kotlinx.coroutines.launch
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

    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val coroutineScope = rememberCoroutineScope()
    var chatSearchQuery by remember { mutableStateOf("") }

    // Диалог настройки провайдеров (OpenCode GO, ZEN, локальный GGUF)
    if (uiState.isProviderDialogVisible) {
        ProviderSettingsDialog(
            uiState = uiState,
            viewModel = viewModel,
            onDismiss = { viewModel.setProviderDialogVisible(false) }
        )
    }

    // Автоматическая прокрутка к последнему сообщению при получении новых токенов
    LaunchedEffect(uiState.messages.size, uiState.messages.lastOrNull()?.text) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(
                modifier = Modifier.width(310.dp)
            ) {
                // Заголовок боковой шторки
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "История диалогов",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    IconButton(
                        onClick = {
                            viewModel.createNewChat()
                            coroutineScope.launch { drawerState.close() }
                        }
                    ) {
                        Icon(Icons.Default.Add, contentDescription = "Новый диалог")
                    }
                }

                HorizontalDivider()

                // Поиск по названию диалогов
                OutlinedTextField(
                    value = chatSearchQuery,
                    onValueChange = { chatSearchQuery = it },
                    placeholder = { Text("Поиск диалога...", fontSize = 12.sp) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    shape = RoundedCornerShape(10.dp),
                    singleLine = true
                )

                // Список диалогов
                val filteredSessions = if (chatSearchQuery.isBlank()) {
                    uiState.chatSessions
                } else {
                    uiState.chatSessions.filter { it.title.contains(chatSearchQuery, ignoreCase = true) }
                }

                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .padding(horizontal = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(filteredSessions, key = { it.id }) { session ->
                        val isSelected = session.id == uiState.currentChatId
                        val dateFormat = SimpleDateFormat("d MMM, HH:mm", Locale.getDefault())
                        val timeStr = dateFormat.format(Date(session.updatedAt))

                        NavigationDrawerItem(
                            label = {
                                Column(modifier = Modifier.padding(vertical = 4.dp)) {
                                    Text(
                                        text = session.title,
                                        maxLines = 1,
                                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                                        style = MaterialTheme.typography.bodyMedium.copy(
                                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                                        )
                                    )
                                    Text(
                                        text = timeStr,
                                        fontSize = 11.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                                    )
                                }
                            },
                            selected = isSelected,
                            onClick = {
                                viewModel.selectChat(session.id)
                                coroutineScope.launch { drawerState.close() }
                            },
                            badge = {
                                IconButton(
                                    onClick = { viewModel.deleteChat(session.id) },
                                    modifier = Modifier.size(28.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Delete,
                                        contentDescription = "Удалить диалог",
                                        tint = MaterialTheme.colorScheme.error.copy(alpha = 0.7f),
                                        modifier = Modifier.size(16.dp)
                                    )
                                }
                            },
                            modifier = Modifier.padding(vertical = 2.dp),
                            colors = NavigationDrawerItemDefaults.colors(
                                selectedContainerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.6f)
                            )
                        )
                    }
                }

                // Кнопка создания нового диалога внизу шторки
                Button(
                    onClick = {
                        viewModel.createNewChat()
                        coroutineScope.launch { drawerState.close() }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Новый диалог")
                }
            }
        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    navigationIcon = {
                        IconButton(onClick = { coroutineScope.launch { drawerState.open() } }) {
                            Icon(Icons.Default.Menu, contentDescription = "Меню диалогов")
                        }
                    },
                    title = {
                        Column {
                            val currentTitle = uiState.chatSessions.find { it.id == uiState.currentChatId }?.title ?: "LLM"
                            Text(
                                text = currentTitle,
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                maxLines = 1,
                                overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                            )
                            val modelDisplayName = when (uiState.providerType) {
                                LLMProviderType.LOCAL_GGUF -> {
                                    if (uiState.isModelLoaded) java.io.File(uiState.modelPath).name.ifBlank { uiState.modelPath }
                                    else "Офлайн GGUF • Helio G99"
                                }
                                LLMProviderType.OPENCODE_GO -> "OpenCode GO • ${uiState.openCodeSelectedModel}"
                                LLMProviderType.OPENCODE_ZEN -> "OpenCode ZEN • ${uiState.openCodeSelectedModel}"
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
                        // Выбор провайдера (Локально / OpenCode GO / ZEN)
                        IconButton(onClick = { viewModel.setProviderDialogVisible(true) }) {
                            Icon(
                                imageVector = Icons.Default.Settings,
                                contentDescription = "Провайдер LLM",
                                tint = if (uiState.providerType != LLMProviderType.LOCAL_GGUF) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                            )
                        }

                        // Кнопка настройки системного промпта
                        IconButton(onClick = { viewModel.toggleSystemPromptExpanded() }) {
                            Icon(
                                imageVector = Icons.Default.Edit,
                                contentDescription = "Системный промпт",
                                tint = if (uiState.isSystemPromptExpanded) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                            )
                        }

                        // Выбор GGUF модели (только в локальном режиме)
                        if (uiState.providerType == LLMProviderType.LOCAL_GGUF) {
                            IconButton(onClick = onSelectModelClick) {
                                Icon(
                                    imageVector = Icons.Default.Add,
                                    contentDescription = "Выбрать GGUF"
                                )
                            }
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

                        // Выгрузить локальную модель
                        if (uiState.providerType == LLMProviderType.LOCAL_GGUF && uiState.isModelLoaded) {
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
                        EmptyChatPlaceholder(
                            uiState = uiState,
                            onSelectModelClick = onSelectModelClick,
                            onOpenProviderSettings = { viewModel.setProviderDialogVisible(true) }
                        )
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

                // 4. Панель быстрых подсказок (телеметрия, саммари, задачи)
                QuickPromptChips(
                    isReady = uiState.isReadyToChat,
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
}

@Composable
fun DeviceStatusStrip(uiState: UiState, viewModel: LLMViewModel) {
    val isLocal = uiState.providerType == LLMProviderType.LOCAL_GGUF
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
                                    isLocal && uiState.isModelLoaded -> Color(0xFF4CAF50)
                                    !isLocal && uiState.openCodeApiKey.isNotBlank() -> Color(0xFF4CAF50)
                                    uiState.isLoadingModel -> Color(0xFF2196F3)
                                    else -> Color(0xFFFF9800)
                                },
                                shape = CircleShape
                            )
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = when {
                            isLocal && uiState.isModelLoaded -> "Модель активна"
                            isLocal && uiState.isLoadingModel -> "Загрузка модели..."
                            isLocal -> "Модель не выбрана"
                            !isLocal && uiState.openCodeApiKey.isNotBlank() -> "${uiState.providerType.displayName} активен"
                            else -> "Требуется API-ключ"
                        },
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                    )
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (isLocal) {
                        Text(
                            text = "RAM: ~${uiState.allocatedRamMb} МБ",
                            style = MaterialTheme.typography.labelSmall
                        )
                    } else {
                        Text(
                            text = "Облачный API",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.secondary
                        )
                    }
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
            val isError = (isLocal && !uiState.isModelLoaded && !uiState.isLoadingModel &&
                          (uiState.statusMessage.contains("Ошибка") ||
                           uiState.statusMessage.contains("не найден") ||
                           uiState.statusMessage.contains("не поддерживается") ||
                           uiState.statusMessage.contains("Недостаточно"))) ||
                          (!isLocal && uiState.statusMessage.contains("Ошибка"))

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

            // Переключатель потоков (для локальной модели)
            if (uiState.providerType == LLMProviderType.LOCAL_GGUF) {
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
}

@Composable
fun EmptyChatPlaceholder(
    uiState: UiState,
    onSelectModelClick: () -> Unit,
    onOpenProviderSettings: () -> Unit
) {
    val isLocal = uiState.providerType == LLMProviderType.LOCAL_GGUF
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
            ),
            modifier = Modifier.fillMaxWidth(0.92f)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = if (isLocal) "Офлайн-чат с LLM" else "Чат с ${uiState.providerType.displayName}",
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = when {
                        isLocal && uiState.isModelLoaded -> "Модель готова к диалогу. Введите сообщение ниже или воспользуйтесь быстрыми кнопками."
                        isLocal -> "Загрузите GGUF-модель (Qwen 2.5, Llama 3.2), чтобы начать общение, либо подключите API OpenCode."
                        !isLocal && uiState.openCodeApiKey.isNotBlank() -> "Модель ${uiState.openCodeSelectedModel} готова к работе через ${uiState.providerType.displayName}."
                        else -> "Введите API-ключ OpenCode для доступа к ${uiState.providerType.displayName}."
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )

                Spacer(modifier = Modifier.height(14.dp))
                if (isLocal && !uiState.isModelLoaded) {
                    Button(
                        onClick = onSelectModelClick,
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Выбрать файл .gguf")
                    }
                } else if (!isLocal && uiState.openCodeApiKey.isBlank()) {
                    Button(
                        onClick = onOpenProviderSettings,
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Icon(Icons.Default.Settings, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Ввести API-ключ")
                    }
                }
            }
        }
    }
}

@Composable
fun ProviderSettingsDialog(
    uiState: UiState,
    viewModel: LLMViewModel,
    onDismiss: () -> Unit
) {
    var apiKeyText by remember(uiState.openCodeApiKey) { mutableStateOf(uiState.openCodeApiKey) }
    var showApiKey by remember { mutableStateOf(false) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = "Провайдер LLM и модели",
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
            )
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text(
                    text = "Выберите источник инференса:",
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                )

                // Чипы выбора провайдера
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    FilterChip(
                        selected = uiState.providerType == LLMProviderType.LOCAL_GGUF,
                        onClick = { viewModel.setProviderType(LLMProviderType.LOCAL_GGUF) },
                        label = { Text("📱 Офлайн", fontSize = 11.sp) }
                    )
                    FilterChip(
                        selected = uiState.providerType == LLMProviderType.OPENCODE_GO,
                        onClick = { viewModel.setProviderType(LLMProviderType.OPENCODE_GO) },
                        label = { Text("⚡ OpenCode GO", fontSize = 11.sp) }
                    )
                    FilterChip(
                        selected = uiState.providerType == LLMProviderType.OPENCODE_ZEN,
                        onClick = { viewModel.setProviderType(LLMProviderType.OPENCODE_ZEN) },
                        label = { Text("🧘 ZEN", fontSize = 11.sp) }
                    )
                }

                if (uiState.providerType != LLMProviderType.LOCAL_GGUF) {
                    // Описание подписки
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.45f)
                        ),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            text = if (uiState.providerType == LLMProviderType.OPENCODE_GO)
                                "⚡ OpenCode GO: открытые и кодинг-модели (DeepSeek V4 Pro, Kimi K2.6, Qwen 3.6, GLM 5.1, MiniMax M3)."
                            else
                                "🧘 OpenCode ZEN: доступ к флагманам frontier (Claude 3.7 Sonnet, GPT-4o, DeepSeek R1, Gemini 2.0).",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            modifier = Modifier.padding(8.dp)
                        )
                    }

                    // Поле ввода API ключа
                    OutlinedTextField(
                        value = apiKeyText,
                        onValueChange = {
                            apiKeyText = it
                            viewModel.setOpenCodeApiKey(it)
                        },
                        label = { Text("API-ключ OpenCode (sk-...)") },
                        placeholder = { Text("sk-...") },
                        singleLine = true,
                        visualTransformation = if (showApiKey) VisualTransformation.None else PasswordVisualTransformation(),
                        trailingIcon = {
                            TextButton(onClick = { showApiKey = !showApiKey }) {
                                Text(if (showApiKey) "Скрыть" else "Показать", fontSize = 10.sp)
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(10.dp)
                    )

                    // Выбор модели из каталога
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Модель из подписки:",
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        TextButton(
                            onClick = { viewModel.fetchRemoteModels() },
                            enabled = !uiState.isFetchingModels && uiState.openCodeApiKey.isNotBlank()
                        ) {
                            Text(if (uiState.isFetchingModels) "Загрузка..." else "🔄 Обновить", fontSize = 11.sp)
                        }
                    }

                    // Список моделей
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 190.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        uiState.openCodeAvailableModels.forEach { modelName ->
                            val isSelected = modelName == uiState.openCodeSelectedModel
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable { viewModel.setOpenCodeSelectedModel(modelName) }
                                    .background(
                                        if (isSelected) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.6f)
                                        else Color.Transparent,
                                        RoundedCornerShape(8.dp)
                                    )
                                    .padding(horizontal = 8.dp, vertical = 4.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                RadioButton(
                                    selected = isSelected,
                                    onClick = { viewModel.setOpenCodeSelectedModel(modelName) }
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    text = modelName,
                                    style = MaterialTheme.typography.bodySmall.copy(
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                                    )
                                )
                            }
                        }
                    }
                } else {
                    Text(
                        text = "В офлайн-режиме модель выполняется локально на процессоре Helio G99 смартфона без подключения к интернету.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        },
        confirmButton = {
            Button(onClick = onDismiss) {
                Text("Готово")
            }
        }
    )
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
    isReady: Boolean,
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
                enabled = isReady
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("storage") },
                label = { Text("💾 Память", fontSize = 11.sp) },
                enabled = isReady
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("battery") },
                label = { Text("🔋 Батарея", fontSize = 11.sp) },
                enabled = isReady
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("device") },
                label = { Text("📱 Телефон", fontSize = 11.sp) },
                enabled = isReady
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("summary") },
                label = { Text("📝 Саммари", fontSize = 11.sp) },
                enabled = isReady
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("tasks") },
                label = { Text("✅ Задачи", fontSize = 11.sp) },
                enabled = isReady
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("format") },
                label = { Text("✨ Markdown", fontSize = 11.sp) },
                enabled = isReady
            )
        }
        item {
            SuggestionChip(
                onClick = { onChipClick("explain") },
                label = { Text("💡 Объясни", fontSize = 11.sp) },
                enabled = isReady
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
    val isReady = uiState.isReadyToChat

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
                        if (isReady) "Сообщение ассистенту..."
                        else if (uiState.providerType == LLMProviderType.LOCAL_GGUF) "Сначала выберите модель GGUF..."
                        else "Укажите API-ключ ${uiState.providerType.displayName}..."
                    )
                },
                modifier = Modifier
                    .weight(1f)
                    .heightIn(min = 52.dp, max = 130.dp),
                shape = RoundedCornerShape(22.dp),
                enabled = isReady && !uiState.isGenerating
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
                    enabled = isReady && uiState.inputText.isNotBlank(),
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            if (isReady && uiState.inputText.isNotBlank()) {
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
                        tint = if (isReady && uiState.inputText.isNotBlank()) {
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
