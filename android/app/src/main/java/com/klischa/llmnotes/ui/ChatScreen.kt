@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.klischa.llmnotes.ui

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
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
import com.klischa.llmnotes.LLMViewModel
import com.klischa.llmnotes.UiState

@Composable
fun ChatScreen(
    viewModel: LLMViewModel,
    onSelectModelClick: () -> Unit
) {
    val uiState = viewModel.uiState.collectAsState().value
    val scrollState = rememberScrollState()
    val clipboardManager = LocalClipboardManager.current
    val context = LocalContext.current

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "Офлайн LLM Ассистент",
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        Text(
                            text = "Infinix Note 30 • Helio G99",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                actions = {
                    IconButton(onClick = onSelectModelClick) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = "Выбрать GGUF"
                        )
                    }
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
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .verticalScroll(scrollState),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // 1. Панель мониторинга устройства и статуса
            DeviceInfoCard(uiState, viewModel)

            // 2. Вкладки режима работы
            TabRow(selectedTabIndex = uiState.selectedTab) {
                Tab(
                    selected = uiState.selectedTab == 0,
                    onClick = { viewModel.setSelectedTab(0) },
                    text = { Text("Заметки и Саммари") }
                )
                Tab(
                    selected = uiState.selectedTab == 1,
                    onClick = { viewModel.setSelectedTab(1) },
                    text = { Text("Вопрос-Ответ") }
                )
            }

            // 3. Поле ввода текста (заметка или вопрос)
            OutlinedTextField(
                value = uiState.inputText,
                onValueChange = { viewModel.updateInputText(it) },
                label = {
                    Text(
                        if (uiState.selectedTab == 0) "Текст заметки или конспекта"
                        else "Ваш вопрос ассистенту"
                    )
                },
                placeholder = {
                    Text(
                        if (uiState.selectedTab == 0) "Вставьте текст для анализа или конспектирования..."
                        else "Спросите о чем угодно на русском языке..."
                    )
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 120.dp, max = 220.dp),
                shape = RoundedCornerShape(12.dp)
            )

            // 4. Панель быстрых действий для заметок
            if (uiState.selectedTab == 0) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = { viewModel.generateSummary() },
                        enabled = uiState.isModelLoaded && !uiState.isGenerating && uiState.inputText.isNotBlank(),
                        modifier = Modifier.weight(1f),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 8.dp)
                    ) {
                        Text("Выжимка", fontSize = 13.sp)
                    }
                    Button(
                        onClick = { viewModel.extractActionItems() },
                        enabled = uiState.isModelLoaded && !uiState.isGenerating && uiState.inputText.isNotBlank(),
                        modifier = Modifier.weight(1f),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 8.dp)
                    ) {
                        Text("Задачи", fontSize = 13.sp)
                    }
                    Button(
                        onClick = { viewModel.formatAndClean() },
                        enabled = uiState.isModelLoaded && !uiState.isGenerating && uiState.inputText.isNotBlank(),
                        modifier = Modifier.weight(1f),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 8.dp)
                    ) {
                        Text("Оформить", fontSize = 13.sp)
                    }
                }
            } else {
                Button(
                    onClick = { viewModel.askQuestion() },
                    enabled = uiState.isModelLoaded && !uiState.isGenerating && uiState.inputText.isNotBlank(),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.Send, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Задать вопрос")
                }
            }

            // Кнопка остановки генерации
            if (uiState.isGenerating) {
                OutlinedButton(
                    onClick = { viewModel.stopGeneration() },
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.Close, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Остановить генерацию")
                }
            }

            // 5. Карточка результата
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Ответ ассистента:",
                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                        )

                        if (uiState.outputText.isNotBlank()) {
                            IconButton(onClick = {
                                clipboardManager.setText(AnnotatedString(uiState.outputText))
                                Toast.makeText(context, "Скопировано в буфер!", Toast.LENGTH_SHORT).show()
                            }) {
                                Icon(
                                    imageVector = Icons.Default.Share,
                                    contentDescription = "Копировать",
                                    modifier = Modifier.size(20.dp)
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    if (uiState.outputText.isBlank()) {
                        Text(
                            text = if (uiState.isGenerating) "Генерация ответа..." else "Здесь появится ответ ассистента.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    } else {
                        SelectionContainer {
                            Text(
                                text = uiState.outputText,
                                style = MaterialTheme.typography.bodyMedium.copy(lineHeight = 22.sp),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }
                    }
                }
            }

            // Статус-бар внизу
            Text(
                text = "Статус: ${uiState.statusMessage}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun DeviceInfoCard(uiState: UiState, viewModel: LLMViewModel) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.4f)
        )
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .background(
                                if (uiState.isModelLoaded) Color(0xFF4CAF50) else Color(0xFFFF9800),
                                shape = RoundedCornerShape(5.dp)
                            )
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (uiState.isModelLoaded) "Модель загружена" else "Модель не выбрана",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                    )
                }

                Text(
                    text = "RAM: ~${uiState.allocatedRamMb} МБ",
                    style = MaterialTheme.typography.labelSmall
                )
            }

            Spacer(modifier = Modifier.height(6.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Скорость: ${String.format("%.1f", uiState.tokensPerSecond)} tok/s",
                    style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold)
                )

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Потоки:", style = MaterialTheme.typography.labelSmall)
                    Spacer(modifier = Modifier.width(4.dp))
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
