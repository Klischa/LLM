package com.klischa.llmnotes

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.klischa.llmnotes.ui.ChatScreen
import com.klischa.llmnotes.ui.LLMNotesTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {

    private lateinit var viewModel: LLMViewModel

    // Лаунчер выбора файла .gguf из проводника Android
    private val openDocumentLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { handleModelUri(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel = ViewModelProvider(this)[LLMViewModel::class.java]
        setContent {
            LLMNotesTheme {
                ChatScreen(
                    viewModel = viewModel,
                    onSelectModelClick = {
                        openDocumentLauncher.launch(arrayOf("*/*"))
                    }
                )
            }
        }
    }

    /**
     * llama.cpp требует прямой путь к файлу для mmap.
     * Копируем выбранный файл из ContentProvider в локальный кэш приложения,
     * если он еще не скопирован, или загружаем напрямую.
     */
    private fun handleModelUri(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Подготовка файла модели...", Toast.LENGTH_SHORT).show()
                }

                val destinationFile = File(filesDir, "model.gguf")

                contentResolver.openInputStream(uri)?.use { inputStream ->
                    FileOutputStream(destinationFile).use { outputStream ->
                        inputStream.copyTo(outputStream, bufferSize = 1024 * 1024)
                    }
                }

                withContext(Dispatchers.Main) {
                    viewModel.loadModel(destinationFile.absolutePath)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Ошибка открытия файла: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
}
