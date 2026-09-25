package com.klischa.llmnotes

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.DocumentsContract
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.ViewModelProvider
import com.klischa.llmnotes.ui.ChatScreen
import com.klischa.llmnotes.ui.LLMNotesTheme
import java.io.File

class MainActivity : ComponentActivity() {

    private lateinit var viewModel: LLMViewModel

    // Лаунчер выбора файла .gguf из проводника Android
    private val openDocumentLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let {
            val directPath = getRealPathFromUri(it)
            if (directPath != null && File(directPath).canRead() && File(directPath).length() > 0) {
                viewModel.loadModel(directPath)
            } else {
                viewModel.loadModelUri(it)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel = ViewModelProvider(this)[LLMViewModel::class.java]

        // Запрос разрешения доступа ко всем файлам для прямого zero-copy чтения моделей
        checkAndRequestAllFilesAccess()

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

    private fun checkAndRequestAllFilesAccess() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                try {
                    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                        data = Uri.parse("package:$packageName")
                    }
                    startActivity(intent)
                } catch (_: Exception) {
                    try {
                        val intent = Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                        startActivity(intent)
                    } catch (_: Exception) {}
                }
            }
        }
    }

    /**
     * Попытка извлечь реальный путь к файлу из URI провайдера Android,
     * если приложению предоставлен доступ к файловой системе (All Files Access).
     */
    private fun getRealPathFromUri(uri: Uri): String? {
        if (uri.scheme == "file") return uri.path
        if (uri.scheme == "content") {
            try {
                if (DocumentsContract.isDocumentUri(this, uri)) {
                    val docId = DocumentsContract.getDocumentId(uri)
                    if (docId.startsWith("primary:")) {
                        val relPath = docId.substring("primary:".length)
                        val extDir = Environment.getExternalStorageDirectory().absolutePath
                        val directFile = File(extDir, relPath)
                        if (directFile.exists() && directFile.canRead()) {
                            return directFile.absolutePath
                        }
                    } else if (docId.startsWith("raw:")) {
                        val rawPath = docId.substring("raw:".length)
                        val rawFile = File(rawPath)
                        if (rawFile.exists() && rawFile.canRead()) {
                            return rawFile.absolutePath
                        }
                    }
                }
            } catch (_: Exception) {}
        }
        return null
    }
}
