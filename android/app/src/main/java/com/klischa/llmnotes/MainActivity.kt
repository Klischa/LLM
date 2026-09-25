package com.klischa.llmnotes

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.ViewModelProvider
import com.klischa.llmnotes.ui.ChatScreen
import com.klischa.llmnotes.ui.LLMNotesTheme

class MainActivity : ComponentActivity() {

    private lateinit var viewModel: LLMViewModel

    // Лаунчер выбора файла .gguf из проводника Android
    private val openDocumentLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { viewModel.loadModelUri(it) }
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
}
