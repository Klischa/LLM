package com.klischa.llmnotes

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.media.MediaScannerConnection
import android.os.Environment
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.regex.Pattern

data class ExtractedCodeBlock(
    val language: String,
    val code: String,
    val suggestedFilename: String
)

/**
 * Менеджер работы с локальными файлами во внутренней памяти устройства:
 * - Поиск и безопасное чтение текстовых файлов по путям (/sdcard/..., Download/...).
 * - Обогащение запроса пользователя содержимым найденных файлов для анализа моделью.
 * - Скачивание и сохранение сгенерированного кода (Python, Bash и др.) в папку Download.
 * - Извлечение блоков кода из Markdown-ответов модели.
 */
object FileStorageManager {
    private const val TAG = "FileStorageManager"
    private const val MAX_READ_CHARS = 25000

    private val PATH_PATTERN = Pattern.compile(
        """(?:/sdcard/|/storage/emulated/0/|/storage/[^\s"':;,]+|Download/[^\s"':;,]+|Documents/[^\s"':;,]+|[a-zA-Z0-9_\-./]+\.(?:py|txt|json|csv|md|log|kt|java|c|cpp|h|sh|html|xml|yaml|yml|sql|js|ts|conf|ini|env))""",
        Pattern.CASE_INSENSITIVE
    )

    private val CODE_BLOCK_PATTERN = Pattern.compile(
        """```(?:([a-zA-Z0-9_+\-]+)(?:[:=\s]([^\n\r]+))?)?[\r\n]+([\s\S]*?)```"""
    )

    /**
     * Поиск и извлечение путей к файлам из текста сообщения.
     */
    fun extractFilePathsFromText(text: String): List<String> {
        val matches = mutableListOf<String>()
        val matcher = PATH_PATTERN.matcher(text)
        while (matcher.find()) {
            val path = matcher.group().trim()
            if (path.length >= 3 && !matches.contains(path)) {
                matches.add(path)
            }
        }
        return matches
    }

    /**
     * Разрешение относительного или абсолютного пути к файлу в память телефона.
     */
    fun resolveLocalFile(rawPath: String): File? {
        val clean = rawPath.trim().removeSurrounding("\"", "\"").removeSurrounding("'", "'")
        val direct = File(clean)
        if (direct.exists() && direct.isFile && direct.canRead()) {
            return direct
        }

        val extStorage = Environment.getExternalStorageDirectory()
        if (clean.startsWith("/sdcard/")) {
            val sub = clean.removePrefix("/sdcard/")
            val f = File(extStorage, sub)
            if (f.exists() && f.isFile && f.canRead()) return f
        }

        if (clean.startsWith("Download/", ignoreCase = true)) {
            val sub = clean.substring("Download/".length)
            val d = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val f = File(d, sub)
            if (f.exists() && f.isFile && f.canRead()) return f
        }

        if (clean.startsWith("Documents/", ignoreCase = true)) {
            val sub = clean.substring("Documents/".length)
            val d = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            val f = File(d, sub)
            if (f.exists() && f.isFile && f.canRead()) return f
        }

        // Поиск в Download по короткому имени (например "script.py")
        if (!clean.contains("/") && !clean.contains("\\")) {
            val d = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val f = File(d, clean)
            if (f.exists() && f.isFile && f.canRead()) return f
        }

        return null
    }

    /**
     * Безопасное чтение текстового файла (с защитой от бинарных данных и переполнения контекста).
     */
    fun readTextFile(file: File, maxChars: Int = MAX_READ_CHARS): Pair<String, Boolean> {
        return try {
            val bytes = file.readBytes()
            // Проверка на бинарный файл (наличие нулей в первых 1024 байтах)
            val checkLen = bytes.size.coerceAtMost(1024)
            for (i in 0 until checkLen) {
                if (bytes[i] == 0.toByte()) {
                    return Pair("⚠️ Файл '${file.name}' является бинарным и не может быть прочитан как текст.", false)
                }
            }

            val text = String(bytes, Charsets.UTF_8)
            if (text.length > maxChars) {
                val truncated = text.substring(0, maxChars) + "\n\n[...файл обрезан, показаны первые $maxChars символов из ${text.length}...]"
                Pair(truncated, true)
            } else {
                Pair(text, false)
            }
        } catch (e: Exception) {
            Pair("⚠️ Ошибка чтения файла '${file.name}': ${e.message}", false)
        }
    }

    /**
     * Автоматическое обогащение запроса содержимым найденных локальных файлов.
     */
    fun enrichPromptWithLocalFiles(prompt: String): Pair<String, List<String>> {
        val paths = extractFilePathsFromText(prompt)
        val loadedFiles = mutableListOf<String>()
        val fileContexts = StringBuilder()

        for (p in paths) {
            val file = resolveLocalFile(p)
            if (file != null && !loadedFiles.contains(file.absolutePath)) {
                val (content, _) = readTextFile(file)
                val ext = file.extension.lowercase()
                fileContexts.append("\n\n[Файл из памяти устройства: ${file.absolutePath} (${file.length() / 1024} КБ)]:\n")
                fileContexts.append("```$ext\n")
                fileContexts.append(content)
                fileContexts.append("\n```\n")
                loadedFiles.add(file.name)
            }
        }

        return if (loadedFiles.isNotEmpty()) {
            val enriched = fileContexts.toString().trim() + "\n\n" + prompt
            Pair(enriched, loadedFiles)
        } else {
            Pair(prompt, emptyList())
        }
    }

    /**
     * Извлечение всех блоков кода из markdown-сообщения.
     */
    fun extractCodeBlocks(markdown: String): List<ExtractedCodeBlock> {
        val blocks = mutableListOf<ExtractedCodeBlock>()
        val matcher = CODE_BLOCK_PATTERN.matcher(markdown)
        var index = 1
        while (matcher.find()) {
            val lang = matcher.group(1)?.trim()?.lowercase() ?: "code"
            val rawFilename = matcher.group(2)?.trim()
            val code = matcher.group(3) ?: ""
            if (code.isNotBlank()) {
                val suggestedName = when {
                    !rawFilename.isNullOrBlank() && rawFilename.contains(".") -> rawFilename
                    else -> generateDefaultFilename(lang, index++)
                }
                blocks.add(ExtractedCodeBlock(lang, code.trimEnd(), suggestedName))
            }
        }
        return blocks
    }

    private fun generateDefaultFilename(lang: String, index: Int): String {
        val time = SimpleDateFormat("HHmmss", Locale.US).format(Date())
        val ext = when (lang) {
            "python", "py" -> "py"
            "bash", "sh", "shell" -> "sh"
            "json" -> "json"
            "kotlin", "kt" -> "kt"
            "java" -> "java"
            "c" -> "c"
            "cpp", "c++" -> "cpp"
            "html" -> "html"
            "css" -> "css"
            "javascript", "js" -> "js"
            "typescript", "ts" -> "ts"
            "markdown", "md" -> "md"
            "sql" -> "sql"
            "xml" -> "xml"
            "yaml", "yml" -> "yaml"
            else -> "txt"
        }
        return if (ext == "py") "script_$time.py" else "code_${index}_$time.$ext"
    }

    /**
     * Сохранение (скачивание) сгенерированного кода в память смартфона (/sdcard/Download).
     */
    fun saveCodeToDownloads(
        context: Context,
        code: String,
        suggestedName: String? = null,
        language: String? = null
    ): File {
        val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        if (!downloadsDir.exists()) {
            downloadsDir.mkdirs()
        }

        val filename = if (!suggestedName.isNullOrBlank() && suggestedName.contains(".")) {
            suggestedName
        } else {
            generateDefaultFilename(language ?: "python", 1)
        }

        val targetFile = File(downloadsDir, filename)
        targetFile.writeText(code, Charsets.UTF_8)

        // Оповещаем систему о новом файле, чтобы он сразу появился в Downloads
        try {
            MediaScannerConnection.scanFile(
                context,
                arrayOf(targetFile.absolutePath),
                null
            ) { path, uri ->
                Log.i(TAG, "Файл успешно просканирован медиа-сканером: $path, $uri")
            }
        } catch (_: Exception) {}

        return targetFile
    }

    /**
     * Копирование текста в системный буфер обмена.
     */
    fun copyToClipboard(context: Context, text: String, label: String = "LLM Code") {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText(label, text)
        clipboard.setPrimaryClip(clip)
    }
}
