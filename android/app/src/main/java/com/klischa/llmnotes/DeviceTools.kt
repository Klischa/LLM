package com.klischa.llmnotes

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class DeviceTelemetry(
    val photoCount: Int,
    val videoCount: Int,
    val audioCount: Int,
    val storageFreeGb: Float,
    val storageTotalGb: Float,
    val storageUsedGb: Float,
    val storageUsedPercent: Int,
    val ramFreeGb: Float,
    val ramTotalGb: Float,
    val batteryPercent: Int,
    val isCharging: Boolean,
    val batteryTempC: Float?,
    val deviceModel: String,
    val androidVersion: String,
    val cpuCores: Int,
    val currentDateTime: String
)

/**
 * Системные инструменты устройства (Device Tools).
 * Обеспечивают сбор телеметрии устройства (фото, видео, память, батарея, характеристики)
 * для передачи в системный контекст модели LLM.
 */
object DeviceTools {
    private const val TAG = "DeviceTools"

    fun getDeviceTelemetry(context: Context): DeviceTelemetry {
        val (photos, videos, audio) = getMediaCounts(context)
        val (storageFree, storageTotal) = getStorageStats()
        val storageUsed = (storageTotal - storageFree).coerceAtLeast(0f)
        val storagePct = if (storageTotal > 0) ((storageUsed / storageTotal) * 100).toInt() else 0

        val (ramFree, ramTotal) = getRamStats(context)
        val (batteryPct, isCharging, batteryTemp) = getBatteryStats(context)

        val dateFormat = SimpleDateFormat("EEEE, d MMMM yyyy г., HH:mm", Locale("ru"))
        val currentDateTime = dateFormat.format(Date())

        val deviceModel = "${Build.MANUFACTURER.replaceFirstChar { it.uppercase() }} ${Build.MODEL}"
        val androidVersion = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})"
        val cpuCores = Runtime.getRuntime().availableProcessors()

        return DeviceTelemetry(
            photoCount = photos,
            videoCount = videos,
            audioCount = audio,
            storageFreeGb = storageFree,
            storageTotalGb = storageTotal,
            storageUsedGb = storageUsed,
            storageUsedPercent = storagePct,
            ramFreeGb = ramFree,
            ramTotalGb = ramTotal,
            batteryPercent = batteryPct,
            isCharging = isCharging,
            batteryTempC = batteryTemp,
            deviceModel = deviceModel,
            androidVersion = androidVersion,
            cpuCores = cpuCores,
            currentDateTime = currentDateTime
        )
    }

    /**
     * Формирование блока контекста телеметрии для системного промпта.
     */
    fun buildTelemetryContext(context: Context): String {
        val t = getDeviceTelemetry(context)
        val chargingStr = if (t.isCharging) "заряжается" else "разряжается"
        val tempStr = if (t.batteryTempC != null) ", ${String.format(Locale.US, "%.1f", t.batteryTempC)}°C" else ""

        return """
[Системные данные устройства пользователя]:
- Устройство: ${t.deviceModel} (${t.androidVersion}, ${t.cpuCores} ядер CPU)
- Количество фото на телефоне: ${t.photoCount}
- Количество видео на телефоне: ${t.videoCount}
- Количество аудиофайлов: ${t.audioCount}
- Память хранилища: свободно ${String.format(Locale.US, "%.1f", t.storageFreeGb)} ГБ из ${String.format(Locale.US, "%.1f", t.storageTotalGb)} ГБ (занято ${t.storageUsedPercent}%)
- Оперативная память (RAM): свободно ${String.format(Locale.US, "%.1f", t.ramFreeGb)} ГБ из ${String.format(Locale.US, "%.1f", t.ramTotalGb)} ГБ
- Аккумулятор: ${t.batteryPercent}% ($chargingStr$tempStr)
- Текущая дата и время: ${t.currentDateTime}
(Если пользователь спрашивает о фотографиях, памяти, батарее или телефоне, отвечай, используя эти точные данные)
""".trimIndent()
    }

    /**
     * Проверка, относится ли запрос пользователя к системным данным устройства.
     */
    fun isDeviceQuery(query: String): Boolean {
        val q = query.lowercase()
        val keywords = listOf(
            "фото", "снимк", "видео", "галере", "медиа", "изображен",
            "памят", "место", "гигабайт", "диск", "накопител", "хранилищ", "мб", "гб",
            "батаре", "заряд", "аккумулятор", "процент",
            "телефон", "устройств", "модель", "характеристик", "процессор", "ядер", "андроид",
            "врем", "дата", "число", "день", "сегодня", "сейчас"
        )
        return keywords.any { q.contains(it) }
    }

    fun getMediaCounts(context: Context): Triple<Int, Int, Int> {
        var photos = 0
        var videos = 0
        var audio = 0

        // 1. Попытка запроса через Android MediaStore
        try {
            val imgCursor = context.contentResolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                arrayOf(MediaStore.Images.Media._ID),
                null, null, null
            )
            photos = imgCursor?.use { it.count } ?: 0

            val vidCursor = context.contentResolver.query(
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                arrayOf(MediaStore.Video.Media._ID),
                null, null, null
            )
            videos = vidCursor?.use { it.count } ?: 0

            val audCursor = context.contentResolver.query(
                MediaStore.Audio.Media.EXTERNAL_CONTENT_URI,
                arrayOf(MediaStore.Audio.Media._ID),
                null, null, null
            )
            audio = audCursor?.use { it.count } ?: 0
        } catch (e: Exception) {
            Log.w(TAG, "Ошибка чтения MediaStore: ${e.message}")
        }

        // 2. Резервный подсчет прямым сканированием DCIM и Pictures, если MediaStore вернул 0
        if (photos == 0 && videos == 0) {
            val (dirPhotos, dirVideos) = scanMediaDirectories()
            if (dirPhotos > 0 || dirVideos > 0) {
                photos = dirPhotos
                videos = dirVideos
            }
        }

        return Triple(photos, videos, audio)
    }

    private fun scanMediaDirectories(): Pair<Int, Int> {
        var photos = 0
        var videos = 0
        val photoExts = setOf("jpg", "jpeg", "png", "webp", "heic", "heif", "dng", "bmp")
        val videoExts = setOf("mp4", "mkv", "mov", "avi", "3gp", "webm")

        val targetDirs = listOf(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM),
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            File(Environment.getExternalStorageDirectory(), "DCIM/Camera")
        )

        for (dir in targetDirs) {
            try {
                if (dir.exists() && dir.isDirectory) {
                    dir.walkTopDown().maxDepth(4).forEach { file ->
                        if (file.isFile) {
                            val ext = file.extension.lowercase()
                            if (ext in photoExts) photos++
                            else if (ext in videoExts) videos++
                        }
                    }
                }
            } catch (_: Exception) {}
        }
        return Pair(photos, videos)
    }

    private fun getStorageStats(): Pair<Float, Float> {
        return try {
            val dataDir = Environment.getDataDirectory()
            val stat = StatFs(dataDir.path)
            val blockSize = stat.blockSizeLong
            val totalBytes = stat.blockCountLong * blockSize
            val freeBytes = stat.availableBlocksLong * blockSize

            val totalGb = totalBytes.toFloat() / (1024f * 1024f * 1024f)
            val freeGb = freeBytes.toFloat() / (1024f * 1024f * 1024f)
            Pair(freeGb, totalGb)
        } catch (e: Exception) {
            Pair(0f, 0f)
        }
    }

    private fun getRamStats(context: Context): Pair<Float, Float> {
        return try {
            val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            val memInfo = ActivityManager.MemoryInfo()
            actManager?.getMemoryInfo(memInfo)

            val totalGb = memInfo.totalMem.toFloat() / (1024f * 1024f * 1024f)
            val freeGb = memInfo.availMem.toFloat() / (1024f * 1024f * 1024f)
            Pair(freeGb, totalGb)
        } catch (e: Exception) {
            Pair(0f, 0f)
        }
    }

    private fun getBatteryStats(context: Context): Triple<Int, Boolean, Float?> {
        return try {
            val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            val batteryStatus = context.registerReceiver(null, filter)

            val level = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
            val batteryPct = if (level >= 0 && scale > 0) ((level.toFloat() / scale.toFloat()) * 100).toInt() else 0

            val status = batteryStatus?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
            val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                             status == BatteryManager.BATTERY_STATUS_FULL

            val tempTenths = batteryStatus?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
            val tempC = if (tempTenths > 0) tempTenths / 10.0f else null

            Triple(batteryPct, isCharging, tempC)
        } catch (e: Exception) {
            Triple(0, false, null)
        }
    }
}
