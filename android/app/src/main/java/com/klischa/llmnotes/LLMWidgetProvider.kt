package com.klischa.llmnotes

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.widget.RemoteViews

/**
 * Виджет для рабочего стола Android.
 * Обеспечивает минималистичный интерфейс:
 * - Отображает текущую активную модель и последний полученный ответ.
 * - При нажатии на строку ввода моментально открывает всплывающее окно быстрого запроса
 *   (QuickPromptActivity) без необходимости полного открытия приложения.
 */
class LLMWidgetProvider : AppWidgetProvider() {

    override fun onUpdate(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray
    ) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val lastResponse = prefs.getString(PREF_WIDGET_LAST_RESPONSE, null)
        val lastModel = prefs.getString(PREF_WIDGET_LAST_MODEL, "LLM • Офлайн / Облако")

        for (appWidgetId in appWidgetIds) {
            updateAppWidget(context, appWidgetManager, appWidgetId, lastResponse, lastModel)
        }
    }

    companion object {
        const val PREFS_NAME = "llm_widget_prefs"
        const val PREF_WIDGET_LAST_RESPONSE = "widget_last_response"
        const val PREF_WIDGET_LAST_MODEL = "widget_last_model"
        const val PREF_WIDGET_LAST_QUERY = "widget_last_query"

        fun updateAppWidget(
            context: Context,
            appWidgetManager: AppWidgetManager,
            appWidgetId: Int,
            response: String?,
            modelName: String?
        ) {
            val views = RemoteViews(context.packageName, R.layout.widget_llm_minimal)

            // Отображение названия модели
            views.setTextViewText(
                R.id.widget_model_badge,
                modelName ?: "Офлайн GGUF"
            )

            // Отображение последнего ответа
            val displayText = if (!response.isNullOrBlank()) {
                response.trim()
            } else {
                "Нажмите здесь или на строку ниже, чтобы задать вопрос модели..."
            }
            views.setTextViewText(R.id.widget_text_response, displayText)

            // Интент для открытия компактного окна быстрого запроса
            val quickIntent = Intent(context, QuickPromptActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val quickPendingIntent = PendingIntent.getActivity(
                context,
                0,
                quickIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            // Привязка клика по строке ввода, кнопке отправки и быстрому запросу
            views.setOnClickPendingIntent(R.id.widget_input_bar, quickPendingIntent)
            views.setOnClickPendingIntent(R.id.widget_btn_send, quickPendingIntent)
            views.setOnClickPendingIntent(R.id.widget_btn_quick, quickPendingIntent)
            views.setOnClickPendingIntent(R.id.widget_text_response, quickPendingIntent)

            appWidgetManager.updateAppWidget(appWidgetId, views)
        }

        /**
         * Метод обновления всех экземпляров виджета на рабочем столе из приложения.
         */
        fun updateAllWidgets(
            context: Context,
            query: String,
            response: String,
            modelName: String? = null
        ) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().apply {
                putString(PREF_WIDGET_LAST_QUERY, query)
                putString(PREF_WIDGET_LAST_RESPONSE, response)
                if (modelName != null) {
                    putString(PREF_WIDGET_LAST_MODEL, modelName)
                }
                apply()
            }

            val appWidgetManager = AppWidgetManager.getInstance(context)
            val componentName = ComponentName(context, LLMWidgetProvider::class.java)
            val appWidgetIds = appWidgetManager.getAppWidgetIds(componentName)

            val displayModel = modelName ?: prefs.getString(PREF_WIDGET_LAST_MODEL, "Офлайн GGUF")
            for (widgetId in appWidgetIds) {
                updateAppWidget(context, appWidgetManager, widgetId, response, displayModel)
            }
        }
    }
}
