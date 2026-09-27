package com.klischa.llmnotes

data class ParsedMessage(
    val thinkingText: String? = null,
    val answerText: String = "",
    val isStillThinking: Boolean = false
)

object ThoughtParser {
    val STOP_MARKERS = listOf(
        "<|",
        "<｜",
        "<|im_end|>",
        "<|im_end|",
        "<|im_start|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<|eot_id|>",
        "<end_of_turn>",
        "</s>",
        "<｜end of sentence｜>",
        "<｜Assistant｜>",
        "<｜User｜>",
        "[|im_end|]"
    )

    /**
     * Обрезает строку при первом обнаружении любого служебного стоп-токена диалога
     * и очищает висячие префиксы тегов (<, |, ｜) и шаблонные заполнители (|name|)
     */
    fun cleanStopTokens(raw: String): String {
        var text = raw.replace("|name|", "LLM")
        for (marker in STOP_MARKERS) {
            val idx = text.indexOf(marker)
            if (idx != -1) {
                text = text.substring(0, idx)
            }
        }
        text = text.trimEnd()
        // Устраняем висячие служебные префиксы в самом конце текста
        while (text.endsWith("<") || text.endsWith("|") || text.endsWith("｜")) {
            text = text.dropLast(1).trimEnd()
        }
        return text
    }

    /**
     * Разделяет рассуждения модели (<think>...</think>) и окончательный ответ пользователю
     */
    fun parse(rawText: String, isStreaming: Boolean = false): ParsedMessage {
        val cleaned = cleanStopTokens(rawText)

        // Случай 1: Присутствует закрывающий тег </think> (модель завершила рассуждения и выдала ответ)
        if (cleaned.contains("</think>")) {
            val parts = cleaned.split("</think>", limit = 2)
            val thought = cleanStopTokens(parts[0].replace("<think>", "").trim())
            val answer = cleanStopTokens(parts[1].trim())
            return ParsedMessage(
                thinkingText = thought.ifBlank { null },
                answerText = answer,
                isStillThinking = false
            )
        }

        // Случай 2: Текст начинается с <think>, но закрывающего тега еще нет (идет процесс размышления)
        if (cleaned.startsWith("<think>")) {
            val thought = cleanStopTokens(cleaned.removePrefix("<think>").trim())
            return ParsedMessage(
                thinkingText = thought.ifBlank { null },
                answerText = "",
                isStillThinking = isStreaming
            )
        }

        // Случай 3: Обычный ответ без рассуждений
        return ParsedMessage(
            thinkingText = null,
            answerText = cleaned,
            isStillThinking = false
        )
    }
}
