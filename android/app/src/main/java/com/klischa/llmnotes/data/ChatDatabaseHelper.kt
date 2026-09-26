package com.klischa.llmnotes.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.klischa.llmnotes.ChatMessage
import com.klischa.llmnotes.MessageRole

data class ChatSession(
    val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long
)

/**
 * Локальное SQLite хранилище для истории нескольких диалогов.
 * Поддерживает создание новых чатов, переключение между ними, сохранение сообщений и удаление.
 */
class ChatDatabaseHelper(context: Context) : SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    companion object {
        private const val DATABASE_NAME = "llm_chats.db"
        private const val DATABASE_VERSION = 1

        private const val TABLE_CHATS = "chats"
        private const val COL_CHAT_ID = "id"
        private const val COL_CHAT_TITLE = "title"
        private const val COL_CHAT_CREATED = "created_at"
        private const val COL_CHAT_UPDATED = "updated_at"

        private const val TABLE_MESSAGES = "messages"
        private const val COL_MSG_ID = "id"
        private const val COL_MSG_CHAT_ID = "chat_id"
        private const val COL_MSG_ROLE = "role"
        private const val COL_MSG_TEXT = "text"
        private const val COL_MSG_TIME = "timestamp"
        private const val COL_MSG_TPS = "tokens_per_sec"
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE $TABLE_CHATS (
                $COL_CHAT_ID TEXT PRIMARY KEY,
                $COL_CHAT_TITLE TEXT NOT NULL,
                $COL_CHAT_CREATED INTEGER NOT NULL,
                $COL_CHAT_UPDATED INTEGER NOT NULL
            )
            """.trimIndent()
        )
        db.execSQL(
            """
            CREATE TABLE $TABLE_MESSAGES (
                $COL_MSG_ID TEXT PRIMARY KEY,
                $COL_MSG_CHAT_ID TEXT NOT NULL,
                $COL_MSG_ROLE TEXT NOT NULL,
                $COL_MSG_TEXT TEXT NOT NULL,
                $COL_MSG_TIME INTEGER NOT NULL,
                $COL_MSG_TPS REAL,
                FOREIGN KEY($COL_MSG_CHAT_ID) REFERENCES $TABLE_CHATS($COL_CHAT_ID) ON DELETE CASCADE
            )
            """.trimIndent()
        )
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_msg_chat ON $TABLE_MESSAGES($COL_MSG_CHAT_ID)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_MESSAGES")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_CHATS")
        onCreate(db)
    }

    override fun onConfigure(db: SQLiteDatabase) {
        super.onConfigure(db)
        db.setForeignKeyConstraintsEnabled(true)
    }

    fun getAllSessions(): List<ChatSession> {
        val list = mutableListOf<ChatSession>()
        val db = readableDatabase
        val cursor = db.query(
            TABLE_CHATS,
            null,
            null,
            null,
            null,
            null,
            "$COL_CHAT_UPDATED DESC"
        )
        cursor.use {
            val idIdx = it.getColumnIndexOrThrow(COL_CHAT_ID)
            val titleIdx = it.getColumnIndexOrThrow(COL_CHAT_TITLE)
            val createdIdx = it.getColumnIndexOrThrow(COL_CHAT_CREATED)
            val updatedIdx = it.getColumnIndexOrThrow(COL_CHAT_UPDATED)

            while (it.moveToNext()) {
                list.add(
                    ChatSession(
                        id = it.getString(idIdx),
                        title = it.getString(titleIdx),
                        createdAt = it.getLong(createdIdx),
                        updatedAt = it.getLong(updatedIdx)
                    )
                )
            }
        }
        return list
    }

    fun createSession(id: String, title: String): ChatSession {
        val now = System.currentTimeMillis()
        val db = writableDatabase
        val values = ContentValues().apply {
            put(COL_CHAT_ID, id)
            put(COL_CHAT_TITLE, title)
            put(COL_CHAT_CREATED, now)
            put(COL_CHAT_UPDATED, now)
        }
        db.insertWithOnConflict(TABLE_CHATS, null, values, SQLiteDatabase.CONFLICT_REPLACE)
        return ChatSession(id, title, now, now)
    }

    fun updateSessionTitle(id: String, title: String) {
        val db = writableDatabase
        val values = ContentValues().apply {
            put(COL_CHAT_TITLE, title)
            put(COL_CHAT_UPDATED, System.currentTimeMillis())
        }
        db.update(TABLE_CHATS, values, "$COL_CHAT_ID = ?", arrayOf(id))
    }

    fun deleteSession(id: String) {
        val db = writableDatabase
        db.delete(TABLE_MESSAGES, "$COL_MSG_CHAT_ID = ?", arrayOf(id))
        db.delete(TABLE_CHATS, "$COL_CHAT_ID = ?", arrayOf(id))
    }

    fun getMessages(chatId: String): List<ChatMessage> {
        val list = mutableListOf<ChatMessage>()
        val db = readableDatabase
        val cursor = db.query(
            TABLE_MESSAGES,
            null,
            "$COL_MSG_CHAT_ID = ?",
            arrayOf(chatId),
            null,
            null,
            "$COL_MSG_TIME ASC"
        )
        cursor.use {
            val idIdx = it.getColumnIndexOrThrow(COL_MSG_ID)
            val roleIdx = it.getColumnIndexOrThrow(COL_MSG_ROLE)
            val textIdx = it.getColumnIndexOrThrow(COL_MSG_TEXT)
            val timeIdx = it.getColumnIndexOrThrow(COL_MSG_TIME)
            val tpsIdx = it.getColumnIndexOrThrow(COL_MSG_TPS)

            while (it.moveToNext()) {
                val roleStr = it.getString(roleIdx)
                val role = if (roleStr == "USER") MessageRole.USER else MessageRole.ASSISTANT
                val tps = if (it.isNull(tpsIdx)) null else it.getFloat(tpsIdx)

                list.add(
                    ChatMessage(
                        id = it.getString(idIdx),
                        role = role,
                        text = it.getString(textIdx),
                        timestamp = it.getLong(timeIdx),
                        tokensPerSec = tps,
                        isStreaming = false
                    )
                )
            }
        }
        return list
    }

    fun saveMessage(chatId: String, message: ChatMessage) {
        val db = writableDatabase
        val values = ContentValues().apply {
            put(COL_MSG_ID, message.id)
            put(COL_MSG_CHAT_ID, chatId)
            put(COL_MSG_ROLE, message.role.name)
            put(COL_MSG_TEXT, message.text)
            put(COL_MSG_TIME, message.timestamp)
            if (message.tokensPerSec != null) {
                put(COL_MSG_TPS, message.tokensPerSec)
            } else {
                putNull(COL_MSG_TPS)
            }
        }
        db.insertWithOnConflict(TABLE_MESSAGES, null, values, SQLiteDatabase.CONFLICT_REPLACE)

        val updateValues = ContentValues().apply {
            put(COL_CHAT_UPDATED, System.currentTimeMillis())
        }
        db.update(TABLE_CHATS, updateValues, "$COL_CHAT_ID = ?", arrayOf(chatId))
    }
}
