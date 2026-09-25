#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <cstring>
#include <cerrno>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <android/log.h>
#include "llama.h"

#define TAG "LlamaJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model   * g_model   = nullptr;
static llama_context * g_context = nullptr;
static std::atomic<bool> g_stop_requested(false);
static std::string g_last_error = "";

static void llama_log_callback_capture(enum ggml_log_level level, const char * text, void * user_data) {
    if (text) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            LOGE("%s", text);
            g_last_error += text;
        } else if (level == GGML_LOG_LEVEL_WARN) {
            LOGE("%s", text);
        } else {
            LOGI("%s", text);
        }
    }
}

static void batch_add_token(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_klischa_llmnotes_LlamaBridge_nativeInit(JNIEnv *env, jobject thiz) {
    LOGI("Инициализация llama_backend_init()");
    llama_backend_init();
    llama_log_set(llama_log_callback_capture, nullptr);
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_klischa_llmnotes_LlamaBridge_nativeLoadModel(
    JNIEnv *env,
    jobject thiz,
    jstring model_path,
    jint n_ctx,
    jint n_threads
) {
    g_last_error.clear();

    const char *path = env->GetStringUTFChars(model_path, nullptr);
    if (!path) {
        return env->NewStringUTF("Не удалось прочитать путь к файлу модели");
    }

    LOGI("Загрузка модели из: %s (контекст: %d, потоков: %d)", path, n_ctx, n_threads);

    // 1. Проверка существования и размера файла
    struct stat st;
    bool stat_ok = false;
    if (strncmp(path, "/proc/self/fd/", 14) == 0) {
        int fd = atoi(path + 14);
        stat_ok = (fstat(fd, &st) == 0);
    } else {
        stat_ok = (stat(path, &st) == 0);
    }

    if (!stat_ok) {
        // Попытка прямого fopen, если stat не сработал
        FILE * f = fopen(path, "rb");
        if (!f) {
            std::string err = "Не удалось открыть файл: " + std::string(strerror(errno));
            LOGE("%s (%s)", err.c_str(), path);
            env->ReleaseStringUTFChars(model_path, path);
            return env->NewStringUTF(err.c_str());
        }
        fseek(f, 0, SEEK_END);
        st.st_size = ftell(f);
        fclose(f);
    }

    if (st.st_size < 1024 * 1024) {
        std::string err = "Файл поврежден или недокачан (размер: " + std::to_string(st.st_size / 1024) + " КБ)";
        LOGE("%s", err.c_str());
        env->ReleaseStringUTFChars(model_path, path);
        return env->NewStringUTF(err.c_str());
    }

    // Освобождение ранее загруженных ресурсов
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }

    // 2. Загрузка модели (сначала пробуем use_mmap = true, при ошибке — use_mmap = false)
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    g_model = llama_load_model_from_file(path, model_params);
    if (!g_model) {
        LOGI("Загрузка с use_mmap=true не удалась, повторная попытка с use_mmap=false...");
        g_last_error.clear();
        model_params.use_mmap = false;
        g_model = llama_load_model_from_file(path, model_params);
    }
    env->ReleaseStringUTFChars(model_path, path);

    if (!g_model) {
        std::string err = "Ошибка llama_load_model_from_file: " + (g_last_error.empty() ? "неверный формат GGUF или поврежденный файл" : g_last_error);
        LOGE("%s", err.c_str());
        return env->NewStringUTF(err.c_str());
    }

    // 3. Создание контекста с адаптивным размером (для моделей 3B-4B)
    int target_ctx = n_ctx > 0 ? n_ctx : 2048;
    int ctx_attempts[] = { target_ctx, 1024, 512 };

    for (int ctx_size : ctx_attempts) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = ctx_size;
        ctx_params.n_batch = 512;
        ctx_params.n_threads = n_threads > 0 ? n_threads : 2;
        ctx_params.n_threads_batch = n_threads > 0 ? n_threads : 2;

        g_context = llama_new_context_with_model(g_model, ctx_params);
        if (g_context) {
            LOGI("Контекст llama успешно создан (n_ctx = %d)!", ctx_size);
            break;
        }
    }

    if (!g_context) {
        llama_free_model(g_model);
        g_model = nullptr;
        std::string err = "Недостаточно памяти для создания контекста: " + g_last_error;
        LOGE("%s", err.c_str());
        return env->NewStringUTF(err.c_str());
    }

    LOGI("Модель успешно загружена в память!");
    return env->NewStringUTF(""); // Пустая строка означает успех
}

JNIEXPORT void JNICALL
Java_com_klischa_llmnotes_LlamaBridge_nativeStop(JNIEnv *env, jobject thiz) {
    LOGI("Запрос на прерывание инференса");
    g_stop_requested.store(true);
}

JNIEXPORT void JNICALL
Java_com_klischa_llmnotes_LlamaBridge_nativeUnload(JNIEnv *env, jobject thiz) {
    LOGI("Выгрузка модели и освобождение памяти");
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
}

JNIEXPORT jstring JNICALL
Java_com_klischa_llmnotes_LlamaBridge_nativeFormatPrompt(
    JNIEnv *env,
    jobject thiz,
    jstring system_prompt_str,
    jstring user_prompt_str
) {
    if (!g_model) return user_prompt_str;

    const char *sys_ptr = system_prompt_str ? env->GetStringUTFChars(system_prompt_str, nullptr) : nullptr;
    const char *usr_ptr = user_prompt_str ? env->GetStringUTFChars(user_prompt_str, nullptr) : nullptr;

    std::string sys_str = sys_ptr ? sys_ptr : "";
    std::string usr_str = usr_ptr ? usr_ptr : "";

    if (sys_ptr) env->ReleaseStringUTFChars(system_prompt_str, sys_ptr);
    if (usr_ptr) env->ReleaseStringUTFChars(user_prompt_str, usr_ptr);

    std::vector<llama_chat_message> chat;
    if (!sys_str.empty()) {
        chat.push_back({"system", sys_str.c_str()});
    }
    if (!usr_str.empty()) {
        chat.push_back({"user", usr_str.c_str()});
    }

    if (chat.empty()) {
        return env->NewStringUTF("");
    }

    int32_t req_len = llama_chat_apply_template(g_model, nullptr, chat.data(), chat.size(), true, nullptr, 0);

    if (req_len > 0) {
        std::vector<char> buf(req_len + 1, 0);
        int32_t res_len = llama_chat_apply_template(g_model, nullptr, chat.data(), chat.size(), true, buf.data(), buf.size());
        if (res_len > 0) {
            return env->NewStringUTF(std::string(buf.data(), res_len).c_str());
        }
    }

    std::string fallback = "";
    if (!sys_str.empty()) {
        fallback += "<|im_start|>system\n" + sys_str + "<|im_end|>\n";
    }
    fallback += "<|im_start|>user\n" + usr_str + "<|im_end|>\n<|im_start|>assistant\n";

    return env->NewStringUTF(fallback.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_klischa_llmnotes_LlamaBridge_nativeGenerate(
    JNIEnv *env,
    jobject thiz,
    jstring prompt_str,
    jint max_tokens,
    jfloat temperature,
    jfloat top_p,
    jobject callback_obj
) {
    if (!g_model || !g_context) {
        LOGE("Модель не инициализирована!");
        return env->NewStringUTF("Ошибка: модель не загружена.");
    }

    const char *prompt = env->GetStringUTFChars(prompt_str, nullptr);
    if (!prompt) return env->NewStringUTF("");

    g_stop_requested.store(false);

    jclass callback_class = nullptr;
    jmethodID on_token_method = nullptr;
    if (callback_obj != nullptr) {
        callback_class = env->GetObjectClass(callback_obj);
        on_token_method = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
    }

    // Токенизация входного промпта
    int n_prompt_tokens = -llama_tokenize(g_model, prompt, strlen(prompt), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(g_model, prompt, strlen(prompt), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        env->ReleaseStringUTFChars(prompt_str, prompt);
        return env->NewStringUTF("Ошибка токенизации");
    }

    env->ReleaseStringUTFChars(prompt_str, prompt);

    // Подготовка KV-кэша и батча
    llama_kv_cache_clear(g_context);
    struct llama_batch batch = llama_batch_init(512, 0, 1);

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        bool need_logits = (i == prompt_tokens.size() - 1);
        batch_add_token(batch, prompt_tokens[i], i, { 0 }, need_logits);
    }

    if (llama_decode(g_context, batch) != 0) {
        LOGE("Ошибка: сбой llama_decode для промпта");
        llama_batch_free(batch);
        return env->NewStringUTF("Ошибка декодирования промпта");
    }

    std::string full_response = "";
    int n_cur = batch.n_tokens;
    int generated_count = 0;
    int32_t n_vocab = llama_n_vocab(g_model);

    // Цикл декодирования токенов
    while (generated_count < max_tokens && !g_stop_requested.load()) {
        float * logits = llama_get_logits_ith(g_context, batch.n_tokens - 1);

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        llama_sample_top_p(g_context, &candidates_p, top_p, 1);
        llama_sample_temp(g_context, &candidates_p, temperature);
        llama_token new_token_id = llama_sample_token(g_context, &candidates_p);

        if (llama_token_is_eog(g_model, new_token_id)) {
            break;
        }

        char piece_buf[256];
        int n_chars = llama_token_to_piece(g_model, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (n_chars > 0) {
            std::string piece(piece_buf, n_chars);
            full_response += piece;

            if (callback_obj != nullptr && on_token_method != nullptr) {
                jstring piece_jstr = env->NewStringUTF(piece.c_str());
                env->CallVoidMethod(callback_obj, on_token_method, piece_jstr);
                env->DeleteLocalRef(piece_jstr);
            }
        }

        batch.n_tokens = 0;
        batch_add_token(batch, new_token_id, n_cur, { 0 }, true);

        if (llama_decode(g_context, batch) != 0) {
            LOGE("Ошибка: сбой декодирования токена");
            break;
        }

        n_cur++;
        generated_count++;
    }

    llama_batch_free(batch);
    return env->NewStringUTF(full_response.c_str());
}

} // extern "C"
