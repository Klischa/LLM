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
            if (g_last_error.length() < 2048) {
                g_last_error += text;
            }
        } else if (level == GGML_LOG_LEVEL_WARN) {
            LOGE("%s", text);
            if (g_last_error.length() < 2048) {
                g_last_error += text;
            }
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
        FILE * f = fopen(path, "rb");
        if (!f) {
            std::string err = "Не удалось открыть файл модели: " + std::string(strerror(errno));
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
    LOGI("Выгрузка модели и контекста");
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
    jstring system_prompt,
    jstring user_prompt
) {
    const char *sys_str = env->GetStringUTFChars(system_prompt, nullptr);
    const char *usr_str = env->GetStringUTFChars(user_prompt, nullptr);

    std::string formatted;

    if (g_model) {
        llama_chat_message chat_msgs[2];
        chat_msgs[0].role = "system";
        chat_msgs[0].content = sys_str;
        chat_msgs[1].role = "user";
        chat_msgs[1].content = usr_str;

        std::vector<char> buf(4096);
        int res = llama_chat_apply_template(
            llama_model_chat_template(g_model, nullptr),
            chat_msgs,
            2,
            true,
            buf.data(),
            buf.size()
        );

        if (res > (int)buf.size()) {
            buf.resize(res + 1);
            res = llama_chat_apply_template(
                llama_model_chat_template(g_model, nullptr),
                chat_msgs,
                2,
                true,
                buf.data(),
                buf.size()
            );
        }

        if (res > 0) {
            formatted = std::string(buf.data(), res);
        }
    }

    if (formatted.empty()) {
        formatted = "<|im_start|>system\n" + std::string(sys_str) +
                    "<|im_end|>\n<|im_start|>user\n" + std::string(usr_str) +
                    "<|im_end|>\n<|im_start|>assistant\n";
    }

    env->ReleaseStringUTFChars(system_prompt, sys_str);
    env->ReleaseStringUTFChars(user_prompt, usr_str);

    return env->NewStringUTF(formatted.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_klischa_llmnotes_LlamaBridge_nativeGenerate(
    JNIEnv *env,
    jobject thiz,
    jstring prompt_str,
    jint max_tokens,
    jfloat temperature,
    jfloat top_p,
    jobject callback
) {
    if (!g_model || !g_context) {
        LOGE("Генерация невозможна: модель не загружена");
        return env->NewStringUTF("");
    }

    g_stop_requested.store(false);

    const char *prompt = env->GetStringUTFChars(prompt_str, nullptr);
    std::string full_prompt(prompt);
    env->ReleaseStringUTFChars(prompt_str, prompt);

    jclass callback_class = nullptr;
    jmethodID on_token_method = nullptr;
    if (callback != nullptr) {
        callback_class = env->GetObjectClass(callback);
        on_token_method = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
    }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);

    // Токенизация входного промпта
    int n_tokens_max = full_prompt.length() + 32;
    std::vector<llama_token> prompt_tokens(n_tokens_max);
    int n_tokens = llama_tokenize(
        vocab,
        full_prompt.c_str(),
        full_prompt.length(),
        prompt_tokens.data(),
        prompt_tokens.size(),
        true,
        true
    );

    if (n_tokens < 0) {
        prompt_tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            vocab,
            full_prompt.c_str(),
            full_prompt.length(),
            prompt_tokens.data(),
            prompt_tokens.size(),
            true,
            true
        );
    }
    prompt_tokens.resize(n_tokens);

    LOGI("Входной промпт токенизирован в %d токенов", n_tokens);

    // Подготовка KV-кэша и батча
    llama_kv_cache_clear(g_context);

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch_add_token(batch, prompt_tokens[i], i, {0}, false);
        if (batch.n_tokens == 512 || i == n_tokens - 1) {
            if (llama_decode(g_context, batch) != 0) {
                LOGE("Ошибка декодирования входного промпта");
                llama_batch_free(batch);
                return env->NewStringUTF("");
            }
            batch.n_tokens = 0;
        }
    }

    // Инициализация сэмплера
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string result_text = "";
    int n_cur = n_tokens;
    int generated_count = 0;

    while (generated_count < max_tokens && !g_stop_requested.load()) {
        llama_token new_token_id = llama_sampler_sample(smpl, g_context, -1);
        llama_sampler_accept(smpl, new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            LOGI("Встречен токен конца последовательности (EOG)");
            break;
        }

        char piece_buf[256];
        int n_piece = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, false);
        if (n_piece > 0) {
            std::string piece(piece_buf, n_piece);
            result_text += piece;

            if (callback != nullptr && on_token_method != nullptr) {
                jstring jpiece = env->NewStringUTF(piece.c_str());
                env->CallVoidMethod(callback, on_token_method, jpiece);
                env->DeleteLocalRef(jpiece);
            }
        }

        generated_count++;
        batch.n_tokens = 0;
        batch_add_token(batch, new_token_id, n_cur, {0}, true);
        n_cur++;

        if (llama_decode(g_context, batch) != 0) {
            LOGE("Ошибка декодирования сгенерированного токена");
            break;
        }
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    LOGI("Генерация завершена. Сгенерировано %d токенов", generated_count);
    return env->NewStringUTF(result_text.c_str());
}

} // extern "C"
