#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <android/log.h>
#include "llama.h"

#define TAG "LlamaJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model   * g_model   = nullptr;
static llama_context * g_context = nullptr;
static std::atomic<bool> g_stop_requested(false);

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_klischa_llmnotes_LlamaBridge_nativeInit(JNIEnv *env, jobject thiz) {
    LOGI("Инициализация llama_backend_init()");
    llama_backend_init();
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_klischa_llmnotes_LlamaBridge_nativeLoadModel(
    JNIEnv *env,
    jobject thiz,
    jstring model_path,
    jint n_ctx,
    jint n_threads
) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    if (!path) {
        LOGE("Не удалось прочитать путь к модели");
        return JNI_FALSE;
    }

    LOGI("Загрузка модели из: %s (контекст: %d, потоков: %d)", path, n_ctx, n_threads);

    // Освобождаем старую модель, если была загружена
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }

    // Параметры модели: mmap включен для быстрой загрузки и экономии ОЗУ
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true;
    model_params.use_mlock = false; // На Android mlock не рекомендуется (LMK)

    g_model = llama_load_model_from_file(path, model_params);
    env->ReleaseStringUTFChars(model_path, path);

    if (!g_model) {
        LOGE("Ошибка: не удалось загрузить модель GGUF");
        return JNI_FALSE;
    }

    // Параметры контекста
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx > 0 ? n_ctx : 2048;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = n_threads > 0 ? n_threads : 2;       // Оптимум 2 ядра A76 на Helio G99
    ctx_params.n_threads_batch = n_threads > 0 ? n_threads : 2;

    g_context = llama_new_context_with_model(g_model, ctx_params);
    if (!g_context) {
        LOGE("Ошибка: не удалось создать контекст llama");
        llama_free_model(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Модель успешно загружена в память!");
    return JNI_TRUE;
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

    // Подготовка Callback метода Java onToken(String token)
    jclass callback_class = nullptr;
    jmethodID on_token_method = nullptr;
    if (callback_obj != nullptr) {
        callback_class = env->GetObjectClass(callback_obj);
        on_token_method = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
    }

    // Токенизация входного промпта
    const struct llama_vocab * vocab = llama_model_get_vocab(g_model);
    int n_prompt_tokens = -llama_tokenize(vocab, prompt, strlen(prompt), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt, strlen(prompt), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        env->ReleaseStringUTFChars(prompt_str, prompt);
        return env->NewStringUTF("Ошибка: сбой токенизации");
    }

    env->ReleaseStringUTFChars(prompt_str, prompt);

    // Инициализация батча и кэша KV
    llama_kv_cache_clear(g_context);
    llama_batch batch = llama_batch_init(512, 0, 1);

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        llama_batch_add(batch, prompt_tokens[i], i, { 0 }, false);
    }
    // Для последнего токена входного промпта нужны логиты
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(g_context, batch) != 0) {
        LOGE("Ошибка: сбой вычисления промпта llama_decode");
        llama_batch_free(batch);
        return env->NewStringUTF("Ошибка вычисления промпта");
    }

    // Инициализация сэмплера
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    std::string full_response = "";
    int n_cur = batch.n_tokens;
    int generated_count = 0;

    // Цикл генерации
    while (generated_count < max_tokens && !g_stop_requested.load()) {
        llama_token new_token_id = llama_sampler_sample(smpl, g_context, -1);
        llama_sampler_accept(smpl, new_token_id);

        // Проверка конца генерации (EOS / EOT токен)
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // Преобразование токена в строку
        char piece_buf[256];
        int n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (n_chars > 0) {
            std::string piece(piece_buf, n_chars);
            full_response += piece;

            // Вызов потокового callback в Kotlin
            if (callback_obj != nullptr && on_token_method != nullptr) {
                jstring piece_jstr = env->NewStringUTF(piece.c_str());
                env->CallVoidMethod(callback_obj, on_token_method, piece_jstr);
                env->DeleteLocalRef(piece_jstr);
            }
        }

        // Подготовка следующего шага декодирования
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

        if (llama_decode(g_context, batch) != 0) {
            LOGE("Ошибка: сбой декодирования токена");
            break;
        }

        n_cur++;
        generated_count++;
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return env->NewStringUTF(full_response.c_str());
}

} // extern "C"
