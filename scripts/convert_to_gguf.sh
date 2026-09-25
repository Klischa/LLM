#!/usr/bin/env bash
# ==============================================================================
# Скрипт конвертации HuggingFace модели в GGUF Q4_K_M для Android (llama.cpp)
# ==============================================================================

set -euo pipefail

MODEL_DIR=${1:-"outputs/qwen2.5-1.5b-notes-merged"}
OUTPUT_DIR=${2:-"models_gguf"}
LLAMA_CPP_DIR=${3:-"llama.cpp"}

mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo " 1. Проверка и сборка llama.cpp"
echo "========================================================"
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Клонирование llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
fi

if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    echo "Сборка утилит llama.cpp через CMake..."
    cmake -B "$LLAMA_CPP_DIR/build" -S "$LLAMA_CPP_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_CPP_DIR/build" --config Release -j $(nproc) --target llama-quantize llama-cli llama-gguf-dump
fi

echo "Установка зависимостей конвертера..."
pip install -q -r "$LLAMA_CPP_DIR/requirements.txt"

echo "========================================================"
echo " 2. Конвертация HuggingFace Safetensors -> GGUF (FP16)"
echo "========================================================"
GGUF_F16="$OUTPUT_DIR/qwen2.5-1.5b-notes-f16.gguf"

python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
    --outfile "$GGUF_F16" \
    --outtype f16

echo "GGUF FP16 создан: $GGUF_F16"
ls -lh "$GGUF_F16"

echo "========================================================"
echo " 3. Квантование GGUF в Q4_K_M (4-bit K-quants)"
echo "========================================================"
GGUF_Q4KM="$OUTPUT_DIR/qwen2.5-1.5b-notes-q4_k_m.gguf"

"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_F16" "$GGUF_Q4KM" Q4_K_M

echo "Квантованная модель готова: $GGUF_Q4KM"
ls -lh "$GGUF_Q4KM"

# Опционально: удаление тяжелого FP16 промежуточного файла для экономии диска
# rm -f "$GGUF_F16"

echo "========================================================"
echo " 4. Проверка метаданных и тестовый инференс"
echo "========================================================"
echo "Метаданные GGUF:"
"$LLAMA_CPP_DIR/build/bin/llama-gguf-dump" "$GGUF_Q4KM" | head -n 25

echo "Тестовый запуск (50 токенов):"
"$LLAMA_CPP_DIR/build/bin/llama-cli" \
    -m "$GGUF_Q4KM" \
    -p "<|im_start|>system\nТы — полезный русский ассистент.<|im_end|>\n<|im_start|>user\nНазови 3 правила хорошей заметки.<|im_end|>\n<|im_start|>assistant\n" \
    -n 50 \
    --temp 0.3 \
    -t 4

echo "========================================================"
echo " 5. Команда для переноса на телефон (Infinix Note 30):"
echo "========================================================"
echo "adb push $GGUF_Q4KM /sdcard/Download/qwen2.5-1.5b-notes-q4_k_m.gguf"
echo "========================================================"
