#!/usr/bin/env python3
"""
Скрипт создания калибровочного набора данных для расчета матрицы важности (imatrix) в llama.cpp.
Использование imatrix позволяет существенно снизить деградацию перплексии русского языка
при 4-битном квантовании (Q4_K_M / IQ4_XS), сохраняя правильные падежные окончания и структуру предложений.
"""

import os
import argparse
import json
from datasets import load_dataset

def prepare_calibration_corpus(output_file: str, max_tokens: int = 150000):
    print(f"Формирование калибровочного корпуса для llama-imatrix (целевой объем: ~{max_tokens} токенов)...")
    
    samples = []
    
    # 1. Тексты из открытого русскоязычного датасета
    try:
        ds = load_dataset("IlyaGusev/saiga_scored", split="train", streaming=True)
        token_estimate = 0
        for item in ds:
            text = ""
            if "messages" in item:
                text = " ".join([m["content"] for m in item["messages"]])
            elif "instruction" in item and "output" in item:
                text = f"{item['instruction']}\n{item.get('input', '')}\n{item['output']}"
            
            if len(text.strip()) > 50:
                samples.append(text.strip())
                token_estimate += len(text.split()) * 1.3  # примерная оценка токенов
                
            if token_estimate >= max_tokens:
                break
        print(f"Собрано {len(samples)} текстов из открытых датасетов (~{int(token_estimate)} токенов).")
    except Exception as e:
        print(f"Внимание: Не удалось загрузить внешний датасет ({e}). Используем локальные примеры.")
        samples = [
            "Заметки и планирование задач: созвон по архитектуре LLM в 15:00. Подготовить бенчмарк инференса для процессора Helio G99.",
            "Проверить расход оперативной памяти: модель Qwen2.5-1.5B в квантовании Q4_K_M занимает около 980 МБ весов и 56 МБ KV-кэша.",
            "Для предотвращения термического троттлинга на смартфоне Infinix Note 30 необходимо ограничить инференс двумя потоками на ядрах Cortex-A76.",
            "Краткое содержание встречи: согласовали формат хранения локальных заметок, шифрование базы данных SQLCipher и офлайн-поиск."
        ] * 100

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s + "\n\n")

    print(f"Калибровочный файл успешно сохранен: {output_file}")
    print("\nИнструкция для запуска llama-imatrix:")
    print(f"./llama.cpp/build/bin/llama-imatrix -m models_gguf/qwen2.5-1.5b-f16.gguf -f {output_file} -o models_gguf/imatrix.dat --chunks 64")
    print("./llama.cpp/build/bin/llama-quantize --imatrix models_gguf/imatrix.dat models_gguf/qwen2.5-1.5b-f16.gguf models_gguf/qwen2.5-1.5b-imatrix-q4_k_m.gguf Q4_K_M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка калибровочного корпуса для llama-imatrix")
    parser.add_argument("--output", type=str, default="data/calibration_ru.txt", help="Выходной текстовый файл")
    args = parser.parse_args()
    prepare_calibration_corpus(args.output)
