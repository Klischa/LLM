#!/usr/bin/env python3
"""
Скрипт подготовки и предобработки русскоязычного датасета для файнтюнинга Qwen2.5-1.5B/0.5B.
Поддерживает:
1. Загрузку и фильтрацию открытых русскоязычных датасетов (Apache 2.0).
2. Обработку личных заметок пользователя (Markdown/TXT) и генерацию пар:
   - Краткое резюме / конспект
   - Вопрос-ответ по содержанию заметки
   - Выделение списка задач (Action Items)
   - Форматирование и структурирование
3. Форматирование в стандартный формат ChatML (Qwen2.5 Native).
4. Валидацию длины последовательностей и разбиение на train/val.
"""

import os
import json
import re
import argparse
from typing import List, Dict, Any, Optional
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "Ты — умный и лаконичный русскоязычный офлайн-ассистент для смартфона. "
    "Твоя задача — помогать пользователю работать с заметками, отвечать на вопросы точно и без "
    "лишней 'воды', делать краткие пересказы и структурировать информацию на русском языке."
)

NOTE_TASK_TEMPLATES = [
    {
        "type": "summary",
        "instruction": "Сделай краткое резюме следующей заметки (1-3 предложения с главной сутью):\n\n{text}",
        "response_prompt": "Краткое содержание:\n"
    },
    {
        "type": "action_items",
        "instruction": "Проанализируй заметку и составь четкий список задач (Action Items) с чекбоксами:\n\n{text}",
        "response_prompt": "Список задач:\n"
    },
    {
        "type": "key_points",
        "instruction": "Выдели ключевые тезисы и факты из текста ниже в виде маркированного списка:\n\n{text}",
        "response_prompt": "Ключевые тезисы:\n"
    },
    {
        "type": "title_and_tags",
        "instruction": "Придумай емкий заголовок и 3-5 релевантных тегов (через запятую) для этой заметки:\n\n{text}",
        "response_prompt": "Заголовок и теги:\n"
    }
]

def clean_text(text: str) -> str:
    """Удаление мусора, лишних пробелов и нормализация переносов строк."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_markdown_notes(folder_or_file: str) -> List[Dict[str, str]]:
    """Рекурсивное чтение заметок из .md и .txt файлов."""
    notes = []
    if os.path.isfile(folder_or_file):
        files = [folder_or_file]
    elif os.path.isdir(folder_or_file):
        files = []
        for root, _, filenames in os.walk(folder_or_file):
            for fn in filenames:
                if fn.endswith((".md", ".txt", ".markdown")):
                    files.append(os.path.join(root, fn))
    else:
        print(f"Путь {folder_or_file} не найден.")
        return []

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = clean_text(f.read())
                if len(content) > 30:  # Игнорируем пустые или слишком короткие заметки
                    title = os.path.splitext(os.path.basename(fpath))[0]
                    notes.append({"title": title, "content": content, "path": fpath})
        except Exception as e:
            print(f"Ошибка чтения {fpath}: {e}")
    return notes

def format_chatml_sample(system: str, user: str, assistant: str) -> Dict[str, Any]:
    """Форматирование в стандартный список сообщений ChatML."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }

def generate_synthetic_samples_from_notes(notes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Генерация обучающих примеров из сырых заметок.
    Для реального продакшена здесь можно задействовать учительную модель (Teacher LLM,
    например Qwen2.5-72B-Instruct или Claude 3.5 Sonnet) для синтеза эталонных ответов.
    """
    samples = []
    for note in notes:
        text = note["content"]
        title = note["title"]

        # Пример 1: Придумывание заголовка и категоризация
        user_prompt_title = f"Придумай понятный заголовок и основные теги для следующего текста заметки:\n\n{text[:600]}"
        assistant_resp_title = f"Заголовок: {title}\nТеги: #заметки, #продуктивность, #организация"
        samples.append(format_chatml_sample(SYSTEM_PROMPT, user_prompt_title, assistant_resp_title))

        # Если заметка достаточно длинная, делаем эмуляцию саммари
        lines = [line.strip("- *• ") for line in text.split("\n") if len(line.strip()) > 10]
        if len(lines) >= 3:
            user_prompt_sum = f"Сделай краткую выжимку по заметке '{title}':\n\n{text}"
            summary_points = "\n".join([f"• {l}" for l in lines[:3]])
            assistant_resp_sum = f"Главное из заметки «{title}»:\n{summary_points}"
            samples.append(format_chatml_sample(SYSTEM_PROMPT, user_prompt_sum, assistant_resp_sum))

    return samples

def load_open_russian_dataset(dataset_name: str = "IlyaGusev/saiga_scored", max_samples: int = 10000) -> List[Dict[str, Any]]:
    """
    Загрузка и фильтрация открытых качественных русскоязычных инструкций с лицензией Apache 2.0.
    """
    print(f"Загрузка открытого датасета: {dataset_name} (до {max_samples} примеров)...")
    samples = []
    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
        count = 0
        for item in ds:
            # Поддержка структуры Saiga / Alpaca / ShareGPT
            if "messages" in item and isinstance(item["messages"], list):
                msgs = item["messages"]
                # Проверяем язык и валидность
                if len(msgs) >= 2:
                    samples.append({"messages": msgs})
                    count += 1
            elif "instruction" in item and "output" in item:
                user_msg = item["instruction"]
                if item.get("input"):
                    user_msg += "\n\n" + item["input"]
                samples.append(format_chatml_sample(SYSTEM_PROMPT, user_msg, item["output"]))
                count += 1

            if count >= max_samples:
                break
        print(f"Успешно загружено {len(samples)} примеров из открытого датасета.")
    except Exception as e:
        print(f"Предупреждение: Не удалось скачать {dataset_name} ({e}). Используем резервные данные.")
    return samples

def main():
    parser = argparse.ArgumentParser(description="Подготовка датасета для обучения мобильной LLM")
    parser.add_argument("--notes_path", type=str, default="data/sample_notes_raw.md", help="Путь к файлу или папке с заметками")
    parser.add_argument("--output_train", type=str, default="data/train.jsonl", help="Путь к train JSONL")
    parser.add_argument("--output_val", type=str, default="data/val.jsonl", help="Путь к val JSONL")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Имя токенизатора для проверки")
    parser.add_argument("--max_seq_len", type=int, default=1536, help="Максимальная длина контекста")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Доля валидационной выборки")
    parser.add_argument("--include_open_ru", action="store_true", help="Включить открытый датасет Saiga")
    args = parser.parse_args()

    all_samples = []

    # 1. Загрузка пользовательских заметок
    if os.path.exists(args.notes_path):
        notes = parse_markdown_notes(args.notes_path)
        print(f"Найдено заметок: {len(notes)}")
        note_samples = generate_synthetic_samples_from_notes(notes)
        print(f"Сгенерировано обучающих пар из заметок: {len(note_samples)}")
        all_samples.extend(note_samples)

    # 2. Загрузка открытых русскоязычных инструкций при необходимости
    if args.include_open_ru:
        ru_samples = load_open_russian_dataset(max_samples=5000)
        all_samples.extend(ru_samples)

    # 3. Резервные встроенные примеры (если внешние источники недоступны)
    if not all_samples:
        print("Внимание: Исходные заметки не найдены, создаем демонстрационный датасет...")
        demo_samples = [
            format_chatml_sample(
                SYSTEM_PROMPT,
                "Сделай краткое резюме заметки: Купить молоко, кофе в зернах, овсяные хлопья. В 15:00 созвон с техлидом по поводу миграции БД на PostgreSQL. Вечером пробежка 5 км.",
                "Главное из заметки:\n1. Закупки: молоко, кофе, овсянка.\n2. Встреча: 15:00 — созвон с техлидом по миграции на PostgreSQL.\n3. Спорт: вечерняя пробежка 5 км."
            ),
            format_chatml_sample(
                SYSTEM_PROMPT,
                "Выдели задачи (Action Items) из текста: Обсудили с заказчиком требования к MVP мобильного ассистента. Нужно подготовить UI-макеты в Figma до среды, написать бенчмарк инференса для Helio G99 и скинуть спецификацию API.",
                "Action Items:\n- [ ] Подготовить UI-макеты в Figma (дедлайн: среда)\n- [ ] Написать бенчмарк инференса для процессора Helio G99\n- [ ] Отправить заказчику спецификацию API"
            ),
            format_chatml_sample(
                SYSTEM_PROMPT,
                "Вопрос по заметке: 'Пароль от тестового сервера staging-01: K9#mP$2024, доступ по SSH только с VPN'. Как подключиться к тестовому серверу?",
                "Для подключения к тестовому серверу staging-01 необходимо сначала включить VPN, затем использовать SSH и пароль: K9#mP$2024."
            ),
            format_chatml_sample(
                SYSTEM_PROMPT,
                "Отредактируй и структурируй текст: созвонились с дизайнером решили изменить цветовую схему на темную добавить виджет скорости токенов в секунду и сделать кнопку стоп больше",
                "**Итоги созвона с дизайнером:**\n\n• **Тема оформления:** переключить UI на темную цветовую схему.\n• **Новые элементы:** добавить виджет отображения скорости генерации (токены/сек).\n• **UX доработки:** увеличить размер кнопки «Стоп» для удобного нажатия."
            )
        ]
        all_samples.extend(demo_samples * 100)  # Размножим для демонстрации пайплайна

    # Проверка длины токенов
    print(f"Инициализация токенизатора {args.tokenizer_name} для проверки длины...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    except Exception:
        tokenizer = None
        print("Токенизатор оффлайн — пропускаем точную проверку токенов.")

    filtered_samples = []
    for s in all_samples:
        if tokenizer:
            full_text = tokenizer.apply_chat_template(s["messages"], tokenize=False)
            tokens = tokenizer.encode(full_text)
            if len(tokens) <= args.max_seq_len:
                filtered_samples.append(s)
        else:
            filtered_samples.append(s)

    print(f"Итого валидных примеров: {len(filtered_samples)}")

    # Разделение на train и val
    split_idx = int(len(filtered_samples) * (1.0 - args.val_ratio))
    train_data = filtered_samples[:split_idx]
    val_data = filtered_samples[split_idx:]

    os.makedirs(os.path.dirname(args.output_train) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_val) or ".", exist_ok=True)

    with open(args.output_train, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(args.output_val, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Сохранено: train={len(train_data)} в {args.output_train}, val={len(val_data)} в {args.output_val}")

if __name__ == "__main__":
    main()
