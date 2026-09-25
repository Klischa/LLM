#!/usr/bin/env python3
"""
Скрипт обучения Qwen2.5-1.5B (или 0.5B) методом QLoRA на русскоязычном датасете.
Использует PyTorch + HuggingFace Transformers + PEFT + BitsAndBytes.
"""

import os
import sys
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA файнтюнинг Qwen2.5 для мобильного ассистента")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Базовая модель HuggingFace")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl", help="Путь к обучающему файлу")
    parser.add_argument("--val_file", type=str, default="data/val.jsonl", help="Путь к валидационному файлу")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2.5-1.5b-notes-lora", help="Каталог для чекпоинтов")
    parser.add_argument("--max_seq_length", type=int, default=1536, help="Максимальная длина последовательности")
    parser.add_argument("--lora_r", type=int, default=16, help="Ранг LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA Dropout")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10, help="Шаг логирования")
    parser.add_argument("--save_steps", type=int, default=100, help="Шаг сохранения")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")
    return parser.parse_args()

def print_trainable_parameters(model):
    """Подсчет обучаемых параметров."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100 * trainable_params / all_param
    logger.info(
        f"Обучаемых параметров: {trainable_params:,} из {all_param:,} ({pct:.2f}%)"
    )

def prepare_chatml_features(examples, tokenizer, max_seq_length):
    """
    Токенизация диалогов ChatML с маскированием (loss masking).
    Loss вычисляется ИСКЛЮЧИТЕЛЬНО на токенах ответа ассистента,
    а токены системного промпта и запроса пользователя маскируются значением -100.
    """
    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    # Специальные токены Qwen2.5 ChatML
    # <|im_start|>assistant\n ... <|im_end|>\n
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    for messages in examples["messages"]:
        # Применяем нативный chat template без токенизации для получения разметки
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Токенизируем весь диалог
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = [-100] * len(input_ids)

        # Вычисляем маску для ответов ассистента:
        # Ищем сегменты между "<|im_start|>assistant\n" и "<|im_end|>"
        prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        suffix_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

        # Поиск позиций ответа ассистента
        i = 0
        while i < len(input_ids):
            # Проверяем совпадение префикса ассистента
            if input_ids[i:i+len(prefix_ids)] == prefix_ids:
                start_idx = i + len(prefix_ids)
                # Ищем конец ответа
                end_idx = start_idx
                while end_idx < len(input_ids) and input_ids[end_idx] != suffix_id:
                    end_idx += 1
                
                # Включаем токены самого ответа и закрывающий токен в подсчет loss
                if end_idx < len(input_ids):
                    end_idx += 1  # захватываем suffix_id
                
                for k in range(start_idx, end_idx):
                    labels[k] = input_ids[k]
                i = end_idx
            else:
                i += 1

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_mask.append(attention_mask)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_mask
    }

def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info(f"Загрузка токенизатора для {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Конфигурация 4-битного квантования (QLoRA)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",               # Нормализованный 4-битный формат (NF4)
        bnb_4bit_use_double_quant=True,         # Вторичное квантование для экономии ~0.4 бит/вес
        bnb_4bit_compute_dtype=compute_dtype     # Вычисления в BF16/FP16
    )

    # 2. Загрузка базовой модели
    logger.info(f"Загрузка модели {args.model_id} в 4-bit...")
    device_map = "auto" if torch.cuda.is_available() else None
    
    # Защита от запуска на чистом CPU без CUDA для демонстрации
    if not torch.cuda.is_available():
        logger.warning("CUDA не обнаружена. Загрузка в обычном FP32 (только для тестов, без bnb).")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=compute_dtype
        )
        # Подготовка модели к k-bit обучению (замораживание весов, cast layer norm в fp32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 3. Конфигурация PEFT / LoRA
    # Таргетируем все линейные проекции внимания и MLP для максимального качества
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    # 4. Загрузка и токенизация датасета
    logger.info("Загрузка данных...")
    data_files = {"train": args.train_file}
    if os.path.exists(args.val_file):
        data_files["val"] = args.val_file
    
    raw_datasets = load_dataset("json", data_files=data_files)
    
    tokenized_train = raw_datasets["train"].map(
        lambda x: prepare_chatml_features(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Токенизация обучающей выборки"
    )

    tokenized_val = None
    if "val" in raw_datasets:
        tokenized_val = raw_datasets["val"].map(
            lambda x: prepare_chatml_features(x, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=raw_datasets["val"].column_names,
            desc="Токенизация валидационной выборки"
        )

    # 5. Data Collator с паддингом
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    # 6. Аргументы обучения
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=(compute_dtype == torch.float16 and torch.cuda.is_available()),
        bf16=(compute_dtype == torch.bfloat16 and torch.cuda.is_available()),
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if tokenized_val else "no",
        eval_steps=args.save_steps if tokenized_val else None,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none"
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator
    )

    logger.info("Запуск процесса обучения...")
    trainer.train()

    logger.info(f"Сохранение LoRA-адаптеров в {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Файнтюнинг успешно завершен!")

if __name__ == "__main__":
    main()
