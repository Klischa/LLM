#!/usr/bin/env python3
"""
Скрипт слияния обученных LoRA-адаптеров с базовой моделью Qwen2.5.
Экспортирует объединенную FP16 модель в формате HuggingFace Safetensors,
готовую для конвертации в GGUF через llama.cpp.
"""

import os
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Слияние LoRA с базовой моделью")
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Имя или путь базовой модели")
    parser.add_argument("--lora_path", type=str, default="outputs/qwen2.5-1.5b-notes-lora", help="Путь к LoRA чекпоинту")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2.5-1.5b-notes-merged", help="Каталог для объединенной модели")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Устройство для слияния")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Загрузка базовой модели {args.base_model_id} в {args.device}...")

    torch_dtype = torch.float16 if args.device == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    logger.info(f"Загрузка и привязка LoRA-весов из {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    logger.info("Слияние весов (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    logger.info(f"Загрузка токенизатора из {args.lora_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)

    logger.info(f"Сохранение объединенной модели в {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Модель успешно объединена и готова к конвертации в GGUF!")

if __name__ == "__main__":
    main()
