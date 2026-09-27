#!/usr/bin/env python3
import sys
import os

def patch_llama_cpp(llama_dir):
    print(f"[patch_llama_cpp] Checking {llama_dir}...")
    llama_file = os.path.join(llama_dir, "src", "llama.cpp")
    vocab_file = os.path.join(llama_dir, "src", "llama-vocab.cpp")

    target_files = []
    if os.path.exists(llama_file):
        target_files.append(llama_file)
    if os.path.exists(vocab_file):
        target_files.append(vocab_file)

    if not target_files:
        print(f"[patch_llama_cpp] Error: Neither {llama_file} nor {vocab_file} found!")
        sys.exit(1)

    patched_any = False
    for fpath in target_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        changed = False

        # 1. Добавляем deepseek-r1-qwen, deepseek-v3, deepseek-r1 в проверку qwen2
        qwen_target = 'tokenizer_pre == "qwen2") {'
        qwen_repl = 'tokenizer_pre == "qwen2" || tokenizer_pre == "deepseek-r1-qwen" || tokenizer_pre == "deepseek-v3" || tokenizer_pre == "deepseek-r1" || tokenizer_pre == "gpt-4o") {'
        if qwen_target in content:
            content = content.replace(qwen_target, qwen_repl, 1)
            changed = True
            print(f"[patch_llama_cpp] Added deepseek-r1-qwen to {os.path.basename(fpath)}")

        # 2. Заменяем throw std::runtime_error("unknown pre-tokenizer type: ...") на fallback на QWEN2
        # чтобы ни одна модель больше не падала с fatal error при загрузке
        if 'throw std::runtime_error(format("unknown pre-tokenizer type:' in content:
            old_throw = 'throw std::runtime_error(format("unknown pre-tokenizer type: \'%s\'", tokenizer_pre.c_str()));'
            new_fallback = 'fprintf(stderr, "%s: unknown pre-tokenizer type: \'%s\', falling back to QWEN2\\n", __func__, tokenizer_pre.c_str()); vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_QWEN2; vocab.tokenizer_clean_spaces = false;'
            if old_throw in content:
                content = content.replace(old_throw, new_fallback, 1)
                changed = True
                print(f"[patch_llama_cpp] Replaced throw with QWEN2 fallback in {os.path.basename(fpath)}")

        if changed:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            patched_any = True

    if patched_any:
        print("[patch_llama_cpp] Patch applied successfully!")
    else:
        print("[patch_llama_cpp] Warning: No patterns matched or already patched.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 patch_llama_cpp.py <path_to_llama_cpp_dir>")
        sys.exit(1)
    patch_llama_cpp(sys.argv[1])
