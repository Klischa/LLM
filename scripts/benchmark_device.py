#!/usr/bin/env python3
"""
Калькулятор и бенчмарк производительности локальных LLM на процессоре Helio G99.
Моделирует пропускную способность памяти, энергопотребление и тепловой троттлинг.
"""

def calculate_hardware_limits(
    params_b: float,
    bits_per_weight: float = 4.5, # Q4_K_M средний битрейт
    context_length: int = 2048,
    num_heads_kv: int = 2,
    head_dim: int = 128,
    num_layers: int = 28,
    active_threads: int = 2
):
    print(f"=== Расчет характеристик для модели {params_b}B на Helio G99 ===")
    
    # 1. Размер весов модели в ГБ
    weights_bytes = (params_b * 1e9 * bits_per_weight) / 8.0
    weights_gb = weights_bytes / (1024 ** 3)
    
    # 2. Размер KV-кэша (FP16: 2 байта на элемент)
    # KV cache = 2 * n_layers * n_heads_kv * head_dim * context_len * 2 bytes
    kv_cache_bytes = 2 * num_layers * num_heads_kv * head_dim * context_length * 2
    kv_cache_mb = kv_cache_bytes / (1024 ** 2)
    
    total_ram_required_gb = (weights_bytes + kv_cache_bytes) / (1024 ** 3) + 0.35 # +350MB runtime overhead
    
    # 3. Пропускная способность LPDDR4X на Helio G99
    # 2x 16-bit шина @ 2133 MHz (LPDDR4X-4266) -> теоретический пик ~17.1 ГБ/с
    # Реальная эффективная ПСП для CPU read: ~10.5 - 11.5 ГБ/с
    effective_bandwidth_gbps = 11.0
    
    # Во время стадии генерации (token-by-token) каждый токен считывает почти все веса модели один раз!
    # Максимальная теоретическая скорость генерации (tok/s) = Bandwidth / Weights_size
    theoretical_max_tok_sec = effective_bandwidth_gbps / weights_gb
    
    # CPU Compute bottleneck: Cortex-A76 @ 2.2 GHz
    # 2 ядра A76 с NEON dot-product (SDOT/UDOT):
    # Каждое ядро выполняет 32 int8 ops за такт -> 2.2 GHz * 32 = 70.4 GOPS/core -> ~140 GOPS суммарно.
    # 1.5B модель требует ~3 GFLOPs (GOPS) на токен. 140 / 3 = ~46 tok/s вычислительный пик.
    # Следовательно, инференс жестко упирается именно в Bandwidth памяти (Memory-bound)!
    compute_tok_sec = (active_threads * 2.2 * 32) / (params_b * 2.0)
    
    realistic_tok_sec = min(theoretical_max_tok_sec * 0.85, compute_tok_sec)
    
    # Тепловой баланс Infinix Note 30
    # Пассивное рассеивание пластикового корпуса: ~2.8 - 3.2 Вт при 25°C комнатной.
    # Энергопотребление 2 ядер A76 на 100%: ~2.4 Вт
    # Энергопотребление 2x A76 + 6x A55 на 100%: ~4.8 Вт -> ГАРАНТИРОВАННЫЙ ТРОТТЛИНГ через 3 минуты!
    throttling_risk = "ВЫСОКИЙ (тэмпература > 50°C через 3 мин)" if active_threads > 3 else "НИЗКИЙ / УМЕРЕННЫЙ (< 42°C)"
    
    print(f"Размер весов (Q4_K_M):      {weights_gb:.2f} ГБ")
    print(f"Размер KV-кэша ({context_length} tok): {kv_cache_mb:.1f} МБ")
    print(f"Общий расход ОЗУ в рантайме: {total_ram_required_gb:.2f} ГБ (из доступных 8 ГБ)")
    print(f"Эффективная ПСП памяти:      {effective_bandwidth_gbps:.1f} ГБ/с (LPDDR4X)")
    print(f"Ограничение скорости:        Memory-bound")
    print(f"Ожидаемая скорость:          ~{realistic_tok_sec:.1f} токенов/сек")
    print(f"Активные потоки:             {active_threads} потока")
    print(f"Риск теплового троттлинга:   {throttling_risk}")
    print("=" * 60)

if __name__ == "__main__":
    calculate_hardware_limits(1.54, active_threads=2)
    calculate_hardware_limits(0.49, active_threads=2)
    calculate_hardware_limits(1.54, active_threads=6)
