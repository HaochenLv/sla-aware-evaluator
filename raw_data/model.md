MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "Llama-3.1-13B-Instruct": {
        "name": "Llama-3.1-13B-Instruct",
        "num_layers": 56,
        "total_params": float(13e9),
        "bytes_per_param": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "flops_per_token_per_layer": float(24 * (4096 ** 2)),  # ~402M FLOPs/层/token
        "activation_mb_per_boundary": 32.0,
    },
    "Llama-3.1-70B-Instruct": {
        "name": "Llama-3.1-70B-Instruct",
        "num_layers": 80,
        "total_params": float(70e9),
        "bytes_per_param": 2, #影响 KV 项
        "hidden_size": 8192, #影响 KV 项
        "num_attention_heads": 64, #影响 KV 项
        "num_key_value_heads": 8, #影响 KV 项
        "flops_per_token_per_layer": float(24 * (8192 ** 2)),  # ~1.61G FLOPs/层/token
        "activation_mb_per_boundary": 64.0,
    },
    "Qwen2.5-7B": {
        "name": "Qwen2.5-7B",
        "num_layers": 28,
        "total_params": float(7e9),
        "bytes_per_param": 2,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "flops_per_token_per_layer": float(24 * (3584 ** 2)),  # ~402M FLOPs/层/token
        "activation_mb_per_boundary": 28,
    },
    "Mistral-7B": {
      "name": "Mistral-7B",
      "num_layers": 32,
      "total_params": float(7e9),
      "bytes_per_param": 2,
      "hidden_size": 4096,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "flops_per_token_per_layer": float(24 * (4096 ** 2)),  # ~402M FLOPs/层/token
      "activation_mb_per_boundary": 32.0,
    },
    "Gemma-2-27B": {
        "name": "Gemma-2-27B",
        "num_layers": 46,
        "total_params": float(27e9),
        "bytes_per_param": 2,
        "hidden_size": 4608,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "flops_per_token_per_layer": float(24 * (4608 ** 2)),   # ~1.01G FLOPs/层/token
        "activation_mb_per_boundary": 36.0,
    },
    "DeepSeek-V2": {
      "name": "DeepSeek-V2",
      "num_layers": 60,
      "total_params": float(236e9),
      "bytes_per_param": 2,
      "hidden_size": 5120,
      "num_attention_heads": 128,
      "num_key_value_heads": 128,
      "flops_per_token_per_layer": float(24 * (5120 ** 2)),  # ~1.58G FLOPs/层/token
      "activation_mb_per_boundary": 40.0,
  }
}