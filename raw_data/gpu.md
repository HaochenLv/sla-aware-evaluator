GPU_CATALOG: Dict[str, Dict[str, Any]] = {
    "V100": {
        "name": "V100",
        "G_TFLOPS": 14,  # FP32 CUDA Core(V100较老,保持FP32)
        "VRAM_bytes": int(16 * 1024**3),
        "cost_per_GB_month": 14.31
    },
    "5090_32GB": {
        "name": "5090_32GB",
        "G_TFLOPS": 109.7,  # Turing架构 FP16推理性能(8x speedup)
        "VRAM_bytes": int(32 * 1024**3),
        "cost_per_GB_month": 14.96
    },
    "A6000_48GB": {
        "name": "A6000_48GB",
        "G_TFLOPS": 38.71,  # Ada架构 FP16 Tensor Core推理性能
        "VRAM_bytes": int(48 * 1024**3),
        "cost_per_GB_month": 8.52
    },
    "A40": {
        "name": "A40",
        "G_TFLOPS": 250.0,  # Ampere架构 FP16 Tensor Core推理性能
        "VRAM_bytes": int(48 * 1024**3),
        "cost_per_GB_month": 6.63
    },
    "A100_80GB": {
        "name": "A100_80GB",
        "G_TFLOPS": 19.5,  # Ampere架构 FP16 Tensor Core推理性能
        "VRAM_bytes": int(80 * 1024**3),
        "cost_per_GB_month": 10.62
    },
    "H100_256GB": {
        "name": "H100_256GB",
        "G_TFLOPS": 183.0,  # Hopper架构 FP16 Tensor Core推理性能(PCIe版本)
        "VRAM_bytes": int(256* 1024**3),
        "cost_per_GB_month": 8.20
    },
    "Pro6000_96GB": {
        "name": "Pro6000_96GB",
        "G_TFLOPS": 126.0,  # Hopper架构 FP16 Tensor Core推理性能
        "VRAM_bytes": int(96 * 1024**3),
        "cost_per_GB_month": 4.99
    }
}



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