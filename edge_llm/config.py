# Model Configuration
MODEL_SMALL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_LARGE_ID = "meta-llama/Llama-3.2-3B-Instruct"

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]
}

# Presets for model switching
PRESETS = {
    "conservative": {
        "small_model_entropy_threshold": 1.0,
        "small_model_varentropy_threshold": 0.2,
        "large_model_entropy_threshold": 2.5,
        "large_model_varentropy_threshold": 0.8,
    },
    "balanced": {
        "small_model_entropy_threshold": 0.5,
        "small_model_varentropy_threshold": 0.1,
        "large_model_entropy_threshold": 3.0,
        "large_model_varentropy_threshold": 1.0,
    },
    "aggressive": {
        "small_model_entropy_threshold": 0.3,
        "small_model_varentropy_threshold": 0.05,
        "large_model_entropy_threshold": 4.0,
        "large_model_varentropy_threshold": 1.5,
    },
}

SELECTED_PRESET = "conservative"

# Generation Parameters
MAX_NEW_TOKENS = 248
TEMPERATURE = 0.7

# EOS Token
EOS_TOKEN = "<|eot_id|>"

# Edge Cloud Config
EDGE_CLOUD_URL = "http://localhost:8081"
