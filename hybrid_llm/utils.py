import math
import torch
import torch.nn.functional as F


def build_prompt(messages: list[dict]):
    prompt = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            prompt += (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{content}<|eot_id|>"
            )
        elif role == "user":
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
        elif role == "assistant":
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{content}"
        else:
            raise ValueError(f"Unknown role: {role}")

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt


def calculate_entropy(logits: torch.Tensor) -> (float, float):
    """
    Calculates entropy and varentropy from logits.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(2)  # bits
    varentropy = torch.sum(
        probs * ((log_probs / math.log(2) + entropy.unsqueeze(-1)) ** 2), dim=-1
    )
    return entropy.item(), varentropy.item()
