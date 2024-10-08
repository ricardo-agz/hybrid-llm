from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math
import json

from edge_llm.config import (
    MODEL_SMALL_ID,
    MODEL_LARGE_ID,
    SPECIAL_TOKENS,
    PRESETS,
    SELECTED_PRESET,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    EOS_TOKEN,
)


# ANSI escape codes for colors
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


class ChatModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SMALL_ID)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        self.model_small = AutoModelForCausalLM.from_pretrained(
            MODEL_SMALL_ID, torch_dtype=torch.bfloat16
        )
        self.model_large = AutoModelForCausalLM.from_pretrained(
            MODEL_LARGE_ID, torch_dtype=torch.bfloat16
        )

        self.model_small.resize_token_embeddings(len(self.tokenizer))
        self.model_large.resize_token_embeddings(len(self.tokenizer))

        self.model_small.eval()
        self.model_large.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_small.to(self.device)
        self.model_large.to(self.device)

        self.preset = PRESETS[SELECTED_PRESET]
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    async def generate_response(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        generated = input_ids
        temperature = TEMPERATURE

        current_model = self.model_small
        small_model_past_key_values = None
        large_model_past_key_values = None
        is_first_pass = True

        tokens_since_last_switch = 0

        for _ in range(MAX_NEW_TOKENS):
            if is_first_pass:
                # For the first pass, generate tokens with both models
                small_outputs = self.model_small(input_ids=generated)
                small_model_past_key_values = small_outputs.past_key_values
                large_outputs = self.model_large(input_ids=generated)
                large_model_past_key_values = large_outputs.past_key_values
                is_first_pass = False
                outputs = large_outputs  # Start with large model
            else:
                # For subsequent passes, use the current model
                current_past_key_values = (
                    small_model_past_key_values
                    if current_model == self.model_small
                    else large_model_past_key_values
                )
                outputs = current_model(
                    input_ids=generated[:, -1:], past_key_values=current_past_key_values
                )
                if current_model == self.model_small:
                    small_model_past_key_values = outputs.past_key_values
                else:
                    large_model_past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]

            # Scale the logits by the temperature parameter
            next_token_logits = next_token_logits / temperature

            # Apply softmax to convert to probabilities
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            probs = torch.exp(log_probs)

            # Compute entropy and varentropy
            entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(
                2
            )  # Convert to bits
            varentropy = torch.sum(
                probs * ((log_probs / math.log(2) + entropy.unsqueeze(-1)) ** 2), dim=-1
            )

            # Adjust sampling strategy and model based on entropy and varentropy
            if (
                entropy.item() < self.preset["small_model_entropy_threshold"]
                and varentropy.item() < self.preset["small_model_varentropy_threshold"]
            ):
                # Low entropy and low varentropy: model is confident
                if current_model != self.model_small:
                    current_model = self.model_small
                    # Recompute past_key_values for the small model
                    generated_tokens_to_reprocess = generated[
                        :, -tokens_since_last_switch:
                    ]
                    outputs = current_model(
                        input_ids=generated_tokens_to_reprocess,
                        past_key_values=small_model_past_key_values,
                    )
                    small_model_past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    tokens_since_last_switch = 0
                # Greedy sampling
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            elif (
                entropy.item() > self.preset["large_model_entropy_threshold"]
                and varentropy.item() > self.preset["large_model_varentropy_threshold"]
            ):
                # High entropy and high varentropy: model is uncertain
                if current_model != self.model_large:
                    current_model = self.model_large
                    # Recompute past_key_values for the large model
                    generated_tokens_to_reprocess = generated[
                        :, -tokens_since_last_switch:
                    ]
                    outputs = current_model(
                        input_ids=generated_tokens_to_reprocess,
                        past_key_values=large_model_past_key_values,
                    )
                    large_model_past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    tokens_since_last_switch = 0
                # Increase temperature for more exploration
                adjusted_temperature = min(1.5, temperature * 1.3)
                next_token_logits = next_token_logits / adjusted_temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Default sampling
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append the next token to the generated sequence
            generated = torch.cat((generated, next_token), dim=-1)
            tokens_since_last_switch += 1

            # Decode the newly generated token
            decoded_token = self.tokenizer.decode(
                next_token[0], skip_special_tokens=False
            )

            # Optionally handle special tokens (depends on your use case)
            # If you decide not to skip special tokens, remove the following lines
            # special_tokens_list = self.tokenizer.all_special_tokens
            # if decoded_token in special_tokens_list:
            #     continue

            color = BLUE if current_model == self.model_small else RED
            print(f"{color}{decoded_token}{RESET}", end="", flush=True)

            # Stop if the EOS token is generated
            if next_token.item() == self.eos_token_id:
                break

            # Prepare the chunk to yield
            chunk = {
                "choices": [
                    {
                        "delta": {"content": decoded_token},
                        "index": 0,
                        "finish_reason": None,
                        "metadata": {
                            "model_used": "small"
                            if current_model == self.model_small
                            else "large"
                        },
                    }
                ],
                "model": current_model.config._name_or_path,
                "usage": {
                    "prompt_tokens": input_ids.numel(),
                    "completion_tokens": generated.numel() - input_ids.numel(),
                },
            }

            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"
