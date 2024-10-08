# edge_cloud_chat_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math
import json
import requests
import uuid

from edge_llm.config import (
    MODEL_SMALL_ID,
    SPECIAL_TOKENS,
    PRESETS,
    SELECTED_PRESET,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    EOS_TOKEN,
    EDGE_CLOUD_URL,  # Ensure this is defined in your config
)

# ANSI escape codes for colors
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def query_cloud_model(
    request_id: str,
    temperature: float,
    generated_ids: list[int],
    num_tokens_to_reprocess: int | None = None,
) -> tuple[int, torch.Tensor]:
    payload = {
        "request_id": request_id,
        "temperature": temperature,
        "generated_ids": generated_ids,
        "num_tokens_to_reprocess": num_tokens_to_reprocess,
    }
    response = requests.post(f"{EDGE_CLOUD_URL}/chat-cloud", json=payload)
    if response.status_code != 200:
        raise Exception("Error in cloud model API call")

    res_json = response.json()
    next_token = res_json["next_token"]
    next_token_logits_list = res_json["next_token_logits"]
    next_token_logits = torch.tensor(
        [next_token_logits_list],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return next_token, next_token_logits


class EdgeCloudChatModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SMALL_ID)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        self.model_small = AutoModelForCausalLM.from_pretrained(
            MODEL_SMALL_ID, torch_dtype=torch.bfloat16
        )
        self.model_small.resize_token_embeddings(len(self.tokenizer))
        self.model_small.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_small.to(self.device)

        self.preset = PRESETS[SELECTED_PRESET]
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    async def generate_response(self, prompt: str):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        generated = input_ids
        temperature = TEMPERATURE

        current_model = self.model_small  # Start with small model
        small_model_past_key_values = None
        is_first_pass = True
        tokens_since_last_switch = 0

        # Generate a unique request ID for the cloud model
        request_id = str(uuid.uuid4())

        for _ in range(MAX_NEW_TOKENS):
            if is_first_pass:
                # For the first pass, generate outputs with both models
                # Small model outputs
                small_outputs = self.model_small(input_ids=generated)
                small_model_past_key_values = small_outputs.past_key_values
                small_next_token_logits = small_outputs.logits[:, -1, :]

                # Cloud model outputs via API call
                generated_ids = generated[0].tolist()
                _, cloud_next_token_logits = query_cloud_model(
                    request_id=request_id,
                    temperature=temperature,
                    generated_ids=generated_ids,
                )

                # Decide which model to start with based on entropy
                # Compute entropy for both models
                small_log_probs = F.log_softmax(small_next_token_logits, dim=-1)
                small_probs = torch.exp(small_log_probs)
                small_entropy = -torch.sum(
                    small_probs * small_log_probs, dim=-1
                ) / math.log(2)

                cloud_log_probs = F.log_softmax(cloud_next_token_logits, dim=-1)
                cloud_probs = torch.exp(cloud_log_probs)
                cloud_entropy = -torch.sum(
                    cloud_probs * cloud_log_probs, dim=-1
                ) / math.log(2)

                # Choose the model with lower entropy
                if small_entropy.item() <= cloud_entropy.item():
                    current_model = self.model_small
                    next_token_logits = small_next_token_logits
                else:
                    current_model = "cloud"
                    next_token_logits = cloud_next_token_logits

                is_first_pass = False
            else:
                if current_model == self.model_small:
                    # Use the small model
                    outputs = self.model_small(
                        input_ids=generated[:, -1:],
                        past_key_values=small_model_past_key_values,
                    )
                    small_model_past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                else:
                    # Use the cloud model
                    generated_ids = generated[0].tolist()
                    num_tokens_to_reprocess = tokens_since_last_switch or 1
                    _, next_token_logits = query_cloud_model(
                        request_id=request_id,
                        temperature=temperature,
                        generated_ids=generated_ids,
                        num_tokens_to_reprocess=num_tokens_to_reprocess,
                    )

            # Scale logits by temperature
            next_token_logits = next_token_logits / temperature

            # Compute log probabilities
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            probs = torch.exp(log_probs)

            # Compute entropy and varentropy
            entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(2)
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
                    tokens_since_last_switch = 0
                    # Recompute past_key_values for the small model
                    generated_tokens_to_reprocess = generated[
                        :, -tokens_since_last_switch or 1 :
                    ]
                    outputs = self.model_small(input_ids=generated_tokens_to_reprocess)
                    small_model_past_key_values = outputs.past_key_values
                    continue  # Use small model in next iteration
                # Greedy sampling
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            elif (
                entropy.item() > self.preset["large_model_entropy_threshold"]
                and varentropy.item() > self.preset["large_model_varentropy_threshold"]
            ):
                # High entropy and high varentropy: model is uncertain
                if current_model != "cloud":
                    current_model = "cloud"
                    tokens_since_last_switch = 0
                    continue  # Will use cloud model in the next iteration
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
                "model": current_model.config._name_or_path
                if current_model != "cloud"
                else "cloud_model",
                "usage": {
                    "prompt_tokens": input_ids.numel(),
                    "completion_tokens": generated.numel() - input_ids.numel(),
                },
            }

            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"
