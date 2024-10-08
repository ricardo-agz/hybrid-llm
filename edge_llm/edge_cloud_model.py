from enum import Enum

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math
import json
import uuid
import aiohttp
from aiohttp import ClientSession, ClientResponseError

from edge_llm.config import (
    MODEL_SMALL_ID,
    SPECIAL_TOKENS,
    PRESETS,
    SELECTED_PRESET,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    EOS_TOKEN,
    EDGE_CLOUD_URL,
)

# ANSI escape codes for colors
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


class SwitchAction(Enum):
    SWITCH_TO_CLOUD = "switch_to_cloud"
    SWITCH_TO_LOCAL = "switch_to_local"


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

    @staticmethod
    async def _init_cloud_model(
        session: ClientSession, request_id: str, generated_ids: list[int]
    ):
        """
        Initializes the cloud model for a given request ID.

        Args:
            session (ClientSession): The HTTP session for making requests.
            request_id (str): The unique request ID.
            generated_ids (list[int]): The generated token IDs.

        Returns:
            dict: The JSON response from the cloud model.
        """
        cloud_url = f"{EDGE_CLOUD_URL}/init-cloud-model"
        payload = {"request_id": request_id, "generated_ids": generated_ids}
        try:
            async with session.post(cloud_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except ClientResponseError as cre:
            error_data = {"error": f"HTTP error: {cre.status} - {cre.message}"}
            return error_data
        except Exception as e:
            error_data = {"error": str(e)}
            return error_data

    async def _stream_cloud_model(self, session: ClientSession, payload: dict):
        """
        Encapsulates the HTTP streaming logic to interact with the cloud model.

        Args:
            session (ClientSession): The HTTP session for making requests.
            payload (dict): The JSON payload to send in the POST request.

        Yields:
            dict: Parsed JSON data from the cloud model.
        """
        cloud_url = f"{EDGE_CLOUD_URL}/stream-chat-cloud"
        try:
            async with session.post(cloud_url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.content:
                    decoded_line = line.decode("utf-8").strip()
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str.strip() in ["[DONE]", ""]:
                            continue
                        try:
                            data = json.loads(data_str)

                            yield data
                        except json.JSONDecodeError:
                            # Optionally log the error
                            continue
        except ClientResponseError as cre:
            error_data = {"error": f"HTTP error: {cre.status} - {cre.message}"}
            yield error_data
        except Exception as e:
            error_data = {"error": str(e)}
            yield error_data

    async def _generate_with_small_model(self, input_ids, past_key_values, temperature):
        """
        Generates the next token using the small local model.

        Args:
            input_ids (torch.Tensor): The current input IDs.
            past_key_values (tuple): The past key values for the model.
            temperature (float): Sampling temperature.

        Returns:
            dict: Contains next_token_id, decoded_token, entropy, varentropy, and updated past_key_values.
        """
        with torch.no_grad():
            outputs = self.model_small(
                input_ids=input_ids[:, -1:],  # Use only the last token
                past_key_values=past_key_values,
                return_dict=True,
            )

        next_token_logits = outputs.logits[:, -1, :] / temperature
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        probs = torch.exp(log_probs)

        entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(2)
        varentropy = torch.sum(
            probs * ((log_probs / math.log(2) + entropy.unsqueeze(-1)) ** 2), dim=-1
        )

        # Determine sampling strategy based on entropy
        if (
            entropy.item() < self.preset["small_model_entropy_threshold"]
            and varentropy.item() < self.preset["small_model_varentropy_threshold"]
        ):
            # Greedy sampling
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        elif (
            entropy.item() > self.preset["large_model_entropy_threshold"]
            and varentropy.item() > self.preset["large_model_varentropy_threshold"]
        ):
            # Uncertain: signal to switch to cloud model
            return {
                "switch_action": SwitchAction.SWITCH_TO_CLOUD,
                "entropy": entropy.item(),
                "varentropy": varentropy.item(),
            }
        else:
            # Default sampling
            next_token = torch.multinomial(probs, num_samples=1)

        # Update input_ids and tokens_generated
        next_token_id = next_token.item()
        decoded_token = self.tokenizer.decode(next_token_id, skip_special_tokens=True)

        return {
            "next_token_id": next_token_id,
            "decoded_token": decoded_token,
            "entropy": entropy.item(),
            "varentropy": varentropy.item(),
            "past_key_values": outputs.past_key_values,
        }

    async def generate_response(self, prompt: str):
        """
        Generates a response to the given prompt by dynamically switching between
        a local small model and a cloud-based large model based on entropy metrics.

        Args:
            prompt (str): The input prompt for the language model.

        Yields:
            str: Generated tokens and metadata in a streaming fashion.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        generated = input_ids
        temperature = TEMPERATURE

        current_model = "small"  # Start with small model
        past_key_values = None
        tokens_generated = 0
        is_first_pass = True
        tokens_since_last_switch = 0

        # Generate a unique request ID for the cloud model
        request_id = str(uuid.uuid4())

        async with aiohttp.ClientSession() as session:
            while tokens_generated < MAX_NEW_TOKENS:
                if is_first_pass:
                    # For the first pass, generate tokens with both models
                    small_outputs = self.model_small(input_ids=generated)
                    past_key_values = small_outputs.past_key_values

                    await self._init_cloud_model(
                        session, request_id, generated[0].tolist()
                    )
                    is_first_pass = False
                else:
                    if current_model == "small":
                        small_model_output = await self._generate_with_small_model(
                            input_ids=generated,
                            past_key_values=past_key_values,
                            temperature=temperature,
                        )
                        switch_to_cloud = small_model_output.get("switch_to_cloud")
                    elif current_model == "cloud":
                        switch_to_cloud = False

                if current_model == "small":
                    # Generate next token with small model
                    small_model_output = await self._generate_with_small_model(
                        input_ids=generated,
                        past_key_values=past_key_values,
                        temperature=temperature,
                    )

                    if small_model_output.get("switch_to_cloud"):
                        # Prepare payload for cloud model
                        payload = {
                            "request_id": request_id,
                            "temperature": temperature,
                            "generated_ids": generated[0].tolist(),
                            "entropy_threshold": self.preset[
                                "small_model_entropy_threshold"
                            ],
                            "varentropy_threshold": self.preset[
                                "small_model_varentropy_threshold"
                            ],
                            "max_tokens": MAX_NEW_TOKENS,
                            "tokens_generated": tokens_generated,
                        }

                        # Switch to cloud model
                        current_model = "cloud"
                        async for data in self._stream_cloud_model(session, payload):
                            if "switch" in data and data["switch"] == "back_to_local":
                                current_model = "small"
                                break
                            elif "next_token" in data:
                                # Process the received token from cloud
                                next_token = data["next_token"]
                                entropy = data.get("entropy")
                                varentropy = data.get("varentropy")

                                # Decode the token
                                next_token_ids = self.tokenizer.encode(
                                    next_token, add_special_tokens=False
                                )
                                if not next_token_ids:
                                    continue  # Skip if token encoding fails
                                next_token_id = next_token_ids[0]
                                generated = torch.cat(
                                    (
                                        generated,
                                        torch.tensor(
                                            [[next_token_id]], device=self.device
                                        ),
                                    ),
                                    dim=-1,
                                )
                                tokens_generated += 1

                                # Yield the token to the caller
                                chunk = {
                                    "choices": [
                                        {
                                            "delta": {"content": next_token},
                                            "index": 0,
                                            "finish_reason": None,
                                            "metadata": {"model_used": "large"},
                                        }
                                    ],
                                    "model": "cloud_model",
                                    "usage": {
                                        "prompt_tokens": input_ids.numel(),
                                        "completion_tokens": generated.numel()
                                        - input_ids.numel(),
                                    },
                                }

                                yield f"data: {json.dumps(chunk)}\n\n"

                                # Print the token (optional)
                                print(f"{RED}{next_token}{RESET}", end="", flush=True)

                                # Check for EOS token
                                if next_token_id == self.eos_token_id:
                                    yield 'data: {"finish": "EOS token generated"}\n\n'
                                    return
                            elif "error" in data:
                                # Handle errors from the cloud API
                                error_chunk = {"error": data["error"]}
                                yield f"data: {json.dumps(error_chunk)}\n\n"
                                return

                    else:
                        # Continue with small model
                        next_token_id = small_model_output["next_token_id"]
                        decoded_token = small_model_output["decoded_token"]
                        entropy = small_model_output["entropy"]
                        varentropy = small_model_output["varentropy"]
                        past_key_values = small_model_output["past_key_values"]

                        # Append the next token to the generated sequence
                        generated = torch.cat(
                            (
                                generated,
                                torch.tensor([[next_token_id]], device=self.device),
                            ),
                            dim=-1,
                        )
                        tokens_generated += 1

                        # Yield the token to the caller
                        chunk = {
                            "choices": [
                                {
                                    "delta": {"content": decoded_token},
                                    "index": 0,
                                    "finish_reason": None,
                                    "metadata": {"model_used": "small"},
                                }
                            ],
                            "model": self.model_small.config._name_or_path,
                            "usage": {
                                "prompt_tokens": input_ids.numel(),
                                "completion_tokens": generated.numel()
                                - input_ids.numel(),
                            },
                        }

                        yield f"data: {json.dumps(chunk)}\n\n"

                        # Print the token (optional)
                        print(f"{BLUE}{decoded_token}{RESET}", end="", flush=True)

                        # Check for EOS token
                        if next_token_id == self.eos_token_id:
                            yield 'data: {"finish": "EOS token generated"}\n\n'
                            return

                elif current_model == "cloud":
                    # No action needed here since streaming is handled in the "small" block
                    # Continue the loop to process any potential further instructions
                    # Remove the asyncio.sleep to prevent unnecessary delays
                    pass
                else:
                    # Undefined model state
                    error_chunk = {"error": "Undefined model state."}
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    break

            # Signal completion if max tokens reached
            if tokens_generated >= MAX_NEW_TOKENS:
                yield 'data: {"finish": "MAX_TOKENS"}\n\n'
