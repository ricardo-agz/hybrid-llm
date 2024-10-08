from enum import Enum
from transformers import (
    AutoTokenizer,
)
import torch
import json

from config import (
    MODEL_SMALL_ID,
    MODEL_LARGE_ID,
    SPECIAL_TOKENS,
    PRESETS,
    SELECTED_PRESET,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    EOS_TOKEN,
)
from model_engines.api_model_engine import APIModelEngine
from model_engines.base import ModelType
from model_engines.local_model_engine import LocalModelEngine

# ANSI escape codes for colors
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


class DynamicModelVariant(Enum):
    LOCAL = "local"
    EDGE_CLOUD = "edge_cloud"


class DynamicModel:
    def __init__(self, variant: DynamicModelVariant = DynamicModelVariant.LOCAL):
        self.variant = variant
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SMALL_ID)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.preset = PRESETS[SELECTED_PRESET]
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    @staticmethod
    def _prepare_chunk(
        decoded_token,
        model,
        model_type: ModelType = None,
        finish_reason: str = None,
        completion_metadata: dict = None,
    ):
        chunk = {
            "choices": [
                {
                    "delta": {"content": decoded_token},
                    "index": 0,
                    "finish_reason": finish_reason,
                    "metadata": {
                        "model_used": model_type.value if model_type else None
                    },
                }
            ],
            "model": model,
        }

        if completion_metadata:
            chunk["usage"] = completion_metadata

        return chunk

    async def generate_response(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        generated = input_ids.clone()
        temperature = TEMPERATURE

        small_engine = LocalModelEngine(
            tokenizer=self.tokenizer,
            model_id=MODEL_SMALL_ID,
            model_type=ModelType.SMALL,
            device=self.device,
            entropy_thresholds=self.preset,
            max_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
        )
        large_engine = (
            LocalModelEngine(
                tokenizer=self.tokenizer,
                model_id=MODEL_LARGE_ID,
                model_type=ModelType.LARGE,
                device=self.device,
                entropy_thresholds=self.preset,
                max_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
            )
            if self.variant == DynamicModelVariant.LOCAL
            else APIModelEngine(
                tokenizer=self.tokenizer,
                model_id=MODEL_LARGE_ID,
                device=self.device,
                entropy_thresholds=self.preset,
                max_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
            )
        )

        large_model_initialized = False

        # init both models
        _, _ = await small_engine.init_model(input_ids)
        current_engine = small_engine

        tokens_since_last_switch = 0
        total_tokens_generated = 0

        prompt_tokens = input_ids.numel()
        usage_metadata = {
            "small_model": {"prompt_tokens": prompt_tokens, "completion_tokens": 0},
            "large_model": {"prompt_tokens": 0, "completion_tokens": 0},
        }

        while total_tokens_generated < MAX_NEW_TOKENS:
            switch_occurred = False

            if current_engine.model_type == ModelType.LARGE and not large_model_initialized:
                _, _ = await large_engine.init_model(input_ids)
                large_model_initialized = True
                usage_metadata["large_model"]["prompt_tokens"] = prompt_tokens
                print(f"{RED}*Initialized large model*{RESET}", end="", flush=True)

            model_used_key = "small_model" if current_engine.model_type == ModelType.SMALL else "large_model"
            other_model_key = "large_model" if model_used_key == "small_model" else "small_model"

            # Add the tokens_since_last_switch to the other model's prompt_tokens
            # usage_metadata[other_model_key]["prompt_tokens"] += tokens_since_last_switch

            async for action in current_engine.stream_generate(
                generated_ids=generated,
                tokens_generated=total_tokens_generated,
                tokens_to_reprocess=tokens_since_last_switch,
            ):
                if "action" in action:
                    if action["action"] == "switch_to_small":
                        # Switch to small model
                        current_engine = small_engine
                        switch_occurred = True
                        break  # Break to restart the loop with the new model

                    elif action["action"] == "switch_to_large":
                        # Switch to large model
                        current_engine = large_engine
                        switch_occurred = True
                        break  # Break to restart the loop with the new model

                    elif action["action"] == "done":
                        chunk = self._prepare_chunk(
                            decoded_token="",
                            model=current_engine.model_id,
                            finish_reason="STOP",
                            model_type=current_engine.model_type,
                            completion_metadata={
                                "prompt_tokens": input_ids.numel(),
                                "completion_tokens": generated.numel()
                                - input_ids.numel(),
                                "by_model": usage_metadata,
                            },
                        )

                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                elif "token" in action:
                    decoded_token = action["token"]
                    token_id = action["token_id"]
                    tokens_since_last_switch += 1
                    total_tokens_generated += 1

                    color = (
                        BLUE if current_engine.model_type == ModelType.SMALL else RED
                    )
                    print(f"{color}{decoded_token}{RESET}", end="", flush=True)

                    model_used_key = "small_model" if current_engine.model_type == ModelType.SMALL else "large_model"
                    usage_metadata[model_used_key]["completion_tokens"] += 1

                    # Append the new token to the generated sequence
                    generated = torch.cat(
                        [generated, torch.tensor([[token_id]], device=self.device)],
                        dim=-1,
                    )

                    chunk = self._prepare_chunk(
                        decoded_token=decoded_token,
                        model=current_engine.model_id,
                        model_type=current_engine.model_type,
                    )

                    yield f"data: {json.dumps(chunk)}\n\n"

                # Terminate if maximum tokens are generated
                if total_tokens_generated >= MAX_NEW_TOKENS:
                    chunk = self._prepare_chunk(
                        decoded_token="",
                        model=current_engine.model_id,
                        finish_reason="MAX_TOKENS",
                        model_type=current_engine.model_type,
                        completion_metadata={
                            "prompt_tokens": input_ids.numel(),
                            "completion_tokens": generated.numel() - input_ids.numel(),
                            "by_model": usage_metadata,
                        },
                    )
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                    return

            if switch_occurred:
                # Allow the new model to reprocess the tokens generated since the last switch
                # Do not reset tokens_since_last_switch here; it should be processed by the new model
                continue  # Restart the loop with the new current_engine

                # If no switch occurred, reset tokens_since_last_switch
            tokens_since_last_switch = 0

        # In case the loop exits without hitting the termination conditions
        yield "data: [DONE]\n\n"
