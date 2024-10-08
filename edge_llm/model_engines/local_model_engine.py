from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch

from model_engines.base import BaseModelEngine, ModelType
from utils import calculate_entropy


class LocalModelEngine(BaseModelEngine):
    def __init__(
        self,
        model_type: ModelType,
        tokenizer: PreTrainedTokenizer,
        model_id: str,
        device: torch.device,
        entropy_thresholds: dict,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        super().__init__(
            model_type=model_type,
            tokenizer=tokenizer,
            model_id=model_id,
            device=device,
            entropy_thresholds=entropy_thresholds,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.eval()
        self.model.to(device)

    async def init_model(self, generated_ids: torch.Tensor):
        logits, past_key_values = await self.generate_next_token(generated_ids)
        self.past_key_values = past_key_values
        entropy, varentropy = calculate_entropy(logits)

        return entropy, varentropy

    async def stream_generate(
        self,
        generated_ids: torch.Tensor,
        tokens_generated: int,
        tokens_to_reprocess: int,
    ):
        """
        Streams tokens asynchronously by generating them one by one.
        Yields tokens or switch actions via the callback.
        """
        generated = generated_ids
        curr_tokens_generated = tokens_generated
        tokens_to_process = (
            tokens_to_reprocess if tokens_to_reprocess > 0 else 1
        )  # if no tokens to reprocess, process one token at a time

        while curr_tokens_generated < self.max_tokens:
            logits, past_key_values = await self.generate_next_token(
                generated[:, -tokens_to_process:]
            )
            tokens_to_process = (
                1  # after recomputing injected tokens, process one token at a time
            )

            entropy, varentropy = calculate_entropy(logits)

            # Determine if a switch is needed
            switch_action = self.should_switch(entropy, varentropy)
            if switch_action != "continue":
                yield {"action": switch_action}
                break

            # Sampling
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append the next token
            generated = torch.cat((generated, next_token), dim=-1)
            curr_tokens_generated += 1

            # Decode the token
            decoded_token = self.tokenizer.decode(
                next_token[0], skip_special_tokens=True
            )

            # Yield the token
            yield {"token": decoded_token, "token_id": next_token.item()}

            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                yield {"action": "done"}
                break

    async def generate_next_token(self, input_ids: torch.Tensor):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, past_key_values=self.past_key_values
            )
        logits = outputs.logits[:, -1, :] / self.temperature
        self.past_key_values = outputs.past_key_values
        return logits, self.past_key_values

    def should_switch(self, entropy: float, varentropy: float) -> str:
        if self.model_type == ModelType.SMALL:
            if (
                entropy > self.entropy_thresholds["large_model_entropy_threshold"]
                and varentropy
                > self.entropy_thresholds["large_model_varentropy_threshold"]
            ):
                return "switch_to_large"
        elif self.model_type == ModelType.LARGE:
            if (
                entropy < self.entropy_thresholds["small_model_entropy_threshold"]
                and varentropy
                < self.entropy_thresholds["small_model_varentropy_threshold"]
            ):
                return "switch_to_small"

        return "continue"
