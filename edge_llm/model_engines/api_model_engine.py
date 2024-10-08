import math
import random
from typing import AsyncGenerator
import aiohttp
import asyncio
import json

import torch
from transformers import PreTrainedTokenizer
from model_engines.base import BaseModelEngine, ModelType

from config import (
    EDGE_CLOUD_URL,
    MAX_NEW_TOKENS,
)


class APIModelEngine(BaseModelEngine):
    def __init__(
        self,
        model_id: str,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        entropy_thresholds: dict,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        super().__init__(
            model_type=ModelType.LARGE,
            model_id=model_id,
            tokenizer=tokenizer,
            device=device,
            entropy_thresholds=entropy_thresholds,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.request_id = None

    async def init_model(self, generated_ids: torch.Tensor):
        """
        Initializes the cloud model by sending the generated_ids to the server.
        """
        self.request_id = self._generate_request_id()

        prompt = self.tokenizer.decode(generated_ids[0])

        payload = {
            "request_id": self.request_id,
            "prompt": prompt,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{EDGE_CLOUD_URL}/init-cloud-model",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(
                        total=30
                    ),  # Set an appropriate timeout
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(
                            f"Failed to initialize cloud model: {response.status}, {text}"
                        )

                    data = await response.json()

                    entropy = data["data"]["entropy"]
                    varentropy = data["data"]["varentropy"]

                    return entropy, varentropy

        except aiohttp.ClientError as e:
            raise Exception(f"HTTP request failed: {e}")

    async def stream_generate(
        self,
        generated_ids: torch.Tensor,
        tokens_generated: int,
        tokens_to_reprocess: int,
    ) -> AsyncGenerator[dict, None]:
        """
        Streams tokens by making a request to the cloud API.
        Yields tokens or switch actions as received from the server.
        """
        curr_prompt_str = self.tokenizer.decode(generated_ids[0])
        prompt_str_to_reprocess = self.tokenizer.decode(
            generated_ids[0][-tokens_to_reprocess:]
        )

        payload = {
            "request_id": self.request_id,
            "temperature": self.temperature,
            "curr_prompt": curr_prompt_str,
            "prompt_substr_to_reprocess": prompt_str_to_reprocess,
            "entropy_threshold": self.entropy_thresholds.get(
                "small_model_entropy_threshold"
            ),
            "varentropy_threshold": self.entropy_thresholds.get(
                "small_model_varentropy_threshold"
            ),
            "max_tokens": self.max_tokens,
            "tokens_generated": tokens_generated,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{EDGE_CLOUD_URL}/stream-chat-cloud", json=payload
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise Exception(
                            f"Failed to stream chat cloud: {resp.status}, {text}"
                        )

                    # aiohttp provides an asynchronous iterator over the response content
                    async for line_bytes in resp.content:
                        line = line_bytes.decode("utf-8").strip()
                        if not line:
                            continue

                        if line.startswith("data: "):
                            data_str = line[len("data: ") :]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "error" in data:
                                    raise Exception(
                                        f"Error from cloud: {data['error']}"
                                    )

                                elif "action" in data:
                                    if data["action"] == "switch_to_local":
                                        yield {
                                            "action": "switch_to_small",
                                        }
                                        break
                                    elif data["action"] == "done":
                                        yield {"action": "done"}
                                        break
                                elif "token" in data:
                                    yield {
                                        "token": data["token"],
                                        "token_id": data["token_id"],
                                    }
                            except json.JSONDecodeError:
                                continue

        except aiohttp.ClientError as e:
            raise Exception(f"HTTP request failed: {e}")

    def should_switch(self, entropy: float, varentropy: float) -> str:
        """
        For APIModelEngine, switching is handled on the server side.
        This method can remain as a no-op or return "continue".
        """
        return "continue"

    @staticmethod
    def _generate_request_id() -> str:
        """
        Generates a unique request ID.
        """
        import uuid

        return str(uuid.uuid4())
