from abc import abstractmethod, ABC
from enum import Enum
import torch
from transformers import PreTrainedTokenizer


class ModelType(Enum):
    SMALL = "small"
    LARGE = "large"


class BaseModelEngine(ABC):
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
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.entropy_thresholds = entropy_thresholds
        self.past_key_values = None
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    async def init_model(self, generated_ids: torch.Tensor):
        """
        Initializes the model with generated token IDs.
        """

    @abstractmethod
    async def stream_generate(
        self,
        generated_ids: torch.Tensor,
        tokens_generated: int,
        tokens_to_reprocess: int,
    ):
        """
        Asynchronously generates tokens and yields them.
        Can yield tokens or switch actions based on entropy and varentropy.
        """
        pass

    @abstractmethod
    def should_switch(self, entropy: float, varentropy: float) -> str:
        """
        Determines whether to switch models based on entropy and varentropy.
        Returns 'switch_to_small', 'switch_to_large', or 'continue'.
        """
        pass
