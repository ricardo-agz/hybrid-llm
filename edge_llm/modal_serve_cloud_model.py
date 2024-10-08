import modal
from modal import method, asgi_app, enter, gpu, Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

from edge_llm.config import (
    MODEL_LARGE_ID,
    SPECIAL_TOKENS,
    EOS_TOKEN,
)

# ANSI escape codes for colors
RED = "\033[91m"
RESET = "\033[0m"

image = Image.debian_slim().pip_install(
    "torch",
    "transformers",
)

GPU_CONFIG = gpu.T4(count=1)
stub = modal.App(
    "edge-cloud-serve-llm",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


@stub.cls(gpu=GPU_CONFIG)
class CloudModel:
    @enter()
    def startup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_LARGE_ID)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_LARGE_ID, torch_dtype=torch.bfloat16
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(EOS_TOKEN)

        self.past_key_values_per_request = {}

    @method()
    def generate(
        self,
        request_id: str,
        temperature: float,
        generated_ids: list[int],
        num_tokens_to_reprocess: int | None = None,
    ):
        past_key_values = self.past_key_values_per_request.get(request_id, None)
        generated_ids_tensor = torch.tensor([generated_ids], device=self.device)

        generated_tokens_to_process = (
            generated_ids_tensor[:, -num_tokens_to_reprocess:]
            if num_tokens_to_reprocess is not None
            else generated_ids_tensor
        )

        outputs = self.model(
            input_ids=generated_tokens_to_process,
            past_key_values=past_key_values,
        )
        self.past_key_values_per_request[request_id] = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]
        # Scale the logits by the temperature parameter
        next_token_logits = next_token_logits / temperature

        # Apply softmax to convert to probabilities
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        probs = torch.exp(log_probs)

        # Sample the next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Return the next token and logits (before softmax)
        return {
            "next_token": next_token.item(),
            "next_token_logits": next_token_logits.squeeze(0).tolist(),
        }


@stub.function()
@asgi_app(label="cloud-llm-api")
def app():
    import fastapi
    from fastapi import Body

    web_app = fastapi.FastAPI()

    @web_app.post("/chat-cloud")
    async def run_prediction(
        request_id: str = Body(..., description="The unique request ID"),
        temperature: float = Body(..., description="The temperature for sampling"),
        generated_ids: list[int] = Body(..., description="The generated token IDs"),
        num_tokens_to_reprocess: int
        | None = Body(None, description="The number of tokens to reprocess"),
    ):
        model = CloudModel()
        result = model.generate.remote(
            request_id, temperature, generated_ids, num_tokens_to_reprocess
        )

        return result

    return web_app
