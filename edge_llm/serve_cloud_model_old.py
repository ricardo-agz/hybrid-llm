from fastapi import FastAPI, Body
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

from edge_llm.config import (
    MODEL_LARGE_ID,
    SPECIAL_TOKENS,
    EOS_TOKEN,
)


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_LARGE_ID)
tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelForCausalLM.from_pretrained(MODEL_LARGE_ID, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

past_key_values_per_request = {}


@app.post("/chat-cloud")
async def run_prediction(
    request_id: str = Body(..., description="The unique request ID"),
    temperature: float = Body(..., description="The temperature for sampling"),
    generated_ids: list[int] = Body(..., description="The generated token IDs"),
    num_tokens_to_reprocess: int
    | None = Body(None, description="The number of tokens to reprocess"),
):
    # Retrieve past_key_values for the request
    past_key_values = past_key_values_per_request.get(request_id, None)
    generated_ids_tensor = torch.tensor([generated_ids], device=device)

    if num_tokens_to_reprocess is not None:
        generated_tokens_to_process = generated_ids_tensor[:, -num_tokens_to_reprocess:]
    else:
        generated_tokens_to_process = generated_ids_tensor

    outputs = model(
        input_ids=generated_tokens_to_process,
        past_key_values=past_key_values,
        return_dict=True,
    )
    past_key_values_per_request[request_id] = outputs.past_key_values

    next_token_logits = outputs.logits[:, -1, :]
    next_token_logits = next_token_logits / temperature

    # Sample the next token
    probs = F.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return {
        "next_token": next_token.item(),
        "next_token_logits": next_token_logits.squeeze(0).tolist(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
