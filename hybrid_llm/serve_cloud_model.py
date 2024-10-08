import traceback
from typing import Optional

from fastapi import FastAPI, Body, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi.responses import StreamingResponse
import json
import logging
import time

from config import (
    MODEL_LARGE_ID,
    SPECIAL_TOKENS,
    EOS_TOKEN,
)
from utils import calculate_entropy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_LARGE_ID)
tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelForCausalLM.from_pretrained(MODEL_LARGE_ID, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

# State management: store past_key_values per request_id
past_key_values_per_request = {}
# Store the timestamp of the last activity per request_id for cleanup
request_timestamps = {}
REQUEST_TIMEOUT = 3600  # 1 hour


async def cleanup_old_requests():
    while True:
        current_time = time.time()
        to_delete = [
            req_id
            for req_id, timestamp in request_timestamps.items()
            if current_time - timestamp > REQUEST_TIMEOUT
        ]
        for req_id in to_delete:
            past_key_values_per_request.pop(req_id, None)
            request_timestamps.pop(req_id, None)
            logger.info(f"Cleaned up expired request_id: {req_id}")
        await asyncio.sleep(600)  # Run cleanup every 10 minutes


def generate_next_token(
    input_ids: torch.Tensor, temperature: float, past_key_values=None
):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=past_key_values)
    logits = outputs.logits[:, -1, :] / temperature
    return logits, outputs.past_key_values


def clean_up_request(request_id: str):
    past_key_values_per_request.pop(request_id, None)
    request_timestamps.pop(request_id, None)
    logger.info(f"Cleaned up state for request_id: {request_id}")


@app.post("/init-cloud-model")
async def init_cloud_model(
    request_id: str = Body(..., description="The unique request ID"),
    prompt: str = Body(..., description="The prompt to initialize the model"),
):
    try:
        # input_ids = torch.tensor([generated_ids], device=device)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Initialize the model with the generated token IDs
        logits, past_key_values = generate_next_token(input_ids, temperature=0.7)
        past_key_values_per_request[request_id] = past_key_values
        request_timestamps[request_id] = time.time()

        entropy, varentropy = calculate_entropy(logits)

        logger.info(f"Initialized cloud model for request_id: {request_id}")
        return {
            "message": "Cloud model initialized",
            "data": {"entropy": entropy, "varentropy": varentropy},
        }
    except Exception as e:
        logger.error(f"Error initializing cloud model for request_id {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream-chat-cloud")
async def stream_chat_cloud(
    request_id: str = Body(..., description="The unique request ID"),
    curr_prompt: str = Body(..., description="The ongoing prompt generation"),
    prompt_substr_to_reprocess: str = Body(
        ..., description="The prompt substring to reprocess"
    ),
    entropy_threshold: float = Body(
        ..., description="Entropy threshold to switch back"
    ),
    varentropy_threshold: float = Body(
        ..., description="Varentropy threshold to switch back"
    ),
    temperature: float = Body(..., description="The temperature for sampling"),
    max_tokens: int = Body(..., description="Maximum number of tokens to generate"),
    tokens_generated: int = Body(..., description="Number of tokens already generated"),
):
    async def token_generator():
        nonlocal request_id, temperature, curr_prompt, prompt_substr_to_reprocess, entropy_threshold, varentropy_threshold, max_tokens, tokens_generated

        try:
            past_key_values = past_key_values_per_request.get(request_id, None)
            if past_key_values is None:
                raise HTTPException(
                    status_code=400, detail="Model not initialized for this request_id"
                )

            generated_ids = tokenizer(curr_prompt, return_tensors="pt").input_ids.to(
                device
            )
            new_prompt_substr_token_ids = tokenizer(
                prompt_substr_to_reprocess, return_tensors="pt"
            ).input_ids.to(device)
            num_tokens_to_reprocess = new_prompt_substr_token_ids.size(1)

            # generated_ids_tensor = torch.tensor([generated_ids], device=device)
            generated = generated_ids
            curr_tokens_generated = tokens_generated
            tokens_to_process = (
                num_tokens_to_reprocess if num_tokens_to_reprocess > 0 else 1
            )

            while curr_tokens_generated < max_tokens:
                logits, past_key_values = generate_next_token(
                    generated[:, -tokens_to_process:],
                    temperature=temperature,
                    past_key_values=past_key_values,
                )
                past_key_values_per_request[request_id] = past_key_values
                request_timestamps[request_id] = time.time()
                tokens_to_process = (
                    1  # after recomputing injected tokens, process one token at a time
                )

                entropy, varentropy = calculate_entropy(logits)

                # Determine if a switch is needed
                if entropy < entropy_threshold and varentropy < varentropy_threshold:
                    data = {"action": "switch_to_local"}
                    yield f"data: {json.dumps(data)}\n\n"
                    break

                # Sampling
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append the next token
                generated = torch.cat((generated, next_token), dim=-1)
                curr_tokens_generated += 1

                # Decode the token
                decoded_token = tokenizer.decode(
                    next_token[0], skip_special_tokens=True
                )

                # Yield the token
                data = {"token": decoded_token, "token_id": next_token.item()}
                yield f"data: {json.dumps(data)}\n\n"

                # Check for EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    data = {"action": "done", "finish_reason": "STOP"}
                    yield f"data: {json.dumps(data)}\n\n"

                    # Cleanup if finished
                    clean_up_request(request_id)

                    break

            if curr_tokens_generated >= max_tokens:
                data = {"action": "done", "finish_reason": "MAX_TOKENS"}
                yield f"data: {json.dumps(data)}\n\n"

                # Cleanup if finished
                clean_up_request(request_id)

        except HTTPException as he:
            logger.error(f"HTTPException for request_id {request_id}: {he.detail}")
            yield f'data: {{"error": "{he.detail}"}}\n\n'

            print(traceback.format_exc())

            raise he

        except Exception as e:
            logger.error(f"Error in stream_chat_cloud for request_id {request_id}: {e}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'

            print(traceback.format_exc())

            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(token_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    import asyncio

    uvicorn.run(app, host="0.0.0.0", port=8081)
