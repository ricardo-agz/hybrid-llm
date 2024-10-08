from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from dynamic_model import DynamicModel, DynamicModelVariant
from utils import build_prompt


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize the chat model
dynamic_local_model = DynamicModel()
edge_cloud_model = DynamicModel(variant=DynamicModelVariant.EDGE_CLOUD)


@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    conversation = data.get("conversation", [])

    async def generate():
        prompt = build_prompt(conversation)
        conversation.append({"role": "assistant", "content": ""})  # Placeholder

        async for chunk in dynamic_local_model.generate_response(prompt):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat-local-cloud")
async def chat_local_cloud_endpoint(request: Request):
    data = await request.json()
    conversation = data.get("conversation", [])

    async def generate():
        prompt = build_prompt(conversation)
        conversation.append({"role": "assistant", "content": ""})

        async for chunk in edge_cloud_model.generate_response(prompt):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
