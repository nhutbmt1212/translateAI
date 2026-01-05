from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from grammar_teacher import GrammarTeacher
import asyncio
import json

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to match your React app's URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the teacher
teacher = GrammarTeacher()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    is_new = data.get("is_new", False)

    if not message:
        return {"error": "Message is empty"}

    def generate():
        buffer = ""
        batch_size = 15 # Increased for maximum performance
        token_count = 0
        
        for token in teacher.ask_stream(message, is_new_sentence=is_new):
            buffer += token
            token_count += 1
            if token_count >= batch_size:
                yield buffer
                buffer = ""
                token_count = 0
        
        if buffer: # Yield remaining tokens
            yield buffer

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/reset")
async def reset():
    teacher.reset_chat()
    return {"status": "chat reset"}

@app.post("/cancel")
async def cancel():
    teacher.stop_generation()
    return {"status": "generation stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
