import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer
import torch

# from src.seq2seq_model import Encoder, Decoder, Seq2Seq
from utils.config import Config
from src.inference_utils import generate_response, load_model

# Initialize FastAPI app
app = FastAPI(title="Furniture Chatbot API")


@app.get("/")
def home():
    return {"message": "HOME"}

# Define request body structure
class ChatRequest(BaseModel):
    message: str

# Load tokenizer and model
tokenizer = Tokenizer.from_file(str(Config.TOKENIZER_PATH))



model = load_model(Config.CHECKPOINT_DIR / "best_model.pt")

@app.get("/chat")
def chat_info():
    return {"message": "Please use POST method with JSON: {'message': 'your message'}"}


# Define API route
@app.post("/chat")
def chat(request: ChatRequest):
    user_input = request.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        response = generate_response(model, tokenizer, user_input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
