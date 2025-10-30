import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# NOTE: Assuming these imports are now resolvable after fixing the tiktoken dependency
from decoder import GPTLanguageModel, Block, Head, MultiHeadAttention, FeedFoward
from generate_text import generate_from_prompt
from eval import evaluate

app = FastAPI()

# --- CORS Configuration (CRITICAL FOR CROSS-DOMAIN COMMUNICATION) ---
origins = [
    # The domain of your frontend
    "https://gpt-ui-feature-integ-cy14.bolt.host", 
    # Add other origins if needed (e.g., development localhost)
    # "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_new_tokens: int = 200

class EvalRequest(BaseModel):
    split: str = "test"
    num_batches: int = 100

@app.post("/generate_text")
def generate_text_endpoint(request: GenerateRequest):
    """
    Generates text based on the provided prompt and generation parameters.
    """
    # NOTE: The generate_from_prompt function needs to be correctly implemented 
    # to return a string under the key "generated_text".
    try:
        text = generate_from_prompt(
            prompt=request.prompt,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens
        )
        # Ensure 'text' is a string
        if not isinstance(text, str):
            text = str(text) 
        
        return {"generated_text": text}
    except Exception as e:
        # Catch any errors during generation and return a proper HTTP 500
        # This helps debugging on the client side
        print(f"Error during text generation: {e}")
        return {"detail": f"Generation failed: {e}"}, 500


@app.post("/eval")
def eval_endpoint(request: EvalRequest):
    """
    Runs a model evaluation and returns average loss and perplexity.
    """
    try:
        # NOTE: The evaluate function needs to be correctly implemented to return 
        # (avg_loss, avg_perplexity) as floats or numbers.
        avg_loss, avg_perplexity = evaluate(
            split=request.split,
            num_batches=request.num_batches
        )
        return {"avg_loss": avg_loss, "avg_perplexity": avg_perplexity}
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {"detail": f"Evaluation failed: {e}"}, 500

# This is only for local testing, Render uses the Start Command
