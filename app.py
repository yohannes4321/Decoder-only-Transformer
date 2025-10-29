from fastapi import FastAPI
from pydantic import BaseModel
from decoder import GPTLanguageModel, Block, Head, MultiHeadAttention, FeedFoward
from generate_text import generate_from_prompt
from eval import evaluate

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_new_tokens: int = 200

class EvalRequest(BaseModel):
    split: str = "test"
    num_batches: int = 100

@app.post("/generate_text")
def generate_text_endpoint(request: GenerateRequest):
    text = generate_from_prompt(
        prompt=request.prompt,
        temperature=request.temperature,
        max_new_tokens=request.max_new_tokens
    )
    return {"generated_text": text}

@app.post("/eval")
def eval_endpoint(request: EvalRequest):
    avg_loss, avg_perplexity = evaluate(
        split=request.split,
        num_batches=request.num_batches
    )
    return {"avg_loss": avg_loss, "avg_perplexity": avg_perplexity}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
