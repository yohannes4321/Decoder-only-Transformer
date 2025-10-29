# tests/test_api.py
import os
import importlib
import torch
import tempfile
from pathlib import Path

# project root is one level up from tests/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
MODEL_PATH = PROJECT_ROOT / "decoder.pth"

# Ensure Data dir with minimal CSVs exists BEFORE importing modules that read them on import
DATA_DIR.mkdir(exist_ok=True)
for fname in ("train.csv", "validation.csv", "test.csv"):
    p = DATA_DIR / fname
    if not p.exists():
        p.write_text("hello\n")   # tiny content so encode() works

# Import the decoder to get GPTLanguageModel class defined
decoder = importlib.import_module("decoder")

# Create a tiny model and save it so generate_text.py and eval.py can load it at import time
if not MODEL_PATH.exists():
    model = decoder.GPTLanguageModel()
    torch.save(model, MODEL_PATH)

# Now import the fastapi app (which will import generate_text)
from app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_generate_text_endpoint():
    payload = {"prompt": "Hello", "temperature": 0.7, "max_new_tokens": 5}
    resp = client.post("/generate_text", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "generated_text" in body
    assert isinstance(body["generated_text"], str)

def test_eval_endpoint():
    payload = {"split": "test", "num_batches": 1}
    resp = client.post("/eval", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "avg_loss" in body
    assert "avg_perplexity" in body
