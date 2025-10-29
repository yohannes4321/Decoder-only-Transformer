# tests/test_decoder_basic.py
import importlib
import torch
decoder = importlib.import_module("decoder")

def test_model_forward_shapes():
    model = decoder.GPTLanguageModel()
    model.eval()
    # create a tiny sample input using existing encode() on a small string
    if hasattr(decoder, "encode"):
        idx = torch.tensor([decoder.encode("hello")], dtype=torch.long)
    else:
        idx = torch.randint(0, 10, (1, 8), dtype=torch.long)
    logits, loss = model(idx)
    assert logits.ndim == 3  # (B, T, C)
