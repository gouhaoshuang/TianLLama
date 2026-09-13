import argparse
from pathlib import Path
import transformers
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

def load_model(MODEL_ID , REVISION):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=False
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            dtype=torch.float32,
            attn_implementation="eager",
            use_safetensors=True,
            trust_remote_code=False,
        )
        .cpu()
        .eval()
    )
    return tokenizer, model
