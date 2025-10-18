import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel
from typing import List, Tuple

def load_adapter(pretrained:str, adapter_dir:str, num_labels:int, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
    base = AutoModelForTokenClassification.from_pretrained(pretrained, num_labels=num_labels, id2label=id2label, label2id=label2id)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model

def sequence_uncertainty(model, tokenizer, tokens:List[str]):
    with torch.no_grad():
        enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
        logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        word_ids = enc.word_ids(0)
        mask = [i for i,w in enumerate(word_ids) if w is not None]
        if not mask:
            return 0.0
        m = probs[mask].max(axis=1)
        return float(1.0 - m.mean())
