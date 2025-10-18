import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from typing import List, Dict, Any
import numpy as np

ENT_LABELS = ["O","B-ENT","I-ENT"]

def map_to_ent_labels(example, id2label):
    new_tags = []
    for t in example["ner_tags"]:
        new_tags.append(0 if t==0 else (1 if str(t).startswith("1") or str(t).startswith("B") else 2))
    example["ner_tags"]=new_tags
    return example

def load_english_biomed(dataset_name:str="jnlpba"):
    ds = load_dataset(dataset_name)
    def to_ent(example):
        tokens = example["tokens"]
        tags = example["ner_tags"]
        new = []
        prev_o = True
        for tag in tags:
            if tag==0:
                new.append(0)
                prev_o = True
            else:
                new.append(1 if prev_o else 2)
                prev_o = False
        return {"tokens":tokens,"ner_tags":new}
    ds = ds.map(to_ent)
    features = ds["train"].features
    features["ner_tags"].feature.names = ENT_LABELS
    return ds

def load_spanish_csv(train_csv:str, test_csv:str):
    def read_csv(p):
        df = pd.read_csv(p)
        rows = []
        for _,r in df.iterrows():
            toks = r["tokens"].split()
            labs = r["labels"].split()
            rows.append({"tokens":toks,"ner_tags":[ENT_LABELS.index(x) for x in labs]})
        return Dataset.from_list(rows)
    train = read_csv(train_csv)
    test = read_csv(test_csv)
    return DatasetDict({"train":train,"validation":test,"test":test})

def align_labels_with_tokens(tokenizer, examples, label_all_tokens=False):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def build_label_maps():
    id2label = {i:l for i,l in enumerate(ENT_LABELS)}
    label2id = {l:i for i,l in enumerate(ENT_LABELS)}
    return id2label,label2id
