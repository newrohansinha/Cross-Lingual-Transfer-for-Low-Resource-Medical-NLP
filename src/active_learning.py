import argparse, os, random, numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets_utils import load_spanish_csv, align_labels_with_tokens, build_label_maps, ENT_LABELS
import evaluate

def split_initial_pool(ds, seed, init_n):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    init_idx = idx[:init_n]
    pool_idx = idx[init_n:]
    return init_idx, pool_idx

def subset(ds, indices):
    return Dataset.from_dict({k:[v[i] for i in indices] for k,v in ds.to_dict().items()})

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [ENT_LABELS[p] for (p,l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ENT_LABELS[l] for (p,l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision":results["overall_precision"],"recall":results["overall_recall"],"f1":results["overall_f1"],"accuracy":results["overall_accuracy"]}

def train_lora(base_model_dir, pretrained, train_ds, val_ds, out_dir, epochs, batch_size, seed):
    id2label,label2id = build_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
    tokenized_train = train_ds.map(lambda x: align_labels_with_tokens(tokenizer, x), batched=True)
    tokenized_val = val_ds.map(lambda x: align_labels_with_tokens(tokenizer, x), batched=True)
    base = AutoModelForTokenClassification.from_pretrained(base_model_dir, num_labels=len(ENT_LABELS), id2label=id2label, label2id=label2id)
    peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query","key","value","dense"])
    model = get_peft_model(base, peft_config)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=3e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=seed
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer, tokenizer, model

def score_uncertainty(model, tokenizer, ds):
    scores = []
    for ex in ds:
        enc = tokenizer(ex["tokens"], is_split_into_words=True, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        word_ids = enc.word_ids(0)
        mask = [i for i,w in enumerate(word_ids) if w is not None]
        if mask:
            m = probs[mask].max(axis=1)
            scores.append(float(1.0 - m.mean()))
        else:
            scores.append(0.0)
    return np.array(scores)

def run_strategy(strategy, base_model_dir, pretrained, full_train, test, round_size, rounds, epochs, batch_size, seed, work_dir):
    init_idx, pool_idx = split_initial_pool(full_train, seed, round_size)
    labeled = subset(full_train, init_idx)
    pool = subset(full_train, pool_idx)
    f1s = []
    for r in range(rounds):
        out_dir = os.path.join(work_dir, f"{strategy}_round_{r}")
        trainer, tokenizer, model = train_lora(base_model_dir, pretrained, labeled, test, out_dir, epochs, batch_size, seed)
        eval_metrics = trainer.evaluate()
        f1s.append(eval_metrics["eval_f1"])
        if len(pool)==0:
            break
        if strategy=="active":
            scores = score_uncertainty(model, tokenizer, pool)
            sel = np.argsort(scores)[-round_size:][::-1]
        else:
            sel = np.random.RandomState(seed+r).choice(len(pool), size=min(round_size,len(pool)), replace=False)
        add = subset(pool, sel.tolist())
        labeled = Dataset.from_dict({k:(labeled[k]+add[k]) for k in labeled.features})
        keep = [i for i in range(len(pool)) if i not in set(sel.tolist())]
        pool = subset(pool, keep) if keep else Dataset.from_list([])
    return f1s, labeled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--pretrained", type=str, default="bert-base-multilingual-cased")
    ap.add_argument("--train_csv", type=str, default="data/es_train.csv")
    ap.add_argument("--test_csv", type=str, default="data/es_test.csv")
    ap.add_argument("--round_size", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/active")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_spanish_csv(args.train_csv, args.test_csv)
    active_f1, labeled_final = run_strategy("active", args.base_model, args.pretrained, ds["train"], ds["validation"], args.round_size, args.rounds, args.epochs, args.batch_size, args.seed, os.path.join(args.out_dir,"runs"))
    rand_f1, _ = run_strategy("random", args.base_model, args.pretrained, ds["train"], ds["validation"], args.round_size, args.rounds, args.epochs, args.batch_size, args.seed, os.path.join(args.out_dir,"runs"))
    rounds = list(range(1, len(active_f1)+1))
    df = pd.DataFrame({"round":rounds,"n_labels":[args.round_size*r for r in rounds],"active_f1":active_f1[:len(rounds)],"random_f1":rand_f1[:len(rounds)]})
    df.to_csv(os.path.join(args.out_dir,"curves.csv"), index=False)
    plt.figure()
    plt.plot(df["n_labels"], df["active_f1"], label="Active Learning")
    plt.plot(df["n_labels"], df["random_f1"], label="Random Sampling")
    plt.xlabel("Labeled examples")
    plt.ylabel("F1")
    plt.title("Active Learning vs Random Sampling")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir,"learning_curves.png"), bbox_inches="tight")
    id2label = {i:l for i,l in enumerate(ENT_LABELS)}
    label2id = {l:i for i,l in enumerate(ENT_LABELS)}
    from transformers import AutoModelForTokenClassification
    from peft import LoraConfig, get_peft_model, TaskType
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)
    base = AutoModelForTokenClassification.from_pretrained(args.base_model, num_labels=len(ENT_LABELS), id2label=id2label, label2id=label2id)
    peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query","key","value","dense"])
    model = get_peft_model(base, peft_config)
    tokenized_train = labeled_final.map(lambda x: align_labels_with_tokens(tokenizer, x), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    args_tr = TrainingArguments(output_dir=os.path.join(args.out_dir,"final_adapter"), per_device_train_batch_size=args.batch_size, num_train_epochs=args.epochs, learning_rate=3e-4, save_strategy="no")
    trainer = Trainer(model=model, args=args_tr, train_dataset=tokenized_train, tokenizer=tokenizer, data_collator=data_collator)
    trainer.train()
    model.save_pretrained(os.path.join(args.out_dir,"final_adapter"))
    tokenizer.save_pretrained(os.path.join(args.out_dir,"final_adapter"))

if __name__ == "__main__":
    main()
