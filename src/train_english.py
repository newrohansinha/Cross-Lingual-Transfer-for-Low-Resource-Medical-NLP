import argparse, os, numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets_utils import load_english_biomed, align_labels_with_tokens, build_label_maps, ENT_LABELS
import evaluate

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, default="bert-base-multilingual-cased")
    ap.add_argument("--dataset", type=str, default="jnlpba")
    ap.add_argument("--output_dir", type=str, default="outputs/english_jnlpba")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    id2label,label2id = build_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)
    ds = load_english_biomed(args.dataset)
    tokenized = ds.map(lambda x: align_labels_with_tokens(tokenizer, x), batched=True)
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained, num_labels=len(ENT_LABELS), id2label=id2label, label2id=label2id)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
