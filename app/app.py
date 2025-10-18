import gradio as gr, argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from peft import PeftModel
from datasets_utils import build_label_maps, ENT_LABELS

def load_pipeline(pretrained, adapter_dir):
    id2label,label2id = build_label_maps()
    tok = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
    base = AutoModelForTokenClassification.from_pretrained(pretrained, num_labels=len(ENT_LABELS), id2label=id2label, label2id=label2id)
    model = PeftModel.from_pretrained(base, adapter_dir)
    nlp = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple")
    return nlp

def infer(nlp, text):
    out = nlp(text)
    tokens = []
    last_end = 0
    for o in out:
        tokens.append((text[o["start"]:o["end"]], o["entity_group"]))
    if not tokens:
        return [(w,"O") for w in text.split()]
    return tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, default="bert-base-multilingual-cased")
    ap.add_argument("--adapter", type=str, default="outputs/active/final_adapter")
    args = ap.parse_args()
    nlp = load_pipeline(args.pretrained, args.adapter)
    demo = gr.Interface(fn=lambda t: infer(nlp,t), inputs=gr.Textbox(lines=3,label="Escribe una oración en español"), outputs=gr.HighlightedText(label="Entidades"))
    demo.launch()

if __name__ == "__main__":
    main()
