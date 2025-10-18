# Low-Resource Clinical NER with LoRA + Active Learning

## Objective
Adapt a clinical NER model to a low-resource language using few-shot learning and simulated active learning.

## Tech Stack
Hugging Face Transformers, PyTorch, PEFT (LoRA), Datasets, Gradio, Pandas, Matplotlib, seqeval.

## Project Structure
```
low_resource_clinical_ner/
├── README.md
├── requirements.txt
├── data/
│   ├── es_train.csv
│   └── es_test.csv
├── src/
│   ├── datasets_utils.py
│   ├── train_english.py
│   ├── active_learning.py
│   └── infer_utils.py
├── app/
│   └── app.py
└── outputs/
```
Data files use space-separated tokens and BIO tags per row.

## Quickstart
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Fine-tune mBERT on English biomedical NER
```
python src/train_english.py   --pretrained bert-base-multilingual-cased   --dataset jnlpba   --output_dir outputs/english_jnlpba   --epochs 3   --batch_size 16
```

### 2) Simulated Active Learning on your low-resource Spanish set
Place your hand-labeled files at `data/es_train.csv` and `data/es_test.csv` with columns:
- `tokens`: space-separated tokens
- `labels`: space-separated BIO tags using {O,B-ENT,I-ENT}

Run:
```
python src/active_learning.py   --base_model outputs/english_jnlpba   --pretrained bert-base-multilingual-cased   --train_csv data/es_train.csv   --test_csv data/es_test.csv   --round_size 20   --rounds 9   --epochs 3   --batch_size 8   --seed 42   --out_dir outputs/active
```

This trains two curves:
- Active-Learning: pick lowest-confidence samples each round
- Random-Sampling: add random samples each round

Artifacts:
- `outputs/active/curves.csv`
- `outputs/active/learning_curves.png`
- Final adapter: `outputs/active/final_adapter`

### 3) Gradio Demo
After training, launch:
```
python app/app.py --pretrained bert-base-multilingual-cased --adapter outputs/active/final_adapter
```
Open the printed local URL and type a Spanish sentence to see highlighted medical entities.

## Data Format Example
```
tokens,labels
"El paciente tiene diabetes mellitus","O O O B-ENT I-ENT"
"Se prescribió ibuprofeno 200 mg","O O B-ENT O O"
```
Tags must align token-for-token. Label set is {O,B-ENT,I-ENT}.

## Target Metrics
- ≥80% F1 on the Spanish holdout set (depends on label quality/size and training budget)
- Active Learning curve dominates Random Sampling in label efficiency

## Repro Tips
- Use a GPU
- Increase rounds/epochs if your F1 plateaus
- Ensure consistent BIO tagging and tokenization
