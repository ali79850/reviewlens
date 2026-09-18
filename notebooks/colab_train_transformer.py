# ReviewLens — Phase 5: DistilBERT sentiment fine-tune (run on Google Colab)
#
# Steps:
#  1. colab.research.google.com -> New notebook -> Runtime -> Change runtime type -> T4 GPU
#  2. Paste this whole file into one cell (or split at the "# ---" markers into
#     separate cells) and run top to bottom.
#  3. When prompted, upload train.parquet and test.parquet from your local
#     D:\reviewlens\reviewlens\data\processed\ folder.
#  4. At the end it saves + zips the model and offers a download — save that
#     zip into D:\reviewlens\reviewlens\models\transformer_sentiment\ locally.

# ---
!pip install -q transformers datasets evaluate accelerate

# ---
from google.colab import files
print("Upload train.parquet and test.parquet now:")
uploaded = files.upload()

# ---
import pandas as pd

train_df = pd.read_parquet("train.parquet")
test_df = pd.read_parquet("test.parquet")

# Keep this run tractable on a free Colab GPU within a normal session length.
# 20k/5k is enough to get a real, honest comparison against the baseline
# without requiring a multi-hour training run.
TRAIN_SUBSET = 20_000
TEST_SUBSET = 5_000

train_df = train_df.sample(n=min(TRAIN_SUBSET, len(train_df)), random_state=42).reset_index(drop=True)
test_df = test_df.sample(n=min(TEST_SUBSET, len(test_df)), random_state=42).reset_index(drop=True)

print(f"Training on {len(train_df)} rows, evaluating on {len(test_df)} rows")
print(train_df["sentiment"].value_counts())

# ---
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text

train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

LABELS = ["Negative", "Neutral", "Positive"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

train_df["label"] = train_df["sentiment"].map(label2id)
test_df["label"] = test_df["sentiment"].map(label2id)

# ---
from datasets import Dataset

train_ds = Dataset.from_pandas(train_df[["text_clean", "label"]])
test_ds = Dataset.from_pandas(test_df[["text_clean", "label"]])

# ---
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text_clean"], truncation=True, max_length=256, padding="max_length")

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.remove_columns(["text_clean"])
test_ds = test_ds.remove_columns(["text_clean"])
train_ds.set_format("torch")
test_ds.set_format("torch")

# ---
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
    }

# ---
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./distilbert_sentiment_out",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=50,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ---
final_metrics = trainer.evaluate()
print("Final transformer metrics on held-out test subset:")
print(final_metrics)

import json
with open("transformer_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

# ---
# Save model + tokenizer, zip, and download
SAVE_DIR = "distilbert_sentiment_finetuned"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

import shutil
shutil.make_archive("distilbert_sentiment_finetuned", "zip", SAVE_DIR)
shutil.copy("transformer_metrics.json", SAVE_DIR)

files.download("distilbert_sentiment_finetuned.zip")
files.download("transformer_metrics.json")