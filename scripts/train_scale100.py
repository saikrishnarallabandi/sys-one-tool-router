# AUTO-GENERATED from the 25-class original for the 101-class scale run.
# Same logic/config; only paths (data/scale100, exp/scale100,
# qwen3-0.6b-tool-router-100) differ.

#!/usr/bin/env python3
"""Fine-tune Qwen3-0.6B for multi-class tool routing, LoRA style.

Mirrors system-one-poc/train.py: fp32 frozen backbone, fp32 LoRA adapters +
score head, fp16 autocast for compute (full fp16 diverged on Pascal).
Differences vs POC1: num_labels=25, class-weighted loss (long tail +
no_tool majority), macro-F1 as the model-selection metric.

Trains on CUDA device set by CUDA_VISIBLE_DEVICES (GPU0: ollama resident).
Saves the MERGED model so infer.py needs no peft.
"""
import json, os, torch
import numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback,
                          DataCollatorWithPadding)
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_LEN = 128
OUT = "exp/checkpoints/qwen3-0.6b-tool-router-100"
DATA_DIR = "data/scale100"

def load_split(path, tok):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"]); labels.append(r["label"])
    enc = tok(texts, truncation=True, max_length=MAX_LEN)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "labels": labels}

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, d):
        self.d = d
    def __len__(self): return len(self.d["labels"])
    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.d.items()}

class WeightedTrainer(Trainer):
    """Cross-entropy with inverse-frequency class weights."""
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        w = self.class_weights.to(logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss

def main():
    with open(f"{DATA_DIR}/label_map.json") as f:
        lm = json.load(f)
    num_labels = lm["num_labels"]
    print(f"num_labels = {num_labels}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=num_labels, trust_remote_code=True, dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
        task_type=TaskType.SEQ_CLS)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_d = ListDataset(load_split(f"{DATA_DIR}/data_train.jsonl", tok))
    val_d = ListDataset(load_split(f"{DATA_DIR}/data_val.jsonl", tok))

    # class weights from the TRAIN split
    counts = np.bincount([r for r in
                          (json.loads(l)["label"] for l in open(f"{DATA_DIR}/data_train.jsonl"))],
                         minlength=num_labels).astype(np.float64)
    weights = counts.sum() / (num_labels * np.maximum(counts, 1))
    print("class counts:", counts.astype(int).tolist())
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print("class weights:", [round(w, 3) for w in weights])

    args = TrainingArguments(
        output_dir=OUT, per_device_train_batch_size=16, per_device_eval_batch_size=64,
        num_train_epochs=3, learning_rate=2e-4, weight_decay=0.01,
        warmup_steps=100, lr_scheduler_type="cosine",
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        fp16=True, logging_steps=50, seed=42, report_to="none",
    )
    trainer = WeightedTrainer(
        class_weights, model=model, args=args, train_dataset=train_d,
        eval_dataset=val_d, compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()

    merged = trainer.model.merge_and_unload()
    best_dir = os.path.join(OUT, "best")
    os.makedirs(best_dir, exist_ok=True)
    merged.save_pretrained(best_dir)
    tok.save_pretrained(best_dir)
    print("saved merged model to", best_dir)

if __name__ == "__main__":
    main()
