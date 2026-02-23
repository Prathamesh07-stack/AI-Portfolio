"""
Experiment 5 – Small LoRA Fine-Tune (PEFT)
Use DistilBERT + LoRA adapters to fine-tune on SST-2 sentiment classification.
Compares trainable-parameter count and accuracy vs. baseline.
"""

import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, TaskType, get_peft_model
import evaluate


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
MODEL_NAME     = "distilbert-base-uncased"
DATASET_NAME   = "sst2"
TRAIN_SAMPLES  = 1000   # subset to keep training fast on CPU
EVAL_SAMPLES   = 500
NUM_EPOCHS     = 3
BATCH_SIZE     = 16
LEARNING_RATE  = 3e-4
MAX_SEQ_LEN    = 128

# LoRA hyper-parameters
LORA_R         = 8       # rank
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.1
# DistilBERT uses q_lin / v_lin inside attention layers
LORA_TARGETS   = ["q_lin", "v_lin"]


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def count_params(model) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def compute_metrics(eval_pred):
    metric     = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def make_trainer(model, tokenized_ds, tokenizer, output_dir: str) -> Trainer:
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",          # disable wandb / tensorboard
        use_cpu=not torch.cuda.is_available(),
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    section(f"Hardware: running on {device.upper()}")

    # ── Load dataset ────────────────────────────────────────────────────
    section("Loading Dataset – SST-2")
    raw = load_dataset("glue", DATASET_NAME)
    raw["train"]      = raw["train"].select(range(TRAIN_SAMPLES))
    raw["validation"] = raw["validation"].select(range(EVAL_SAMPLES))
    print(f"  Train samples      : {len(raw['train'])}")
    print(f"  Validation samples : {len(raw['validation'])}")
    print(f"  Label names        : {raw['train'].features['label'].names}")

    # ── Tokenize ────────────────────────────────────────────────────────
    section("Tokenizing")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=MAX_SEQ_LEN)

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # ── Baseline (pre-fine-tune) accuracy ───────────────────────────────
    section("Baseline – Zero-Shot Accuracy (no fine-tuning)")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    base_model.eval()

    base_trainer = make_trainer(base_model, tokenized, tokenizer, "./results_baseline")
    baseline_results = base_trainer.evaluate()
    baseline_acc     = baseline_results.get("eval_accuracy", 0.0)
    print(f"  Baseline accuracy  : {baseline_acc:.4f}  ({baseline_acc*100:.1f}%)")

    # ── Build LoRA model ─────────────────────────────────────────────────
    section("Applying LoRA Adapters (PEFT)")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    lora_model  = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    lora_model  = get_peft_model(lora_model, lora_config)

    # ── Parameter count comparison ───────────────────────────────────────
    section("Parameter Count Comparison")
    full_trainable, full_total = count_params(
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
        )
    )
    lora_trainable, lora_total = count_params(lora_model)

    print(f"  {'':30} {'Trainable':>15} {'Total':>15} {'Efficiency':>12}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*12}")
    print(f"  {'Full Fine-Tune':<30} {full_trainable:>15,} {full_total:>15,} {'100.0%':>12}")
    pct = lora_trainable / full_total * 100
    print(f"  {'LoRA Fine-Tune':<30} {lora_trainable:>15,} {lora_total:>15,} {pct:>11.2f}%")
    print(f"\n  LoRA trains only {pct:.2f}% of parameters – dramatically reducing")
    print(f"  memory, compute, and storage overhead.")
    lora_model.print_trainable_parameters()

    # ── Fine-tune ────────────────────────────────────────────────────────
    section(f"Fine-Tuning with LoRA ({NUM_EPOCHS} epochs)")
    lora_trainer = make_trainer(lora_model, tokenized, tokenizer, "./results_lora")

    t0 = time.time()
    lora_trainer.train()
    train_time = time.time() - t0
    print(f"\n  Training time : {train_time:.1f}s  ({train_time/60:.1f} min)")

    # ── Evaluate after fine-tune ─────────────────────────────────────────
    section("Post Fine-Tune Evaluation")
    lora_results  = lora_trainer.evaluate()
    lora_acc      = lora_results.get("eval_accuracy", 0.0)

    print(f"\n  {'Metric':<30} {'Baseline':>12} {'LoRA FT':>12} {'Improvement':>14}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*14}")
    improvement = lora_acc - baseline_acc
    print(f"  {'Accuracy':<30} {baseline_acc:>12.4f} {lora_acc:>12.4f} {improvement:>+14.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    section("Summary & Takeaways")
    print(f"""
  Model           : {MODEL_NAME}
  Dataset         : SST-2 (subset {TRAIN_SAMPLES} train / {EVAL_SAMPLES} val)
  Epochs          : {NUM_EPOCHS}
  LoRA rank (r)   : {LORA_R}
  Target modules  : {LORA_TARGETS}

  Trainable params with LoRA : {lora_trainable:,} ({pct:.2f}% of full model)
  Training time              : {train_time:.1f}s

  Takeaways:
  • LoRA inserts tiny rank-decomposition matrices (A · B, rank={LORA_R}) into
    the attention Q and V projections. Only A and B are trained; the original
    weights are frozen, saving GPU memory and making adapters tiny to store.
  • Despite training <{int(pct)+1}% of parameters, LoRA achieves substantial
    accuracy gain over the baseline random initialisation.
  • For production: multiple LoRA adapters can be swapped onto the same
    frozen base model for different tasks — very storage efficient.
""")

    # ── Save adapter weights ─────────────────────────────────────────────
    lora_model.save_pretrained("./lora_adapter")
    print(f"  LoRA adapter weights saved to ./lora_adapter/")
    print(f"  (These are tiny files – only the adapter parameters are stored!)")


if __name__ == "__main__":
    main()
