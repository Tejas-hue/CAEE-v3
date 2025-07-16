import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch
import os

# === Configs ===
MODEL_NAME = "google/gemma-2b-it"
JSONL_PATH = "data/reddit_empathy.jsonl"
MAX_LENGTH = 512
USE_LORA = True
BATCH_SIZE = 4
EPOCHS = 3
OUTPUT_DIR = "models/caee-gemma2b"

# === Load and format dataset ===
def load_data(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f]

    # Convert to HF dataset
    dataset = Dataset.from_list(samples)
    return dataset

# === Prompt Template ===
def format_example(example):
    prompt = "\n".join(example["context"]) + "\nFriend:"
    response = example["response"]
    labels = example["labels"]
    return {
        "text": prompt,
        "response": response,
        "labels": labels
    }

# === Tokenize ===
def tokenize_function(example, tokenizer, label2id):
    formatted = format_example(example)
    prompt = formatted["text"]
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]

    # Convert labels to binary vector
    labels = [0] * len(label2id)
    for lbl in formatted["labels"]:
        if lbl in label2id:
            labels[label2id[lbl]] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.float)
    }

# === Label list (16 needs)
LABELS = [
    "acknowledgment", "celebration", "clarity", "comfort", "connection",
    "empathy", "encouragement", "guidance", "motivation", "neutral",
    "reassurance", "safety", "support", "understanding", "validation", "companionship"
]
label2id = {label: i for i, label in enumerate(LABELS)}

# === Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")

if USE_LORA:
    base_model = prepare_model_for_kbit_training(base_model)
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    base_model = get_peft_model(base_model, config)

# === Load and tokenize dataset
dataset = load_data(JSONL_PATH)
tokenized_dataset = dataset.map(lambda ex: tokenize_function(ex, tokenizer, label2id), remove_columns=dataset.column_names)

# === Training args
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    output_dir=OUTPUT_DIR,
    save_total_limit=1,
    evaluation_strategy="no",
    logging_dir=f"{OUTPUT_DIR}/logs",
    fp16=True,
    report_to="none"
)

# === Custom Trainer with sigmoid loss
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :len(label2id)]  # only last token logits for multi-label
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss

# === Train!
trainer = MultiLabelTrainer(
    model=base_model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
base_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete.")
