import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # You can change this
DATA_PATH = "data/reddit_empathy.jsonl"
SAVE_DIR = "models/lora/"

# 1. Load dataset from JSONL
def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    return Dataset.from_list(samples)

# 2. Format input/output for instruction tuning
def format_example(example):
    joined_context = "\n".join(example["context"])
    formatted = {
        "text": f"### Input:\n{joined_context}\n\n### Response:\n{example['response']}\n\n### Labels:\n{', '.join(example['labels'])}"
    }
    return formatted

# 3. Tokenize
def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

def main():
    print("📦 Loading dataset...")
    raw_dataset = load_json_dataset(DATA_PATH)
    formatted_dataset = raw_dataset.map(format_example)
    
    print("🔠 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Prepare model for LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    print("🧪 Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(lambda e: tokenize_function(e, tokenizer), remove_columns=formatted_dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("🛠️ Starting training...")
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    print(f"✅ Saving LoRA adapter to {SAVE_DIR}")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

if __name__ == "__main__":
    main()
