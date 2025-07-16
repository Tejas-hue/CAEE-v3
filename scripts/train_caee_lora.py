import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig

# MODEL NAME
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Adjust depending on architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
data_path = "data/reddit_empathy.jsonl"
with open(data_path, "r") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)

# Tokenization function
def tokenize(batch):
    prompt = ""
    for turn in batch["context"]:
        prompt += turn.strip() + "\n"
    prompt += f"Response: {batch['response']}"
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return {
        "input_ids": input_ids["input_ids"][0],
        "attention_mask": input_ids["attention_mask"][0],
        "labels": input_ids["input_ids"][0],
    }

tokenized_dataset = dataset.map(tokenize)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="outputs/caee-v3-mistral",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_dir="logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    bf16=False,  # Colab GPUs usually support float16, not bf16
    fp16=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save model
trainer.save_model("outputs/caee-v3-mistral")
tokenizer.save_pretrained("outputs/caee-v3-mistral")
