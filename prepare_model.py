from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_path = "."

model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_path)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged-mistral-lora")
