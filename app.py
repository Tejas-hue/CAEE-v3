import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

#  Load base + LoRA adapter
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "."

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

#  Response generator
def generate_response(prompt, max_tokens=200, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI design
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("# Empathic AI Chatbot (LoRA Fine-tuned Mistral 7B)")
    gr.Markdown("Fine-tuned to respond with emotional intelligence and support.")

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Your Prompt", lines=4, placeholder="What's on your mind?")
            max_tokens = gr.Slider(50, 512, value=200, label="Max Tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Response", lines=10)

    generate_btn = gr.Button("Generate Response")
    generate_btn.click(
        generate_response,
        inputs=[prompt, max_tokens, temperature, top_p],
        outputs=output
    )

demo.launch()
