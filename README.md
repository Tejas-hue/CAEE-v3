# CAEE: Context-Aware Empathic Engine (Mistral 7B Version)

This repository contains the **Mistral 7B version** of **CAEE (Context-Aware Empathic Engine)** — a chatbot designed to engage in emotionally intelligent, context-sensitive support conversations. It leverages **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.

---

## Status: Work in Progress

Due to compute limitations, this version remains **untested** and **incomplete**.  
The model has been **fine-tuned on Mistral-7B**, but I have **not retrained the LoRA adapter on smaller base models**. This repository exists to document and archive:

- A working Gradio demo interface (`app.py`)
- Preliminary model loading logic
- Architecture compatibility experiments
- The original LoRA weights (trained on Mistral-7B)

---

## Capabilities

This version of CAEE aims to:

-Detect emotional shifts in conversations

-Respond with empathetic, supportive language modeled after peer-support discourse

-Maintain context across multiple user turns

-Begin exploring intent sensitivity (e.g., differentiating venting vs. advice-seeking)

-Avoid superficial sentiment repetition by using deeper context

>⚠️ Note: Some of these capabilities are aspirational in this version and depend on future fine-tuning and testing.

---

## What's Included

- `app.py`: Gradio interface for interacting with the chatbot.
- `adapter_config.json` and `adapter_model.bin`: LoRA adapter trained on Mistral 7B.
- `requirements.txt`: Python package dependencies.
- `README.md`: This documentation.

---

## Training Details

- **Base model used**: Mistral-7B  
- **Fine-tuning method**: LoRA via Hugging Face PEFT  
- **Training data**: A paraphrased version of short, anonymized peer-support-style conversations based on the *TalkLife empathy dataset* (see citation below).  
- **Objective**: Help the model learn how to engage in emotionally supportive, validating, and context-aware interactions.

Note: While the LoRA adapter was trained on Mistral-7B, it has **not yet been tested** or validated due to compute limitations.

---

Citation
If this code or dataset contributes to your research, please cite:

@inproceedings{sharma2020empathy,
  title={A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support},
  author={Sharma, Ashish and Miner, Adam S and Atkins, David C and Althoff, Tim},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}


## How to Run Locally

```bash
git clone https://github.com/your-username/caee-mistral7b.git
cd caee-mistral7b
pip install -r requirements.txt
python app.py

---


