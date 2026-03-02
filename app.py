import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

# Login using HF token stored in Space secrets
login(os.environ["HF_TOKEN"])

model_name = "google/gemma-2-2b-it"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def chat(user_message, history):
    if history is None:
        history = []

    # Build conversation history
    messages = []

    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

    messages.append({"role": "user", "content": user_message})

    # Apply Gemma chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )

    # Proper token-level slicing (removes prompt completely)
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    history.append((user_message, response))
    return history, history


with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Nyx - AI Chatbot (Powered by Gemma 2B)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message")
    clear = gr.Button("Clear Chat")

    msg.submit(chat, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()