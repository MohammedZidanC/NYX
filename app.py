import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

# Authenticate (HF_TOKEN must be in Space secrets)
login(os.environ["HF_TOKEN"])

model_name = "google/gemma-2-2b-it"

# Load tokenizer and model (GPU auto if available)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def chat(user_message, history):
    if history is None:
        history = []

    # Create structured chat messages
    messages = [
        {"role": "system", "content": "You are Nyx, an intelligent AI assistant. Answer clearly and directly."},
        {"role": "user", "content": user_message}
    ]

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

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Extract only model reply
    response = response.split("model")[-1].strip()

    history.append((user_message, response))
    return history, history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Nyx - AI Chatbot (Powered by Gemma 2B)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message")
    clear = gr.Button("Clear Chat")

    msg.submit(chat, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()