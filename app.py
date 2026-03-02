import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Disable gradients (important for memory)
torch.set_grad_enabled(False)

model_name = "google/gemma-2-2b-it"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in reduced memory mode
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)

model.eval()

def chat(user_message, history):
    if history is None:
        history = []

    messages = []

    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,   # reduce generation length
            do_sample=False
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    history.append((user_message, response))
    return history, history


with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Nyx - AI Chatbot (Gemma 2-2B-it)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message")
    clear = gr.Button("Clear Chat")

    msg.submit(chat, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()