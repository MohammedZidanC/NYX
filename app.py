import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import os

# Step 1: Authenticate using the Hugging Face token stored as a Space secret
login(os.environ["HF_TOKEN"])  #added HF_TOKEN in your Space secrets

# Step 2: Load Gemma-2-2b-it model
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Step 3: Chat function
def chat(user_message, history=None):
    if history is None:
        history = []

    # Simple greeting handling
    greetings = ["hi", "hello", "hey", "hola"]
    if user_message.lower() in greetings:
        response = "NYX: Hello! How are you?"
    else:
        # Generate a response from the model
        result = generator(
            f"You are NYX, a helpful chatbot. Answer clearly and concisely: {user_message}",
            max_length=200,
            do_sample=False
        )[0]['generated_text']
        response = f"NYX: {result}"

    history.append((user_message, response))
    return history, history

# Step 4: Gradio Interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg_box = gr.Textbox(label="Type your message")
    clear = gr.Button("Clear Chat")

    # Submit message
    msg_box.submit(chat, [msg_box, chatbot], [chatbot, chatbot])

    # Clear chat button
    clear.click(lambda: [], None, chatbot)

demo.launch()