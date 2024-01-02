import gradio as gr
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForCausalLM
import argparse
model_cache = {}
conversations = {}

def load_model(model_name):
    if model_name not in model_cache:
        model_cache[model_name] = pipeline("conversational", model=model_name)
    conversations[model_name] = Conversation()
    return "Model loaded. You can now start chatting."

def chat_with_model(model_name, user_input, temperature, max_length):
    if model_name in model_cache:
        chat = model_cache[model_name]
        conversation = conversations[model_name]
        conversation.add_user_input(user_input)
        response = chat(conversation, temperature=temperature, max_length=max_length)
        return str(conversation)
    else:
        return "Model is still loading. Please wait."

def main(share=False):
    with gr.Blocks() as demo:
        with gr.Row():
            model_selector = gr.Dropdown(
                ["microsoft/DialoGPT-medium", "gpt2", "mistralai/Mixtral-8x7B-v0.1"],
                label="Model"
            )
            model_selector.change(
                fn=load_model,
                inputs=model_selector,
                outputs=[]
            )
            # default slider value is 0.7
            temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="Temperature", value=0.7)
            max_length = gr.Slider(minimum=5, maximum=100, step=1, label="Max Length", value=20)
        with gr.Row():
            # named textbox
            chat_output = gr.Textbox(interactive=False, lines=10, placeholder="Chat will appear here...", label="Chat")
        with gr.Row():
            user_input = gr.Textbox(lines=2, placeholder="Enter your message here...", scale=4)
            submit_button = gr.Button("Send")

        def update_chat(model_name, user_input, temperature, max_length):
            new_chat = chat_with_model(model_name, user_input, temperature, max_length)
            return new_chat, ""  # Return new chat content and clear the input box

        submit_button.click(
            fn=update_chat,
            inputs=[model_selector, user_input, temperature, max_length],
            outputs=[chat_output, user_input]
        )

    demo.launch(share=True)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--share", action="store_true")
    args = arg_parser.parse_args()
    main(share=args.share)
