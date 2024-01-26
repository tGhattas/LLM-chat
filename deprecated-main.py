import argparse
from huggingface_hub import InferenceClient
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"


def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt


def generate(
        prompt, history, temperature=0.2, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0
):
    global model_id, is_offline
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)
    if is_offline:
        # offline mode
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
        output = model.generate(input_ids, **generate_kwargs)
        yield tokenizer.decode(output[0], skip_special_tokens=True)

    client = InferenceClient(model_id)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True,
                                    return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output


def main(share=False):
    mychatbot = gr.Chatbot(
        # avatar_images=["./user.png", "./botm.png"],
        bubble_full_width=False, show_label=False, show_copy_button=True,
        likeable=True, )

    demo = gr.ChatInterface(fn=generate,
                            chatbot=mychatbot,
                            title=f"Vanilla Chat - {model_id}",
                            retry_btn=None,
                            undo_btn=None
                            )

    demo.queue().launch(show_api=False, share=share)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--share", action="store_true")
    arg_parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    arg_parser.add_argument("--offline", action="store_true")

    args = arg_parser.parse_args()
    model_id = args.model_id
    is_offline = args.offline
    main(share=args.share)
