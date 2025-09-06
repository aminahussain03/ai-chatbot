from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the chatbot name (optional)
BOT_NAME = "AI Bot"

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

print(f"\n{BOT_NAME} is ready to chat! Type 'quit' to exit.\n")

chat_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print(f"{BOT_NAME}: Goodbye!")
        break

    prompt = chat_history + f"You: {user_input}\n{BOT_NAME}:"
    response = generate_response(prompt)
    print(f"{BOT_NAME}: {response}")

    chat_history += f"You: {user_input}\n{BOT_NAME}: {response}\n"
