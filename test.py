from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Define the model path
model_path = "/urPath/RAG/model"

# Load the tokenizer and model
print("Loading the model...")
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)

print("Model loaded successfully!")

# Chat loop
def chat_with_model():
    print("You can now chat with the model! Type 'exit' to quit.")
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        # Tokenize input and generate response
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1, temperature=0.7)

        # Decode and print the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nModel:", response)

# Run the chat loop
if __name__ == "__main__":
    chat_with_model()