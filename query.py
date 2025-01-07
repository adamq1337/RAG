from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"
cache_dir = "/urPath/RAG/model"

tokenizer = AutoTokenizer.from_pretrained(cache_dir)
model = AutoModelForCausalLM.from_pretrained(cache_dir)

def generate_response(prompt, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, do_sample=True, top_k=50, top_p=0.95, num_beams=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

prompt = "What is the capital of France?"
response = generate_response(prompt)
print(response)