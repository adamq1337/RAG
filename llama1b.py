
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"
cache_dir = "/urPath/RAG/model"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir,)

# Test the model
#prompt = "What is the capital of France?"
#inputs = tokenizer(prompt, return_tensors="pt")
#outputs = model.generate(**inputs, max_new_tokens=50)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))

import pandas as pd

# Load the metadata
metadata_path = "/UrPath/RAG/metadata.csv"
metadata = pd.read_csv(metadata_path)


from sentence_transformers import SentenceTransformer

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight embedding model

# Generate embeddings for the summaries
metadata["embeddings"] = metadata["Brief Summary"].apply(lambda x: embedding_model.encode(str(x), show_progress_bar=False))


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_relevant_trials(question, metadata, top_k=3):
    question_embedding = embedding_model.encode(question)
    metadata["similarity"] = metadata["embeddings"].apply(lambda x: cosine_similarity([question_embedding], [x])[0][0])
    return metadata.nlargest(top_k, "similarity")

def generate_response(question, model, tokenizer, metadata):
    # Find relevant trials
    relevant_trials = find_relevant_trials(question, metadata)

    # Format the trial information
    citations = "\n\n".join(
        f"NCT Number: {row['NCT Number']}\nStudy Title: {row['Study Title']}\nStudy URL: {row['Study URL']}"
        for _, row in relevant_trials.iterrows()
    )

    # Create the prompt
    prompt = (
        f"Answer the question based on the following clinical trials:\n"
        f"{citations}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # Generate the response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response
while True:
    question = input("Ask a question about cardiology (or 'exit' to quit): ")
    if question.lower() == "exit":
        break
    
    response = generate_response(question, model, tokenizer, metadata)
    print("\nResponse:")
    print(response)

