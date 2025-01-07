import pandas as pd

# Load the CSV file
data = pd.read_csv('ctg-studies.csv')

# Inspect the data
print(data.head())

data.drop_duplicates(subset='NCT Number', inplace=True)
data.fillna('N/A', inplace=True)

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Combine fields for embeddings
data['combined_text'] = data['Study Title'] + ' ' + data['Conditions'] + ' ' + data['Interventions']

# Generate embeddings
embeddings = model.encode(data['combined_text'].tolist())

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity
index.add(np.array(embeddings))

# Save FAISS index for reuse
faiss.write_index(index, 'ctg-studies.index')

data.to_csv('metadata.csv', index=False)

