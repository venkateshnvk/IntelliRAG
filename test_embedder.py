from embeddings.embedder import Embedder

embedder = Embedder()

text = "Patient has atrial fibrillation."

vector = embedder.generate_embedding(text)

print("Embedding length:", len(vector))