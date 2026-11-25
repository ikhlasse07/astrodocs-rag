from time import perf_counter

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Loading the embedding model...", flush=True)

start = perf_counter()
model = SentenceTransformer(MODEL_NAME, device="cpu")

print(f"Model loaded in {perf_counter() - start:.2f} seconds")
print("Maximum input length:", model.max_seq_length, "tokens")

passages = [
    "The Sun produces energy through nuclear fusion in its core.",
    "Mars has two small moons named Phobos and Deimos.",
]

question = "How does the Sun produce energy?"

passage_vectors = model.encode(
    passages,
    normalize_embeddings=True,
)

question_vector = model.encode(
    [question],
    normalize_embeddings=True,
)

scores = (question_vector @ passage_vectors.T)[0]

print("\nEmbedding shape:", passage_vectors.shape)
print("\nQuestion:", question)

for passage, score in zip(passages, scores):
    print(f"\nSimilarity: {float(score):.4f}")
    print("Passage:", passage)

best_index = int(scores.argmax())
print("\nBest matching passage:", passages[best_index])
