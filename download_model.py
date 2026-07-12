# To download the fastembed model
from fastembed import TextEmbedding

print("Downloading model...")
model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir="./fastembed_cache"
)
print("Done! Model cached in ./fastembed_cache")

# Quick sanity check
test = list(model.embed(["hello world"]))
print(f"Embedding dimension: {len(test[0])}")