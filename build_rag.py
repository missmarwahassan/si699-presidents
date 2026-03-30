import pandas as pd
from tqdm import tqdm  # <-- progress bar
from presidential_speeches_full import PRESIDENTIAL_SPEECHES

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")
df = PRESIDENTIAL_SPEECHES.copy()

# -----------------------------
# Chunking function
# -----------------------------
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

documents = []

# -----------------------------
# Chunk speeches with progress bar
# -----------------------------
print("Chunking speeches...")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing speeches"):
    text = str(row["text"])
    president = row["president"]
    date = row["date"]
    title = row["title"]

    chunks = chunk_text(text, chunk_size=400)

    for chunk in chunks:
        documents.append({
            "text": chunk,
            "president": president,
            "date": date,
            "title": title
        })

print(f"Total chunks created: {len(documents)}")


pd.DataFrame(documents).to_pickle("documents.pkl")
print("Done!")