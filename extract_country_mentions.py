import pandas as pd
import spacy
import pycountry
from tqdm import tqdm

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading dataset...")
df = pd.read_csv("presidential_speeches_full.csv")

# filter only Donald Trump speeches (for class demo)
df = df[df["president"].str.contains("Donald J. Trump", na=False)]

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year

valid_countries = {c.name for c in pycountry.countries}

results = []

print("Processing speeches...")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing speeches"):

    text = str(row["text"])
    president = row["president"]
    year = row["year"]

    doc = nlp(text)

    for ent in doc.ents:

        if ent.label_ == "GPE" and ent.text in valid_countries:

            results.append({
                "country": ent.text,
                "president": president,
                "year": year
            })

country_mentions = pd.DataFrame(results)

print("Aggregating counts...")

counts = (
    country_mentions
    .groupby(["country","president","year"])
    .size()
    .reset_index(name="mentions")
)

print("Saving results...")

counts.to_csv("country_mentions_counts.csv", index=False)

print("Done!")