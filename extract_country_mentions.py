import pandas as pd
import spacy
import pycountry
from tqdm import tqdm
from presidential_speeches_full import PRESIDENTIAL_SPEECHES

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading dataset...")
df = PRESIDENTIAL_SPEECHES.copy()

# print unique presidents and years for debugging
print("Unique presidents:", df["president"].unique())

valid_presidents = [
    "Donald J. Trump (2nd Term)",
    "Joseph R. Biden, Jr.",
    "Donald J. Trump (1st Term)",
    "George W. Bush",
    "Barack Obama",
    "William J. Clinton",
    "George Bush",
    "Ronald Reagan",
    "Jimmy Carter",
    "Gerald R. Ford",
    "Richard Nixon",
    "Lyndon B. Johnson",
    "John F. Kennedy",
    "Dwight D. Eisenhower",
    "Harry S. Truman",
    "Franklin D. Roosevelt",
    "Herbert Hoover",
    "Calvin Coolidge",
    "Warren G. Harding",
    "Woodrow Wilson",
    "Theodore Roosevelt",
    "William Howard Taft",
    "William McKinley",
    "Grover Cleveland",
    "Benjamin Harrison",
    "James A. Garfield",
    "Rutherford B. Hayes",
    "Ulysses S. Grant",
    "Chester A. Arthur",
    "Andrew Johnson",
    "Abraham Lincoln",
    "James Buchanan",
    "Franklin Pierce",
    "Zachary Taylor",
    "John Tyler",
    "James K. Polk",
    "Martin van Buren",
    "William Henry Harrison",
    "Andrew Jackson",
    "John Quincy Adams",
    "James Monroe",
    "James Madison",
    "Thomas Jefferson",
    "John Adams",
    "George Washington",
]

df = df[df["president"].isin(valid_presidents)]

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
            results.append({"country": ent.text, "president": president, "year": year})

country_mentions = pd.DataFrame(results)

print("Aggregating counts...")

counts = (
    country_mentions.groupby(["country", "president", "year"])
    .size()
    .reset_index(name="mentions")
)

print("Saving results...")

counts.to_csv("country_mentions_counts.csv", index=False)

print("Done!")
