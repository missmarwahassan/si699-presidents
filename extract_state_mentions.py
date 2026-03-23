import pandas as pd
import spacy
from tqdm import tqdm
from presidential_speeches_full import PRESIDENTIAL_SPEECHES

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading dataset...")
df = PRESIDENTIAL_SPEECHES.copy()

# -----------------------------
# Filter to valid presidents
# -----------------------------
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

# -----------------------------
# Date processing
# -----------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year

# -----------------------------
# US states dictionary
# -----------------------------
us_states = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
    "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
    "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
    "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO",
    "Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
    "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND",
    "Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI",
    "South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT",
    "Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV",
    "Wisconsin":"WI","Wyoming":"WY"
}

state_abbrevs = set(us_states.values())

results = []

print("Processing speeches...")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing speeches"):

    text = str(row["text"])
    president = row["president"]
    year = row["year"]

    doc = nlp(text)

    for ent in doc.ents:

        if ent.label_ == "GPE":

            # full state name
            if ent.text in us_states:
                state = ent.text
                abbr = us_states[state]

            # abbreviation (e.g., CA, TX)
            elif ent.text in state_abbrevs:
                abbr = ent.text
                state = [k for k, v in us_states.items() if v == abbr][0]

            else:
                continue

            results.append({
                "state": state,
                "abbr": abbr,
                "president": president,
                "year": year
            })

# -----------------------------
# Aggregate
# -----------------------------
print("Aggregating...")

counts = (
    pd.DataFrame(results)
    .groupby(["state", "abbr", "president", "year"])
    .size()
    .reset_index(name="mentions")
)

# -----------------------------
# Save
# -----------------------------
counts.to_csv("state_mentions_counts.csv", index=False)

print("Done!")