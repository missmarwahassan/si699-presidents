import pandas as pd
import spacy
from tqdm import tqdm
from spacy.matcher import PhraseMatcher
from presidential_speeches_full import PRESIDENTIAL_SPEECHES

# -----------------------------
# Load spaCy (FAST config)
# -----------------------------
print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])

# -----------------------------
# US states
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

abbr_to_state = {v: k for k, v in us_states.items()}

# -----------------------------
# PhraseMatcher (VERY FAST)
# -----------------------------
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

patterns = []

for state in us_states:
    patterns.append(nlp.make_doc(state))

for abbr in us_states.values():
    patterns.append(nlp.make_doc(abbr))

matcher.add("US_STATES", patterns)

# -----------------------------
# Disambiguation logic
# -----------------------------
def resolve_state(span, doc, person_spans):
    text = span.text
    text_lower = text.lower()

    # 🚫 skip PERSON entities (e.g., George Washington)
    if (span.start, span.end) in person_spans:
        return None

    # direct matches (fast path)
    if text in us_states:
        return text
    if text in abbr_to_state:
        return abbr_to_state[text]

    # context window
    window = doc[max(span.start-4, 0): span.end+4].text.lower()

    # -------------------------
    # Washington disambiguation
    # -------------------------
    if text_lower == "washington":

        # DC
        if "dc" in window or "d.c" in window:
            return None

        # Person cues
        if any(x in window for x in ["george", "president", "general", "mr.", "washington's"]):
            return None

        # State cues
        if any(x in window for x in ["state", "seattle", "olympia"]):
            return "Washington"

        return "Washington"  # default bias

    # -------------------------
    # Georgia disambiguation
    # -------------------------
    if text_lower == "georgia":
        if any(x in window for x in ["russia", "europe", "soviet"]):
            return None
        return "Georgia"

    # -------------------------
    # New York (state vs city)
    # -------------------------
    if text_lower == "new york":
        return "New York"

    return None

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")
df = PRESIDENTIAL_SPEECHES.copy()

# filter presidents
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

# dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year

# -----------------------------
# Processing loop (OPTIMIZED)
# -----------------------------
print("Processing speeches...")

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    doc = nlp(str(row["text"]))

    # precompute PERSON spans (fast lookup)
    person_spans = {(ent.start, ent.end) for ent in doc.ents if ent.label_ == "PERSON"}

    seen = set()

    # -------------------------
    # 1. PhraseMatcher (primary)
    # -------------------------
    for _, start, end in matcher(doc):
        if (start, end) in seen:
            continue
        seen.add((start, end))

        span = doc[start:end]
        state = resolve_state(span, doc, person_spans)

        if state:
            results.append({
                "state": state,
                "abbr": us_states[state],
                "president": row["president"],
                "year": row["year"]
            })

    # -------------------------
    # 2. NER fallback
    # -------------------------
    for ent in doc.ents:
        if ent.label_ == "GPE":
            span_key = (ent.start, ent.end)
            if span_key in seen:
                continue

            state = resolve_state(ent, doc, person_spans)

            if state:
                results.append({
                    "state": state,
                    "abbr": us_states[state],
                    "president": row["president"],
                    "year": row["year"]
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