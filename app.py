import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("🇺🇸 Presidential Speech Analysis + AI Insights")

# -----------------------------
# OPENAI CLIENT (NEW API)
# -----------------------------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_documents():
    return pd.read_pickle("documents.pkl")

@st.cache_data
def load_country_data():
    return pd.read_csv("country_mentions_counts.csv")

@st.cache_data
def load_state_data():
    return pd.read_csv("state_mentions_counts.csv")

documents = load_documents()
country_df = load_country_data()
state_df = load_state_data()

# -----------------------------
# BUILD TF-IDF
# -----------------------------
@st.cache_data
def build_vectorizer(documents):
    texts = documents["text"].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

vectorizer, tfidf_matrix = build_vectorizer(documents)

# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------
def retrieve_docs(query, president=None, k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = scores.argsort()[-k:][::-1]

    results = []
    for i in top_indices:
        row = documents.iloc[i]

        if president:
            if president.lower() not in row["president"].lower():
                continue

        results.append({
            "text": row["text"][:200],  # keep cheap
            "president": row["president"],
            "date": row["date"]
        })

    return results

# -----------------------------
# GPT CALL (UPDATED API)
# -----------------------------
@st.cache_data(show_spinner=False)
def generate_answer(context, question):

    prompt = f"""
Context:
{context}

Question:
{question}

Analyze how presidents discuss this topic.
Highlight tone, priorities, and key differences.
Do NOT copy sentences directly.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a political analyst specializing in U.S. presidential speeches."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300
    )

    return response.choices[0].message.content

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

presidents = sorted(country_df["president"].unique())

selected_president = st.sidebar.selectbox(
    "Select President",
    ["All Presidents"] + presidents
)

year_range = st.sidebar.slider(
    "Year Range",
    int(country_df.year.min()),
    int(country_df.year.max()),
    (int(country_df.year.min()), int(country_df.year.max()))
)

def apply_filters(df):
    filtered = df.copy()

    if selected_president != "All Presidents":
        filtered = filtered[filtered["president"] == selected_president]

    filtered = filtered[
        (filtered["year"] >= year_range[0]) &
        (filtered["year"] <= year_range[1])
    ]

    return filtered

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🌍 Country Analysis",
    "🇺🇸 State Analysis",
    "🤖 AI Insights"
])

# -----------------------------
# COUNTRY TAB
# -----------------------------
with tab1:
    st.header("Country Mentions")

    filtered = apply_filters(country_df)
    country_counts = filtered.groupby("country")["mentions"].sum().reset_index()

    fig_map = px.choropleth(
        country_counts,
        locations="country",
        locationmode="country names",
        color="mentions",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Top Countries")
    top_countries = country_counts.sort_values("mentions", ascending=False).head(15)

    fig_bar = px.bar(
        top_countries,
        x="mentions",
        y="country",
        orientation="h"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# STATE TAB
# -----------------------------
with tab2:
    st.header("State Mentions")

    filtered = apply_filters(state_df)
    state_counts = filtered.groupby("state")["mentions"].sum().reset_index()

    fig_map = px.choropleth(
        state_counts,
        locations="state",
        locationmode="USA-states",
        color="mentions",
        scope="usa",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Top States")
    top_states = state_counts.sort_values("mentions", ascending=False).head(15)

    fig_bar = px.bar(
        top_states,
        x="mentions",
        y="state",
        orientation="h"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# AI INSIGHTS TAB
# -----------------------------
with tab3:
    st.header("Ask Questions About Presidential Speeches")

    query = st.text_input("Enter your question")

    pres_filter = st.text_input("Filter by president (optional)")

    if query:
        with st.spinner("Analyzing speeches..."):

            docs = retrieve_docs(query, pres_filter if pres_filter else None, k=5)

            if not docs:
                st.warning("No relevant speeches found.")
            else:
                context = "\n\n".join([
                    f"{d['president']} ({d['date']}): {d['text']}"
                    for d in docs
                ])

                answer = generate_answer(context, query)

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source Chunks")
                for d in docs:
                    st.write(f"**{d['president']} ({d['date']})**")
                    st.write(d["text"] + "...")