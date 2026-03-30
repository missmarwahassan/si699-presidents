import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Presidential Speech Analysis",
    page_icon="🇺🇸",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #1f2230;
}
div[data-testid="stMetric"] {
    background-color: #151925;
    border: 1px solid #2c3143;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)
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
st.sidebar.markdown("## 🎛️ Filters")
st.sidebar.caption("Explore how U.S. presidents referenced countries and states over time.")

presidents = sorted(country_df["president"].dropna().unique())

selected_president = st.sidebar.selectbox(
    "Select President",
    ["All Presidents"] + presidents
)

year_range = st.sidebar.slider(
    "Year Range",
    int(country_df["year"].min()),
    int(country_df["year"].max()),
    (int(country_df["year"].min()), int(country_df["year"].max()))
)

st.sidebar.markdown("---")
st.sidebar.caption("SI 699 • Presidential rhetoric dashboard")

def style_bar_chart(fig, title):
    fig.update_layout(
        title=title,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        title_font_size=22,
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False
    )
    return fig

def style_map(fig, title):
    fig.update_layout(
        title=title,
        paper_bgcolor="#0e1117",
        font_color="white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=520
    )
    return fig
# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🌍 Country Analysis",
    "🇺🇸 State Analysis",
    "🤖 AI Insights"
])
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
# COUNTRY TAB
# -----------------------------
with tab1:
    st.header("🌍 Country Mentions")

    filtered = apply_filters(country_df)
    country_counts = filtered.groupby("country", as_index=False)["mentions"].sum()

    total_mentions = int(country_counts["mentions"].sum()) if not country_counts.empty else 0
    unique_countries = int(country_counts["country"].nunique()) if not country_counts.empty else 0
    top_country = (
        country_counts.sort_values("mentions", ascending=False).iloc[0]["country"]
        if not country_counts.empty else "N/A"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Mentions", f"{total_mentions:,}")
    c2.metric("Countries Mentioned", unique_countries)
    c3.metric("Most Mentioned Country", top_country)

    fig_map = px.choropleth(
        country_counts,
        locations="country",
        locationmode="country names",
        color="mentions",
        color_continuous_scale="Viridis"
    )
    fig_map = style_map(fig_map, "Global Country Mentions")

    map_col1, map_col2, map_col3 = st.columns([1, 6, 1])
    with map_col2:
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Top Countries")
    top_countries = country_counts.sort_values("mentions", ascending=False).head(15)
    top_countries = top_countries.sort_values("mentions", ascending=True)

    fig_bar = px.bar(
        top_countries,
        x="mentions",
        y="country",
        orientation="h",
        color="mentions",
        color_continuous_scale="Tealgrn"
    )
    fig_bar = style_bar_chart(fig_bar, "Top 15 Countries Mentioned")
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# STATE TAB
# -----------------------------
with tab2:
    st.header("🇺🇸 State Mentions")

    filtered = apply_filters(state_df)
    state_counts = filtered.groupby("state", as_index=False)["mentions"].sum()

    total_mentions = int(state_counts["mentions"].sum()) if not state_counts.empty else 0
    unique_states = int(state_counts["state"].nunique()) if not state_counts.empty else 0
    top_state = (
        state_counts.sort_values("mentions", ascending=False).iloc[0]["state"]
        if not state_counts.empty else "N/A"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Mentions", f"{total_mentions:,}")
    c2.metric("States Mentioned", unique_states)
    c3.metric("Most Mentioned State", top_state)

    fig_map = px.choropleth(
        state_counts,
        locations="state",
        locationmode="USA-states",
        color="mentions",
        scope="usa",
        color_continuous_scale="Blues"
    )
    fig_map = style_map(fig_map, "U.S. State Mentions")
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Top States")
    top_states = state_counts.sort_values("mentions", ascending=False).head(15)
    top_states = top_states.sort_values("mentions", ascending=True)

    fig_bar = px.bar(
        top_states,
        x="mentions",
        y="state",
        orientation="h",
        color="mentions",
        color_continuous_scale="Blues"
    )
    fig_bar = style_bar_chart(fig_bar, "Top 15 States Mentioned")
    st.plotly_chart(fig_bar, use_container_width=True)
# -----------------------------
# AI INSIGHTS TAB
# -----------------------------
with tab3:
    st.header("🤖 Ask Questions About Presidential Speeches")
    st.caption("Use retrieval + AI summarization to compare how presidents discussed a topic.")

    if documents is None or vectorizer is None:
        st.warning("AI Insights are unavailable because documents.pkl is missing.")
    else:
        query = st.text_input("Enter your question", placeholder="Example: How did presidents talk about war over time?")

        pres_options = ["All Presidents"] + sorted(documents["president"].dropna().unique().tolist())
        pres_filter = st.selectbox("Filter by president (optional)", pres_options)

        if st.button("Generate Answer", use_container_width=True):
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing speeches..."):
                    chosen_president = None if pres_filter == "All Presidents" else pres_filter
                    docs = retrieve_docs(query, chosen_president, k=5)

                    if not docs:
                        st.warning("No relevant speeches found.")
                    else:
                        context = "\n\n".join(
                            f"{d['president']} ({d['date']}): {d['text']}"
                            for d in docs
                        )
                        answer = generate_answer(context, query)

                        st.subheader("Answer")
                        st.info(answer)

                        st.subheader("Source Chunks")
                        for i, d in enumerate(docs, start=1):
                            with st.expander(f"Source {i}: {d['president']} ({d['date']})"):
                                st.write(d["text"])