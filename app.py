import streamlit as st
import pandas as pd
import plotly.express as px
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from presidential_speeches_full import DOCUMENTS

from openai import OpenAI

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Presidential Speech Analysis", page_icon="🇺🇸", layout="wide"
)

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)
# -----------------------------
# OPENAI CLIENT (NEW API)
# -----------------------------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_documents():
    if os.path.exists("documents.pkl"):
        return pd.read_pickle("documents.pkl")
    else:
        return DOCUMENTS.copy()


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

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


vectorizer, tfidf_matrix = build_vectorizer(documents)


# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------
def retrieve_docs(query=None, location=None, president=None, k=5, diverse=True):
    results = []

    if query:
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        df = documents.copy()
        df["score"] = scores

        # Optional president filter
        if president:
            df = df[df["president"].str.contains(president, case=False, na=False)]

        # Sort by relevance
        df = df.sort_values(by="score", ascending=False)

        # -----------------------------
        # DIVERSITY LOGIC
        # -----------------------------
        if diverse and not president:
            # 1 per president
            per_president = df.groupby("president", as_index=False).first()
            selected = per_president.to_dict("records")

            # Fill remaining slots
            if len(selected) < k:
                remaining = df[~df.index.isin(per_president.index)]
                extra = remaining.head(k - len(selected)).to_dict("records")
                selected.extend(extra)

            selected = selected[:k]
        else:
            # Just top-k
            selected = df.head(k).to_dict("records")

        # Format results
        for row in selected:
            results.append(
                {
                    "text": row["text"],
                    "president": row["president"],
                    "date": row["date"],
                    "score": row["score"],
                }
            )

    elif location:
        df = documents[documents["text"].str.contains(location, case=False, na=False)]

        if president:
            df = df[df["president"].str.contains(president, case=False, na=False)]

        df = df.copy()
        df["score"] = 1.0  # no ranking here

        # -----------------------------
        # DIVERSITY LOGIC
        # -----------------------------
        if diverse and not president:
            # 1 per president
            per_president = df.groupby("president", as_index=False).first()
            selected = per_president.to_dict("records")

            # Fill remaining
            if len(selected) < k:
                remaining = df[~df.index.isin(per_president.index)]
                extra = remaining.head(k - len(selected)).to_dict("records")
                selected.extend(extra)

            selected = selected[:k]
        else:
            # Just top-k
            selected = df.head(k).to_dict("records")

        # Format results
        for row in selected:
            results.append(
                {
                    "text": row["text"],
                    "president": row["president"],
                    "date": row["date"],
                    "score": row["score"],
                }
            )

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
            {
                "role": "system",
                "content": "You are a political analyst specializing in U.S. presidential speeches.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=300,
    )

    return response.choices[0].message.content


@st.cache_data(show_spinner=False)
def generate_country_analysis(context, country):

    prompt = f"""
You are analyzing U.S. presidential rhetoric about {country}.

Based on the excerpts below, describe:

- Overall tone (e.g., cooperative, hostile, strategic)
- How priorities differ across presidents
- Any shifts over time in how {country} is discussed
- Key geopolitical themes (war, trade, diplomacy, etc.)
- A key insight that is contrary to common perceptions about U.S. relations with {country}. Keep it in bold.

Do NOT quote directly. Do NOT repeat yourself across excerpts, and do NOT pad the text by saying the same thing. Write it like a well-researched essay that synthesizes the information into a compelling narrative. Mention all the presidents and time periods to illustrate your points.

Context:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a geopolitical analyst specializing in U.S. presidential rhetoric.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=500,
    )

    return response.choices[0].message.content


# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.markdown("## 🎛️ Filters")
st.sidebar.caption(
    "Explore how U.S. presidents referenced countries and states over time."
)

presidents = sorted(country_df["president"].dropna().unique())

selected_president = st.sidebar.selectbox(
    "Select President", ["All Presidents"] + presidents
)

year_range = st.sidebar.slider(
    "Year Range",
    int(country_df["year"].min()),
    int(country_df["year"].max()),
    (int(country_df["year"].min()), int(country_df["year"].max())),
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
        coloraxis_showscale=False,
    )
    return fig


def style_map(fig, title):
    fig.update_layout(
        title=title,
        paper_bgcolor="#0e1117",
        font_color="white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
    )
    return fig


# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(
    [
        "🌍 Country Analysis",
        #   "🇺🇸 State Analysis",
        "AI Insights",
    ]
)


def apply_filters(df):
    filtered = df.copy()

    if selected_president != "All Presidents":
        filtered = filtered[filtered["president"] == selected_president]

    filtered = filtered[
        (filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])
    ]

    return filtered


# -----------------------------
# SESSION STATE FOR CLICKED ITEM
# -----------------------------
if "clicked_country" not in st.session_state:
    st.session_state.clicked_country = None
# if "clicked_state" not in st.session_state:
#     st.session_state.clicked_state = None


# -----------------------------
# FUNCTION TO EXTRACT SENTENCES CONTAINING LOCATION
# -----------------------------
def extract_sentences(text, keyword, window=3):
    """
    Extracts sentences containing the keyword from text and highlights it.
    """
    sentences = re.split(r"(?<=[.!?]) +", text)
    matches = [s for i, s in enumerate(sentences) if keyword.lower() in s.lower()]

    styled = []
    for s in matches:
        styled.append(
            re.sub(
                f"(?i)({re.escape(keyword)})",
                r'<span style="color:#00FFFF; font-weight:bold;">\1</span>',
                s,
            )
        )

    if window > 0:
        expanded = []
        for i, s in enumerate(sentences):
            if s in matches:
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                context = " ".join(sentences[start:end])

                context = re.sub(
                    f"(?i)({re.escape(keyword)})",
                    r'<span style="color:#00FFFF; font-weight:bold;">\1</span>',
                    context,
                )
                expanded.append(context)
        return expanded

    return styled


# -----------------------------
# COUNTRY TAB
# -----------------------------
with tab1:
    st.header("🌍 Country Mentions")

    filtered = apply_filters(country_df)
    country_counts = filtered.groupby("country", as_index=False)["mentions"].sum()

    total_mentions = (
        int(country_counts["mentions"].sum()) if not country_counts.empty else 0
    )
    unique_countries = (
        int(country_counts["country"].nunique()) if not country_counts.empty else 0
    )
    top_country = (
        country_counts.sort_values("mentions", ascending=False).iloc[0]["country"]
        if not country_counts.empty
        else "N/A"
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
        color_continuous_scale="Cividis",
        title="Country Mentions",
        height=650,
    )
    fig_map.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))

    fig_map = style_map(fig_map, "Global Country Mentions")

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.plotly_chart(fig_map, use_container_width=True)

    with right_col:
        options = [""] + sorted(country_counts["country"].unique())

        selected_country = st.selectbox(
            "Select a country",
            options,
            index=options.index("China") if "China" in options else 0,
            key="country_dropdown",
        )

        if selected_country != st.session_state.clicked_country:
            st.session_state.clicked_country = selected_country
            st.rerun()

        if st.session_state.clicked_country:
            st.write(f"Showing results for **{st.session_state.clicked_country}**")

            docs = retrieve_docs(
                location=st.session_state.clicked_country,
                president=(
                    None
                    if selected_president == "All Presidents"
                    else selected_president
                ),
            )

            for i, d in enumerate(docs):
                with st.expander(f"{d['president']} ({d['date']})", expanded=False):
                    sentences = extract_sentences(
                        d["text"], st.session_state.clicked_country
                    )

                    if sentences:
                        for s in sentences:
                            st.markdown(s, unsafe_allow_html=True)
                    else:
                        st.write("No direct sentence match found.")

    if docs:
        context = "\n\n".join(
            f"{d['president']} ({d['date']}): {d['text'][:500]}" for d in docs
        )

        with st.spinner("Analyzing geopolitical sentiment..."):
            analysis = generate_country_analysis(
                context, st.session_state.clicked_country
            )

        st.subheader("🌐 Geopolitical Analysis")
        st.info(analysis)

    # -----------------------------
    # MENTIONS OVER TIME (NEW)
    # -----------------------------
    st.subheader("📈 Mentions Over Time")

    time_df = country_df.copy()

    # Apply same filters
    if selected_president != "All Presidents":
        time_df = time_df[time_df["president"] == selected_president]

    time_df = time_df[
        (time_df["year"] >= year_range[0]) & (time_df["year"] <= year_range[1])
    ]

    # Filter for selected country
    time_df = time_df[time_df["country"] == st.session_state.clicked_country]

    if not time_df.empty:
        time_series = (
            time_df.groupby("year", as_index=False)["mentions"]
            .sum()
            .sort_values("year")
        )

        fig_time = px.line(time_series, x="year", y="mentions", markers=True)

        fig_time.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            margin=dict(l=20, r=20, t=30, b=20),
        )

        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.write("No data available for this country in selected filters.")

    st.subheader("Top Countries")
    top_countries = country_counts.sort_values("mentions", ascending=False).head(15)
    top_countries = top_countries.sort_values("mentions", ascending=True)

    fig_bar = px.bar(
        top_countries,
        x="mentions",
        y="country",
        orientation="h",
        color="mentions",
        color_continuous_scale="Tealgrn",
    )
    fig_bar = style_bar_chart(fig_bar, "Top 15 Countries Mentioned")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Highlight selected country with border
    if st.session_state.clicked_country:
        highlight_df = country_counts[
            country_counts["country"] == st.session_state.clicked_country
        ]

        fig_map.add_trace(
            px.choropleth(
                highlight_df, locations="country", locationmode="country names"
            ).data[0]
        )

        fig_map.data[-1].update(
            showscale=False,
            marker=dict(line=dict(color="white", width=3)),
            hoverinfo="skip",
        )

# -----------------------------
# STATE TAB
# -----------------------------
# with tab2:
#     st.header("🇺🇸 State Mentions")

#     filtered = apply_filters(state_df)
#     state_counts = filtered.groupby("state", as_index=False)["mentions"].sum()

#     total_mentions = int(state_counts["mentions"].sum()) if not state_counts.empty else 0
#     unique_states = int(state_counts["state"].nunique()) if not state_counts.empty else 0
#     top_state = (
#         state_counts.sort_values("mentions", ascending=False).iloc[0]["state"]
#         if not state_counts.empty else "N/A"
#     )

#     c1, c2, c3 = st.columns(3)
#     c1.metric("Total Mentions", f"{total_mentions:,}")
#     c2.metric("States Mentioned", unique_states)
#     c3.metric("Most Mentioned State", top_state)

#     fig_map = px.choropleth(
#         state_counts,
#         locations="state",
#         locationmode="USA-states",
#         color="mentions",
#         scope="usa",
#         color_continuous_scale="Blues"
#     )
#     fig_map = style_map(fig_map, "U.S. State Mentions")
#     st.plotly_chart(fig_map, use_container_width=True)

#     st.subheader("Top States")
#     top_states = state_counts.sort_values("mentions", ascending=False).head(15)
#     top_states = top_states.sort_values("mentions", ascending=True)

#     fig_bar = px.bar(
#         top_states,
#         x="mentions",
#         y="state",
#         orientation="h",
#         color="mentions",
#         color_continuous_scale="Blues"
#     )
#     fig_bar = style_bar_chart(fig_bar, "Top 15 States Mentioned")
#     st.plotly_chart(fig_bar, use_container_width=True)
# -----------------------------
# AI INSIGHTS TAB
# -----------------------------
with tab2:
    st.header("🤖 Ask Questions About Presidential Speeches")
    st.caption(
        "Use retrieval + AI summarization to compare how presidents discussed a topic."
    )

    if documents is None or vectorizer is None:
        st.warning("AI Insights are unavailable because documents.pkl is missing.")
    else:
        query = st.text_input(
            "Enter your question",
            placeholder="Example: How did presidents talk about war over time?",
        )

        pres_options = ["All Presidents"] + sorted(
            documents["president"].dropna().unique().tolist()
        )
        pres_filter = st.selectbox("Filter by president (optional)", pres_options)

        if st.button("Generate Answer", use_container_width=True):
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing speeches..."):
                    chosen_president = (
                        None if pres_filter == "All Presidents" else pres_filter
                    )
                    docs = docs = retrieve_docs(
                        query=query, president=chosen_president, k=5, diverse=False
                    )

                    if not docs:
                        st.warning("No relevant speeches found.")
                    else:
                        context = "\n\n".join(
                            f"{d['president']} ({d['date']}): {d['text']}" for d in docs
                        )
                        answer = generate_answer(context, query)

                        st.subheader("Answer")
                        st.info(answer)

                        st.subheader("Source Chunks")
                        for i, d in enumerate(docs, start=1):
                            with st.expander(
                                f"Source {i}: {d['president']} ({d['date']})"
                            ):
                                st.write(d["text"])
