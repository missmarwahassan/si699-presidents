import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Presidential Rhetoric Dashboard",
    page_icon="🇺🇸",
    layout="wide"
)

# -----------------------------
# STYLING
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b2030 0%, #171b26 100%);
    }

    div[data-testid="stMetric"] {
        background-color: #141926;
        border: 1px solid #2a3144;
        padding: 14px;
        border-radius: 16px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02);
    }

    .hero {
        padding: 1.2rem 1.25rem;
        border: 1px solid #2a3144;
        border-radius: 18px;
        background: linear-gradient(135deg, #121826 0%, #182235 100%);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .hero-sub {
        font-size: 1rem;
        color: #c9d2e3;
        line-height: 1.6;
    }

    .section-card {
        padding: 1rem 1.1rem;
        border: 1px solid #2a3144;
        border-radius: 16px;
        background: #101521;
    }

    .insight-card {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid #2b3349;
        background: #121826;
        margin-bottom: 0.75rem;
    }

    .insight-title {
        font-size: 0.85rem;
        color: #98a4bd;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .insight-body {
        font-size: 1rem;
        font-weight: 600;
        color: #f2f5fb;
        line-height: 1.45;
    }

    .small-note {
        color: #9aa7bf;
        font-size: 0.95rem;
    }

    .pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        border: 1px solid #33405b;
        background: #121826;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# FILE PATHS
# -----------------------------
BASE_DIR = Path(".")
DOCUMENTS_PATH = BASE_DIR / "documents.pkl"
COUNTRY_PATH = BASE_DIR / "country_mentions_counts.csv"
STATE_PATH = BASE_DIR / "state_mentions_counts.csv"

# -----------------------------
# OPENAI CLIENT
# -----------------------------
api_key = ""
try:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    api_key = os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(api_key=api_key) if api_key else None


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_documents():
    if DOCUMENTS_PATH.exists():
        return pd.read_pickle(DOCUMENTS_PATH)
    return None

@st.cache_data
def load_country_data():
    if COUNTRY_PATH.exists():
        return pd.read_csv(COUNTRY_PATH)
    return None

@st.cache_data
def load_state_data():
    if STATE_PATH.exists():
        return pd.read_csv(STATE_PATH)
    return None


documents = load_documents()
country_df = load_country_data()
state_df = load_state_data()

# -----------------------------
# VALIDATION
# -----------------------------
if country_df is None or state_df is None:
    st.error(
        "Required data files are missing. Make sure these files are in the same folder as app.py:\n"
        "- country_mentions_counts.csv\n"
        "- state_mentions_counts.csv\n"
        "- documents.pkl (optional, needed for AI Insights)"
    )
    st.stop()

# Clean year columns just in case
country_df["year"] = pd.to_numeric(country_df["year"], errors="coerce")
state_df["year"] = pd.to_numeric(state_df["year"], errors="coerce")

country_df = country_df.dropna(subset=["year"]).copy()
state_df = state_df.dropna(subset=["year"]).copy()

country_df["year"] = country_df["year"].astype(int)
state_df["year"] = state_df["year"].astype(int)

# -----------------------------
# BUILD TF-IDF
# -----------------------------
@st.cache_data
def build_vectorizer(_documents):
    if _documents is None or _documents.empty:
        return None, None

    texts = _documents["text"].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

vectorizer, tfidf_matrix = build_vectorizer(documents)


# -----------------------------
# HELPERS
# -----------------------------
def apply_filters(df, selected_president, year_range):
    filtered = df.copy()

    if selected_president != "All Presidents":
        filtered = filtered[filtered["president"] == selected_president]

    filtered = filtered[
        (filtered["year"] >= year_range[0]) &
        (filtered["year"] <= year_range[1])
    ]
    return filtered


def retrieve_docs(query, president=None, k=5):
    if documents is None or vectorizer is None or tfidf_matrix is None:
        return []

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-k * 3:][::-1]

    results = []
    for i in top_indices:
        row = documents.iloc[i]

        if president and president.lower() not in str(row["president"]).lower():
            continue

        results.append({
            "text": str(row["text"])[:550],
            "president": row.get("president", "Unknown"),
            "date": row.get("date", "Unknown"),
            "title": row.get("title", "Untitled")
        })

        if len(results) >= k:
            break

    return results


@st.cache_data(show_spinner=False)
def generate_answer(context, question):
    if client is None:
        return "AI Insights are unavailable because no OpenAI API key was found."

    prompt = f"""
Context:
{context}

Question:
{question}

Please analyze how presidents discussed this topic across speeches.
Focus on:
- tone
- priorities
- rhetorical framing
- major differences across presidents or eras

Do not quote long passages directly.
Be concise but insightful.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a political analyst specializing in U.S. presidential speeches and rhetoric."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4,
        max_tokens=350
    )

    return response.choices[0].message.content


def style_plot(fig, title=None, height=430):
    fig.update_layout(
        title=title,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=height,
        title_font_size=22,
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=False)
    return fig


def style_map(fig, title=None, height=480):
    fig.update_layout(
        title=title,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="white",
        margin=dict(l=10, r=10, t=60, b=10),
        height=height
    )
    return fig


def safe_top_value(df, label_col, value_col):
    if df.empty:
        return "N/A"
    return df.sort_values(value_col, ascending=False).iloc[0][label_col]


def compute_fastest_growth(df, group_col):
    if df.empty:
        return "N/A"

    yearly = df.groupby([group_col, "year"], as_index=False)["mentions"].sum()
    pivot = yearly.pivot(index="year", columns=group_col, values="mentions").fillna(0)

    if pivot.shape[0] < 2:
        return "N/A"

    growth = pivot.iloc[-1] - pivot.iloc[0]
    if growth.empty:
        return "N/A"

    return growth.sort_values(ascending=False).index[0]


def compute_peak_year(df):
    if df.empty:
        return "N/A"

    yearly = df.groupby("year", as_index=False)["mentions"].sum()
    if yearly.empty:
        return "N/A"

    return int(yearly.sort_values("mentions", ascending=False).iloc[0]["year"])


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.markdown("## 🎛️ Filters")
st.sidebar.caption("Explore how U.S. presidents referenced countries and states over time.")

presidents = sorted(country_df["president"].dropna().unique().tolist())

selected_president = st.sidebar.selectbox(
    "Select President",
    ["All Presidents"] + presidents
)

min_year = int(min(country_df["year"].min(), state_df["year"].min()))
max_year = int(max(country_df["year"].max(), state_df["year"].max()))

year_range = st.sidebar.slider(
    "Year Range",
    min_year,
    max_year,
    (min_year, max_year)
)

show_data = st.sidebar.checkbox("Show filtered data tables", value=False)
show_story = st.sidebar.checkbox("Show insight cards", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("SI 699 • Presidential rhetoric dashboard")


# -----------------------------
# HERO
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🇺🇸 Presidential Rhetoric Dashboard</div>
        <div class="hero-sub">
            Explore how U.S. presidents referenced countries and states across time, and use AI-assisted retrieval
            to ask deeper questions about tone, priorities, and rhetorical change. This project connects historical
            speech patterns to civic understanding, public discourse, and the evolving priorities of political leadership.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <span class="pill">🌍 Foreign focus</span>
    <span class="pill">🇺🇸 Domestic focus</span>
    <span class="pill">📈 Change over time</span>
    <span class="pill">🤖 AI speech analysis</span>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🌍 Country Analysis",
    "🇺🇸 State Analysis",
    "🤖 AI Insights"
])


# =============================
# COUNTRY TAB
# =============================
with tab1:
    filtered = apply_filters(country_df, selected_president, year_range)
    country_counts = filtered.groupby("country", as_index=False)["mentions"].sum()
    yearly_country = filtered.groupby("year", as_index=False)["mentions"].sum()

    total_mentions = int(country_counts["mentions"].sum()) if not country_counts.empty else 0
    unique_countries = int(country_counts["country"].nunique()) if not country_counts.empty else 0
    top_country = safe_top_value(country_counts, "country", "mentions")
    fastest_growth_country = compute_fastest_growth(filtered, "country")
    peak_year = compute_peak_year(filtered)

    st.header("🌍 Country Mentions")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Mentions", f"{total_mentions:,}")
    m2.metric("Countries Mentioned", unique_countries)
    m3.metric("Most Mentioned Country", top_country)

    if show_story:
        left_insight, right_insight = st.columns(2)
        with left_insight:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">Peak rhetorical moment</div>
                    <div class="insight-body">Country references were highest in <b>{peak_year}</b>.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with right_insight:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">Fastest growth across selected years</div>
                    <div class="insight-body"><b>{fastest_growth_country}</b> shows the strongest increase across the selected range.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    map_col, side_col = st.columns([2.2, 1])

    with map_col:
        fig_map = px.choropleth(
            country_counts,
            locations="country",
            locationmode="country names",
            color="mentions",
            color_continuous_scale="Viridis",
            hover_name="country",
            hover_data={"mentions": True}
        )
        fig_map = style_map(fig_map, "Global Country Mentions", height=500)
        st.plotly_chart(fig_map, use_container_width=True)

    with side_col:
        st.markdown("### Key Takeaways")
        st.markdown(
            f"""
            <div class="section-card">
                <p><b>Top country:</b> {top_country}</p>
                <p><b>Total mentions:</b> {total_mentions:,}</p>
                <p><b>Distinct countries:</b> {unique_countries}</p>
                <p class="small-note">
                    Use the year slider and president filter to see how global attention shifts across eras.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    c_left, c_right = st.columns([1.15, 1])

    with c_left:
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
        fig_bar = style_plot(fig_bar, "Top 15 Countries Mentioned", height=500)
        fig_bar.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        fig_line = px.line(
            yearly_country,
            x="year",
            y="mentions",
            markers=True
        )
        fig_line = style_plot(fig_line, "Country Mentions Over Time", height=500)
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Compare Presidents by Country")
    compare_presidents = st.multiselect(
        "Select up to 4 presidents",
        options=presidents,
        default=presidents[:2] if len(presidents) >= 2 else presidents,
        max_selections=4,
        key="country_compare_presidents"
    )

    compare_country = st.selectbox(
        "Select country to compare",
        options=sorted(country_df["country"].dropna().unique().tolist()),
        index=0,
        key="compare_country"
    )

    if compare_presidents:
        comp_df = country_df[
            (country_df["president"].isin(compare_presidents)) &
            (country_df["country"] == compare_country) &
            (country_df["year"] >= year_range[0]) &
            (country_df["year"] <= year_range[1])
        ]
        comp_df = comp_df.groupby(["year", "president"], as_index=False)["mentions"].sum()

        fig_compare = px.line(
            comp_df,
            x="year",
            y="mentions",
            color="president",
            markers=True
        )
        fig_compare = style_plot(fig_compare, f"{compare_country} Mentions by President", height=440)
        st.plotly_chart(fig_compare, use_container_width=True)

    if show_data:
        with st.expander("Show filtered country data"):
            st.dataframe(filtered, use_container_width=True)


# =============================
# STATE TAB
# =============================
with tab2:
    filtered = apply_filters(state_df, selected_president, year_range)
    state_counts = filtered.groupby("state", as_index=False)["mentions"].sum()
    yearly_state = filtered.groupby("year", as_index=False)["mentions"].sum()

    total_mentions = int(state_counts["mentions"].sum()) if not state_counts.empty else 0
    unique_states = int(state_counts["state"].nunique()) if not state_counts.empty else 0
    top_state = safe_top_value(state_counts, "state", "mentions")
    fastest_growth_state = compute_fastest_growth(filtered, "state")
    peak_year = compute_peak_year(filtered)

    st.header("🇺🇸 State Mentions")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Mentions", f"{total_mentions:,}")
    m2.metric("States Mentioned", unique_states)
    m3.metric("Most Mentioned State", top_state)

    if show_story:
        left_insight, right_insight = st.columns(2)
        with left_insight:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">Peak year</div>
                    <div class="insight-body">State references peaked in <b>{peak_year}</b>.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with right_insight:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">Fastest growth across selected years</div>
                    <div class="insight-body"><b>{fastest_growth_state}</b> shows the strongest increase across the selected range.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    map_col, side_col = st.columns([2.2, 1])

    with map_col:
        state_map_df = state_df.copy()
        state_map_df = state_map_df[
            (state_map_df["year"] >= year_range[0]) &
            (state_map_df["year"] <= year_range[1])
        ]
        if selected_president != "All Presidents":
            state_map_df = state_map_df[state_map_df["president"] == selected_president]

        if "abbr" in state_map_df.columns:
            state_map_df = state_map_df.groupby(["state", "abbr"], as_index=False)["mentions"].sum()

            fig_map = px.choropleth(
                state_map_df,
                locations="abbr",
                locationmode="USA-states",
                color="mentions",
                scope="usa",
                color_continuous_scale="Blues",
                hover_name="state",
                hover_data={"abbr": False, "mentions": True}
            )
        else:
            fig_map = px.choropleth(
                state_counts,
                locations="state",
                locationmode="USA-states",
                color="mentions",
                scope="usa",
                color_continuous_scale="Blues"
            )

        fig_map = style_map(fig_map, "U.S. State Mentions", height=500)
        st.plotly_chart(fig_map, use_container_width=True)

    with side_col:
        st.markdown("### Key Takeaways")
        st.markdown(
            f"""
            <div class="section-card">
                <p><b>Top state:</b> {top_state}</p>
                <p><b>Total mentions:</b> {total_mentions:,}</p>
                <p><b>States referenced:</b> {unique_states}</p>
                <p class="small-note">
                    This view helps show which states dominate rhetorical attention in speeches and remarks.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    s_left, s_right = st.columns([1.15, 1])

    with s_left:
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
        fig_bar = style_plot(fig_bar, "Top 15 States Mentioned", height=500)
        fig_bar.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with s_right:
        fig_line = px.line(
            yearly_state,
            x="year",
            y="mentions",
            markers=True
        )
        fig_line = style_plot(fig_line, "State Mentions Over Time", height=500)
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Compare Presidents by State")
    compare_presidents = st.multiselect(
        "Select up to 4 presidents",
        options=presidents,
        default=presidents[:2] if len(presidents) >= 2 else presidents,
        max_selections=4,
        key="state_compare_presidents"
    )

    compare_state = st.selectbox(
        "Select state to compare",
        options=sorted(state_df["state"].dropna().unique().tolist()),
        index=0,
        key="compare_state"
    )

    if compare_presidents:
        comp_df = state_df[
            (state_df["president"].isin(compare_presidents)) &
            (state_df["state"] == compare_state) &
            (state_df["year"] >= year_range[0]) &
            (state_df["year"] <= year_range[1])
        ]
        comp_df = comp_df.groupby(["year", "president"], as_index=False)["mentions"].sum()

        fig_compare = px.line(
            comp_df,
            x="year",
            y="mentions",
            color="president",
            markers=True
        )
        fig_compare = style_plot(fig_compare, f"{compare_state} Mentions by President", height=440)
        st.plotly_chart(fig_compare, use_container_width=True)

    if show_data:
        with st.expander("Show filtered state data"):
            st.dataframe(filtered, use_container_width=True)


# =============================
# AI TAB
# =============================
with tab3:
    st.header("🤖 Ask Questions About Presidential Speeches")
    st.caption("Use retrieval + AI summarization to compare how presidents discussed a topic across speeches.")

    example_cols = st.columns(3)
    examples = [
        "How did presidents talk about war over time?",
        "How did presidents describe democracy?",
        "How did presidents discuss China versus Russia?"
    ]

    selected_example = None
    for idx, example in enumerate(examples):
        with example_cols[idx]:
            if st.button(example, use_container_width=True, key=f"example_{idx}"):
                selected_example = example

    default_query = selected_example if selected_example else ""
    query = st.text_input(
        "Enter your question",
        value=default_query,
        placeholder="Example: How did presidents talk about war over time?"
    )

    pres_options = ["All Presidents"] + sorted(documents["president"].dropna().unique().tolist()) if documents is not None else ["All Presidents"]
    pres_filter = st.selectbox("Filter by president (optional)", pres_options)

    if documents is None or vectorizer is None:
        st.warning("AI Insights are unavailable because documents.pkl is missing.")
    elif client is None:
        st.warning("AI Insights are unavailable because no OpenAI API key was found.")
    else:
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
                            f"{d['president']} ({d['date']}) - {d['title']}: {d['text']}"
                            for d in docs
                        )
                        answer = generate_answer(context, query)

                        st.subheader("Answer")
                        st.markdown(
                            f"""
                            <div class="section-card">
                                {answer}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.subheader("Retrieved Source Chunks")
                        for i, d in enumerate(docs, start=1):
                            with st.expander(f"Source {i}: {d['president']} ({d['date']})"):
                                st.markdown(f"**Title:** {d['title']}")
                                st.write(d["text"])

    st.markdown("---")
    st.markdown("### Why this matters")
    st.markdown(
        """
        Presidential rhetoric helps shape public understanding of threats, alliances, national priorities,
        and civic identity. By tracking what presidents emphasize over time, this dashboard supports more
        informed engagement with political speech and historical change.
        """
    )