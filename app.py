import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Presidential Speech Country Mentions")

# -----------------------------
# Load preprocessed data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("country_mentions_counts.csv")
    return df

df = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

presidents = sorted(df["president"].unique())

selected_president = st.sidebar.selectbox(
    "Select President",
    ["All Presidents"] + presidents
)

year_range = st.sidebar.slider(
    "Year Range",
    int(df.year.min()),
    int(df.year.max()),
    (int(df.year.min()), int(df.year.max()))
)

filtered = df.copy()

if selected_president != "All Presidents":
    filtered = filtered[filtered["president"] == selected_president]

filtered = filtered[
    (filtered["year"] >= year_range[0]) &
    (filtered["year"] <= year_range[1])
]

# Aggregate country mentions
country_counts = (
    filtered.groupby("country")["mentions"]
    .sum()
    .reset_index()
)

# -----------------------------
# World Map
# -----------------------------
st.header("Countries Mentioned")

fig_map = px.choropleth(
    country_counts,
    locations="country",
    locationmode="country names",
    color="mentions",
    color_continuous_scale="Reds",
    title="Country Mentions in Presidential Speeches"
)

st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------
# Top Countries Chart
# -----------------------------
st.header("Most Mentioned Countries")

top_countries = country_counts.sort_values(
    "mentions", ascending=False
).head(15)

fig_bar = px.bar(
    top_countries,
    x="mentions",
    y="country",
    orientation="h",
    title="Top Countries Mentioned"
)

st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Mentions Over Time
# -----------------------------
st.header("Mentions Over Time")

timeline = (
    filtered.groupby("year")["mentions"]
    .sum()
    .reset_index()
)

fig_time = px.line(
    timeline,
    x="year",
    y="mentions",
    title="Total Country Mentions Per Year"
)

st.plotly_chart(fig_time, use_container_width=True)

# -----------------------------
# Country Comparison Chart
# -----------------------------
st.header("Compare Countries")

countries = sorted(df["country"].unique())

selected_countries = st.multiselect(
    "Choose countries",
    countries,
    default=["Israel", "Afghanistan"]
)

country_compare = filtered[
    filtered["country"].isin(selected_countries)
]

fig_compare = px.line(
    country_compare,
    x="year",
    y="mentions",
    color="country",
    title="Country Mentions Over Time"
)

st.plotly_chart(fig_compare, use_container_width=True)

# -----------------------------
# Dataset Explorer
# -----------------------------
st.header("Dataset")

st.dataframe(filtered)