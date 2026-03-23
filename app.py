import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Presidential Speech Geographic Mentions")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_country_data():
    return pd.read_csv("country_mentions_counts.csv")

@st.cache_data
def load_state_data():
    return pd.read_csv("state_mentions_counts.csv")

country_df = load_country_data()
state_df = load_state_data()

# -----------------------------
# Shared Filters
# -----------------------------
st.sidebar.header("Filters")

# combine presidents from both datasets
all_presidents = sorted(
    list(set(country_df["president"]).union(set(state_df["president"])))
)

selected_president = st.sidebar.selectbox(
    "Select President",
    ["All Presidents"] + all_presidents
)

# combine year ranges
min_year = int(min(country_df.year.min(), state_df.year.min()))
max_year = int(max(country_df.year.max(), state_df.year.max()))

year_range = st.sidebar.slider(
    "Year Range",
    min_year,
    max_year,
    (min_year, max_year)
)

# -----------------------------
# Filter function
# -----------------------------
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
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Countries", "US States"])

# =====================================================
# 🌍 COUNTRY TAB
# =====================================================
with tab1:

    st.header("Countries Mentioned")

    filtered = apply_filters(country_df)

    country_counts = (
        filtered.groupby("country")["mentions"]
        .sum()
        .reset_index()
    )

    # Map
    fig_map = px.choropleth(
        country_counts,
        locations="country",
        locationmode="country names",
        color="mentions",
        color_continuous_scale="Viridis",
        title="Country Mentions"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Top countries
    st.subheader("Top Countries")

    top_countries = country_counts.sort_values(
        "mentions", ascending=False
    ).head(15)

    fig_bar = px.bar(
        top_countries,
        x="mentions",
        y="country",
        orientation="h"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # Timeline
    st.subheader("Mentions Over Time")

    timeline = (
        filtered.groupby("year")["mentions"]
        .sum()
        .reset_index()
    )

    fig_time = px.line(
        timeline,
        x="year",
        y="mentions"
    )

    st.plotly_chart(fig_time, use_container_width=True)

    # Compare countries
    st.subheader("Compare Countries")

    countries = sorted(country_df["country"].unique())

    selected_countries = st.multiselect(
        "Choose countries",
        countries,
        default=countries[:2]
    )

    compare = filtered[
        filtered["country"].isin(selected_countries)
    ]

    fig_compare = px.line(
        compare,
        x="year",
        y="mentions",
        color="country"
    )

    st.plotly_chart(fig_compare, use_container_width=True)

# =====================================================
# 🇺🇸 STATE TAB
# =====================================================
with tab2:

    st.header("US State Mentions")

    filtered = apply_filters(state_df)

    state_counts = (
        filtered.groupby(["state","abbr"])["mentions"]
        .sum()
        .reset_index()
    )

    # Map
    fig_map = px.choropleth(
        state_counts,
        locations="abbr",
        locationmode="USA-states",
        color="mentions",
        scope="usa",
        color_continuous_scale="Reds",
        title="State Mentions"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Top states
    st.subheader("Top States")

    top_states = state_counts.sort_values(
        "mentions", ascending=False
    ).head(15)

    fig_bar = px.bar(
        top_states,
        x="mentions",
        y="state",
        orientation="h"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # Timeline
    st.subheader("Mentions Over Time")

    timeline = (
        filtered.groupby("year")["mentions"]
        .sum()
        .reset_index()
    )

    fig_time = px.line(
        timeline,
        x="year",
        y="mentions"
    )

    st.plotly_chart(fig_time, use_container_width=True)

    # Compare states
    st.subheader("Compare States")

    states = sorted(state_df["state"].unique())

    selected_states = st.multiselect(
        "Choose states",
        states,
        default=states[:2]
    )

    compare = filtered[
        filtered["state"].isin(selected_states)
    ]

    fig_compare = px.line(
        compare,
        x="year",
        y="mentions",
        color="state"
    )

    st.plotly_chart(fig_compare, use_container_width=True)

# -----------------------------
# Dataset Viewer
# -----------------------------
st.header("Dataset Preview")

view_option = st.radio(
    "Choose dataset",
    ["Countries", "States"]
)

if view_option == "Countries":
    st.dataframe(country_df)
else:
    st.dataframe(state_df)