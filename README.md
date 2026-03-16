# Presidential Speech Country Mentions Analysis

This project analyzes **U.S. presidential speeches to identify which countries are mentioned and how those references change over time**. Using web scraping, natural language processing, and interactive visualization, the project builds a reproducible pipeline that collects speeches, extracts geopolitical entities, and visualizes patterns in presidential rhetoric.

The data is sourced from the **UCSB American Presidency Project**, which hosts thousands of presidential speeches and official remarks.

---

# Project Overview

This project answers questions such as:

* Which countries are mentioned most frequently in presidential speeches?
* How do country mentions change across presidential administrations?
* How does international focus shift over time?
* Which countries are discussed together during specific periods?

The project combines:

* Web scraping
* Natural language processing (NLP)
* Data aggregation
* Interactive visualization

The final result is an **interactive Streamlit dashboard** that allows users to explore country mentions by president and year.

---

# Data Pipeline

The project follows a multi-step pipeline.

```
scrape_links.py
      ↓
speech_links_full.csv
      ↓
scrape_pres_doc_data.py
      ↓
presidential_speeches_full.csv
      ↓
extract_country_mentions.py
      ↓
country_mentions_counts.csv
      ↓
app.py (Streamlit Dashboard)
```

---

# Repository Structure

```
project/
│
├── scrape_links.py
├── scrape_pres_doc_data.py
├── extract_country_mentions.py
├── app.py
│
├── speech_links_full.csv
├── presidential_speeches_full.csv
├── country_mentions_counts.csv
│
├── LICENSE
└── README.md
```

---

# Step 1: Scraping Speech Links

`scrape_links.py` collects metadata and URLs for presidential speeches from the UCSB Presidency Project website.

The script extracts:

* Speech title
* Speech URL
* Date
* President

These links are saved to:

```
speech_links_full.csv
```

The scraper iterates through thousands of pages of presidential documents to build the dataset. 

---

# Step 2: Scraping Full Speech Text

`scrape_pres_doc_data.py` downloads the full speech text from each URL collected in the previous step.

For each speech the script extracts:

* Title
* URL
* Full speech text
* Date
* President

The completed dataset is saved as:

```
presidential_speeches_full.csv
```

The script uses **multithreading** to speed up scraping. 

---

# Step 3: Extracting Country Mentions

`extract_country_mentions.py` processes speeches using **spaCy Named Entity Recognition (NER)**.

The script:

1. Loads the speech dataset
2. Identifies geopolitical entities (GPE)
3. Filters entities to valid countries using the `pycountry` library
4. Counts mentions by president and year

The output dataset contains:

```
country
president
year
mentions
```

Saved as:

```
country_mentions_counts.csv
```

Country mentions are extracted from geopolitical entities detected in each speech. 

---

# Step 4: Interactive Visualization Dashboard

`app.py` builds a **Streamlit dashboard** that visualizes the processed data.

The dashboard includes:

### World Map

Displays global country mentions using a choropleth map.

### Top Countries Chart

Shows the most frequently mentioned countries.

### Mentions Over Time

Tracks the number of country mentions by year.

### Country Comparison Tool

Allows users to compare mentions for selected countries.

### Dataset Explorer

Displays the filtered dataset for further exploration.

The dashboard reads the processed dataset and updates dynamically based on user filters. 

---

# Technologies Used

* Python
* pandas
* spaCy
* pycountry
* BeautifulSoup
* requests
* Plotly
* Streamlit
* tqdm

---

# Running the Project

## Install Dependencies

```
pip install pandas spacy pycountry requests beautifulsoup4 plotly streamlit tqdm
```

Download the spaCy language model:

```
python -m spacy download en_core_web_sm
```

---

## Run the Dashboard

```
streamlit run app.py
```

This will launch the interactive dashboard locally.

---

# Example Research Questions

This dataset can be used to explore:

* How foreign policy focus shifts between presidents
* How major global events affect country mentions
* Which regions receive the most rhetorical attention
* Differences in geopolitical framing between administrations

---

# Future Improvements

Possible extensions of this project include:

* Sentiment analysis of country mentions
* Topic modeling for foreign policy themes
* Network analysis of co-mentioned countries
* Comparing rhetorical focus across political parties
* Detecting geopolitical crises through speech trends

---

# Data Source

American Presidency Project
[https://www.presidency.ucsb.edu](https://www.presidency.ucsb.edu)

---

# License

See the `LICENSE` file for details.
