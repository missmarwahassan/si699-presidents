import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE = "https://www.presidency.ucsb.edu"
CATEGORY = "/documents/app-categories/presidential/spoken-addresses-and-remarks"
ITEMS_PER_PAGE = 60

# Get the first page to detect total pages
url = f"{BASE}{CATEGORY}?items_per_page={ITEMS_PER_PAGE}&page=0"
r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")

# Detect total pages
pager = soup.find("ul", class_="pager")
total_pages = 3318

print("Total pages detected:", total_pages)

# Collect all links + metadata
data = []

for page in range(total_pages):
    print("Scraping page", page)
    url = f"{BASE}{CATEGORY}?items_per_page={ITEMS_PER_PAGE}&page={page}"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.find_all("div", class_="views-row")
    for row in rows:
        a_tag = row.find("a")
        if a_tag:
            link = BASE + a_tag["href"]
            title = a_tag.text.strip()
            
            # Get date and president from row
            date_tag = row.find("span", class_="date-display-single")
            date = date_tag.text.strip() if date_tag else None
            
            president_tag = row.find("h3")  # Sometimes president is in <h3>
            president = president_tag.text.strip() if president_tag else None
            
            data.append({
                "title": title,
                "url": link,
                "date": date,
                "president": president
            })
    
    time.sleep(1)  # be gentle on the server

# Save all links + metadata
df = pd.DataFrame(data)
df.to_csv("speech_links_full.csv", index=False)
print("Saved", len(df), "speech links with metadata")
