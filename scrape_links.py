import pandas as pd
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

links_df = pd.read_csv("speech_links_full.csv")

def fetch_speech(row):
    url = row['url']
    title = row['title']
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        # Text
        text_div = soup.find("div", class_="field-docs-content")
        text = text_div.get_text(separator=" ", strip=True) if text_div else ""
        # Date
        date_tag = soup.find("span", class_="date-display-single")
        date = date_tag.text.strip() if date_tag else None
        # President
        pres_tag = soup.find("h3")
        president = pres_tag.text.strip() if pres_tag else None
        return {"title": title, "url": url, "text": text, "date": date, "president": president}
    except Exception as e:
        print("Error:", url, e)
        return None

results = []

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_speech, row) for idx, row in links_df.iterrows()]
    for future in tqdm(as_completed(futures), total=len(futures)):
        res = future.result()
        if res:
            results.append(res)
        time.sleep(0.05)  # be gentle on the server

# Save full dataset
df = pd.DataFrame(results)
df.to_csv("presidential_speeches_full.csv", index=False)
print("Scraped", len(df), "speeches")