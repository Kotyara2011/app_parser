import re
import io
import time
import queue
import zipfile
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================== #
# ---------- Utils --------- #
# ========================== #

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

HEADERS = {"User-Agent": USER_AGENT}
DEFAULT_TIMEOUT = 10

# ========================== #
# ---------- Data ---------- #
# ========================== #

@dataclass
class PageResult:
    url: str
    status_code: Optional[int]
    text: str

# ========================== #
# --------- Crawler -------- #
# ========================== #

class Crawler:
    def __init__(self, max_workers=5, session=None):
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def fetch(self, url: str) -> PageResult:
        try:
            r = self.session.get(url, timeout=DEFAULT_TIMEOUT)
            if not r.ok:
                return PageResult(url, r.status_code, "")
            soup = BeautifulSoup(r.text, "lxml")
            text = soup.get_text(" ", strip=True)[:200_000]  # до 200К текста
            return PageResult(url, r.status_code, text)
        except Exception as e:
            return PageResult(url, None, f"ERROR: {e}")

    def crawl_batch(self, urls: List[str]) -> List[PageResult]:
        futures = {self.executor.submit(self.fetch, url): url for url in urls}
        results = []
        for f in as_completed(futures):
            results.append(f.result())
        return results

# ========================== #
# ------- Extraction ------- #
# ========================== #

def extract_contacts(text: str) -> Dict[str, List[str]]:
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phones = re.findall(r"(?:\+?\d[\d\-\s]{7,}\d)", text)
    return {
        "emails": list(set(emails)),
        "phones": list(set(phones)),
    }

# ========================== #
# --------- Streamlit ------ #
# ========================== #

def main():
    st.title("🔎 Поиск контактов на сайтах")
    uploaded_file = st.file_uploader("Загрузите список доменов (txt, csv, xlsx)", type=["txt", "csv", "xlsx"])

    if uploaded_file:
        # читаем список доменов
        if uploaded_file.name.endswith(".txt"):
            domains = [line.strip() for line in uploaded_file if line.strip()]
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            domains = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            df = pd.read_excel(uploaded_file)
            domains = df.iloc[:, 0].dropna().astype(str).tolist()

        st.write(f"Загружено доменов: {len(domains)}")

        if st.button("🚀 Начать обработку"):
            crawler = Crawler(max_workers=5)
            batch_size = 200
            all_records = []

            for i in range(0, len(domains), batch_size):
                batch = domains[i:i+batch_size]
                st.info(f"Обрабатываю {i+1} - {i+len(batch)} доменов...")

                urls = [f"http://{d.strip()}" for d in batch]
                results = crawler.crawl_batch(urls)

                for r in results:
                    contacts = extract_contacts(r.text)
                    all_records.append({
                        "url": r.url,
                        "status": r.status_code,
                        "emails": "; ".join(contacts["emails"]),
                        "phones": "; ".join(contacts["phones"]),
                    })

                # промежуточный результат
                df = pd.DataFrame(all_records)
                st.dataframe(df.tail(10))

            # сохраняем результат
            df = pd.DataFrame(all_records)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="contacts")
            st.download_button("📥 Скачать Excel", data=output.getvalue(),
                               file_name="contacts.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Скачать CSV", data=csv_data,
                               file_name="contacts.csv", mime="text/csv")


if __name__ == "__main__":
    main()
