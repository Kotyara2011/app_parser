import re
import io
import time
import queue
import concurrent.futures
import urllib.parse as urlparse
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
import zipfile

import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st

# ==========================
# ---------- Utils ---------
# ==========================

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.9"}
DEFAULT_TIMEOUT = 15

CONTACT_PATH_GUESSES = [
    "/", "/contact", "/contacts", "/kontakty", "/kontact", "/o-kompanii",
    "/about", "/company", "/rekvizity", "/requisites", "/support"
]

# только телефоны начинающиеся с +
PHONE_RE = re.compile(r"\+\d[\d\s\-()]{8,}\d")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+")
INN_CANDIDATE_RE = re.compile(r"(?<!\d)(\d{10}|\d{12})(?!\d)")
INN_NEAR_RE = re.compile(r"ИНН\s*[:№#\-]*\s*(\d{10}|\d{12})", re.IGNORECASE)


def validate_inn(inn: str) -> bool:
    if not inn.isdigit():
        return False
    if len(inn) == 10:
        coeffs = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        checksum = sum(int(d) * c for d, c in zip(inn[:-1], coeffs)) % 11 % 10
        return checksum == int(inn[-1])
    elif len(inn) == 12:
        coeffs1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        coeffs2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        n11 = sum(int(d) * c for d, c in zip(inn[:-2], coeffs1)) % 11 % 10
        n12 = sum(int(d) * c for d, c in zip(inn[:-1], coeffs2)) % 11 % 10
        return n11 == int(inn[-2]) and n12 == int(inn[-1])
    return False


def clean_phone(p: str) -> str:
    s = re.sub(r"[^\d+]+", "", p)   # оставляем только цифры и +
    s = re.sub(r"^\++", "+", s)     # нормализуем префикс
    if not s.startswith("+"):       # фильтр — только номера с +
        return ""
    return s


def extract_contacts(html: str) -> Tuple[Set[str], Set[str], Set[str]]:
    phones: Set[str] = set()
    emails: Set[str] = set()
    inns: Set[str] = set()

    for m in EMAIL_RE.finditer(html):
        emails.add(m.group(0).lower())

    for m in PHONE_RE.finditer(html):
        ph = clean_phone(m.group(0))
        if ph:
            phones.add(ph)

    for m in INN_NEAR_RE.finditer(html):
        cand = m.group(1)
        if validate_inn(cand):
            inns.add(cand)

    for m in INN_CANDIDATE_RE.finditer(html):
        cand = m.group(1)
        if validate_inn(cand):
            inns.add(cand)

    return phones, emails, inns


def fetch_url(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    """Загрузить страницу и вернуть HTML, либо None если ошибка"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200 and "text" in resp.headers.get("Content-Type", ""):
            return resp.text
    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException):
        return None
    return None


def process_domain(domain: str) -> Tuple[str, Set[str], Set[str], Set[str]]:
    base_url = f"http://{domain}"
    found_phones, found_emails, found_inns = set(), set(), set()

    for path in CONTACT_PATH_GUESSES:
        url = urlparse.urljoin(base_url, path)
        html = fetch_url(url)
        if not html:
            continue  # пропускаем ошибки
        phones, emails, inns = extract_contacts(html)
        found_phones.update(phones)
        found_emails.update(emails)
        found_inns.update(inns)

    return domain, found_phones, found_emails, found_inns


# ==========================
# ---------- Streamlit -----
# ==========================

st.title("Парсер контактов с сайтов")

uploaded_file = st.file_uploader("Загрузите ZIP с доменами (.txt)", type=["zip"])

if uploaded_file is not None:
    with zipfile.ZipFile(uploaded_file, "r") as z:
        name_list = z.namelist()
        if not name_list:
            st.error("В архиве нет файлов")
        else:
            file_name = name_list[0]
            with z.open(file_name) as f:
                domains = [line.decode("utf-8").strip() for line in f if line.strip()]

    st.write(f"Найдено доменов: {len(domains)}")

    results = []
    progress = st.progress(0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_domain, d): d for d in domains}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            domain, phones, emails, inns = future.result()
            results.append({
                "domain": domain,
                "phones": ", ".join(sorted(phones)),
                "emails": ", ".join(sorted(emails)),
                "inns": ", ".join(sorted(inns)),
            })
            progress.progress((i + 1) / len(domains))

    df = pd.DataFrame(results)
    st.dataframe(df)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False, sep=";")
    st.download_button("Скачать CSV", csv_buf.getvalue(), "contacts.csv", "text/csv")

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button("Скачать Excel", xlsx_buf.getvalue(), "contacts.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
