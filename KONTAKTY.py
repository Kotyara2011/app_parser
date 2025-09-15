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

PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+")
INN_CANDIDATE_RE = re.compile(r"(?<!\d)(\d{10}|\d{12})(?!\d)")
INN_NEAR_RE = re.compile(r"ИНН\s*[:№#\-]*\s*(\d{10}|\d{12})", re.IGNORECASE)

# ---------- INN validation ----------

def _checksum(digits: List[int], coeffs: List[int]) -> int:
    s = sum(d * c for d, c in zip(digits, coeffs))
    return (s % 11) % 10


def validate_inn(inn: str) -> bool:
    if not inn.isdigit():
        return False
    if len(inn) == 10:
        coeffs10 = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        d = [int(x) for x in inn]
        return _checksum(d[:9], coeffs10) == d[9]
    if len(inn) == 12:
        coeffs11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        coeffs12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        d = [int(x) for x in inn]
        ctrl11 = _checksum(d[:10], coeffs11)
        ctrl12 = _checksum(d[:11], coeffs12)
        return ctrl11 == d[10] and ctrl12 == d[11]
    return False

# ---------- Helpers for domain normalization ----------

DATE_RE_SIMPLE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DOMAIN_LIKE_RE = re.compile(r"^(?:https?://)?(?:www\.)?([A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63})+)(?:[:/].*)?$")


def looks_like_date(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s:
        return False
    if DATE_RE_SIMPLE.match(s):
        return True
    try:
        ts = pd.to_datetime(s, errors='coerce')
        return not pd.isna(ts)
    except Exception:
        return False


def normalize_domain(raw: object) -> Optional[str]:
    if raw is None:
        return None
    if hasattr(raw, 'tzinfo') or isinstance(raw, (pd.Timestamp,)):
        return None

    s = str(raw).strip()
    if not s:
        return None
    if s == ';' or (';' in s and len(s.strip(';').strip()) == 0):
        return None
    if looks_like_date(s):
        return None

    parts = re.split(r"[;,\s]+", s)
    for part in parts:
        token = part.strip().strip('"\'')
        if not token or token == ';':
            continue
        if '@' in token:
            continue
        m = DOMAIN_LIKE_RE.match(token)
        host = None
        if m:
            host = m.group(1)
        else:
            try:
                p = urlparse.urlparse(token if token.startswith('http') else 'https://' + token)
                if p.netloc:
                    host = p.netloc
                else:
                    host = token
            except Exception:
                host = None
        if host:
            host = host.split('@')[-1].split(':')[0].lower()
            if '.' in host and not host.replace('.', '').isdigit():
                if host.startswith('www.'):
                    host = host[4:]
                return f"https://{host}"
    return None


def urljoin_keep(base: str, path: str) -> str:
    return urlparse.urljoin(base if base.endswith('/') else base + '/', path)

# ---------- Fetching ----------

session = requests.Session()
session.headers.update(HEADERS)

def fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        ct = r.headers.get('content-type', '')
        if r.status_code >= 200 and r.status_code < 300 and ('text' in ct or 'html' in ct):
            # ограничим размер, чтобы не переполнять память
            content = r.text[:200_000]
            soup = BeautifulSoup(content, "lxml")
            return soup.get_text(" ", strip=True)
    except Exception:
        return None
    return None

# ---------- Extraction ----------

def clean_phone(p: str) -> str:
    s = re.sub(r"[^\d+]+", "", p)
    s = re.sub(r"^\++", "+", s)
    return s


def extract_contacts(text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    phones: Set[str] = set()
    emails: Set[str] = set()
    inns: Set[str] = set()

    for m in EMAIL_RE.finditer(text):
        emails.add(m.group(0).lower())

    for m in PHONE_RE.finditer(text):
        phones.add(clean_phone(m.group(0)))

    for m in INN_NEAR_RE.finditer(text):
        cand = m.group(1)
        if validate_inn(cand):
            inns.add(cand)

    for m in INN_CANDIDATE_RE.finditer(text):
        cand = m.group(1)
        if validate_inn(cand):
            inns.add(cand)

    return phones, emails, inns

# ---------- Crawl per domain ----------

@dataclass
class CrawlResult:
    domain: str
    page_hits: List[Dict[str, object]] = field(default_factory=list)

    def aggregate(self) -> Dict[str, object]:
        agg_phones: Set[str] = set()
        agg_emails: Set[str] = set()
        agg_inns: Set[str] = set()
        for row in self.page_hits:
            agg_phones.update(row.get('phones', set()))
            agg_emails.update(row.get('emails', set()))
            agg_inns.update(row.get('inns', set()))
        return {
            "Домен": self.domain,
            "Телефоны": ", ".join(sorted(agg_phones)) if agg_phones else "",
            "Почты": ", ".join(sorted(agg_emails)) if agg_emails else "",
            "ИНН": ", ".join(sorted(agg_inns)) if agg_inns else "",
        }


def same_host(url: str, base: str) -> bool:
    try:
        p1 = urlparse.urlparse(url)
        p2 = urlparse.urlparse(base)
        return p1.netloc.lower().lstrip("www.") == p2.netloc.lower().lstrip("www.")
    except Exception:
        return False


def crawl_domain(base: str, max_pages: int = 15, timeout: int = DEFAULT_TIMEOUT) -> CrawlResult:
    seen: Set[str] = set()
    q: "queue.Queue[str]" = queue.Queue()

    for path in CONTACT_PATH_GUESSES:
        q.put(urljoin_keep(base, path))

    result = CrawlResult(domain=base)

    while not q.empty() and len(seen) < max_pages:
        url = q.get()
        if url in seen:
            continue
        seen.add(url)

        text = fetch(url, timeout=timeout)
        if not text:
            continue

        phones, emails, inns = extract_contacts(text)
        if phones or emails or inns:
            result.page_hits.append({
                "url": url,
                "phones": phones,
                "emails": emails,
                "inns": inns,
            })

        # ссылки теперь не парсим, чтобы снизить нагрузку
    return result

# ==========================
# ---------- Streamlit UI --
# ==========================

st.set_page_config(page_title="Парсер контактов по доменам", page_icon="📇", layout="wide")

st.title("📇 Парсер контактов: Телефоны / Email / ИНН")

with st.sidebar:
    st.header("Настройки")
    max_pages = st.slider("Максимум страниц на домен", 3, 20, 10, step=1)
    timeout = st.slider("Таймаут запроса (сек)", 5, 60, DEFAULT_TIMEOUT, step=1)
    workers = st.slider("Количество потоков", 2, 10, 5, step=1)

uploaded = st.file_uploader("Загрузите Excel (.xlsx) или CSV (.csv)", type=["xlsx", "csv"])

if uploaded:
    try:
        if uploaded.name.lower().endswith('.csv'):
            try:
                df_in = pd.read_csv(uploaded, sep=None, engine='python')
            except Exception:
                uploaded.seek(0)
                df_in = pd.read_csv(uploaded, delimiter=';', engine='python')
        else:
            df_in = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")
        st.stop()

    domains_set: Set[str] = set()
    for col in df_in.columns:
        for val in df_in[col].dropna().astype(str).tolist():
            d = normalize_domain(val)
            if d:
                domains_set.add(d)

    domains = sorted(domains_set)
    st.info(f"Обнаружено доменов: {len(domains)}")
    if len(domains) > 0:
        st.write("Примеры:", domains[:50])

    if st.button("Запустить сбор", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        all_results: List[CrawlResult] = []
        records: List[Dict[str, object]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_dom = {
                executor.submit(crawl_domain, dom, max_pages=max_pages, timeout=int(timeout)): dom
                for dom in domains
            }

            for i, future in enumerate(concurrent.futures.as_completed(future_to_dom), start=1):
                dom = future_to_dom[future]
                status.write(f"Обрабатываю {dom} ({i}/{len(domains)}) …")
                try:
                    res = future.result()
                    all_results.append(res)
                    for hit in res.page_hits:
                        records.append({
                            "Домен": dom,
                            "URL": hit["url"],
                            "Телефоны": ", ".join(sorted(hit["phones"])) if hit["phones"] else "",
                            "Почты": ", ".join(sorted(hit["emails"])) if hit["emails"] else "",
                            "ИНН": ", ".join(sorted(hit["inns"])) if hit["inns"] else "",
                        })
                except Exception as e:
                    records.append({
                        "Домен": dom,
                        "URL": "",
                        "Телефоны": "",
                        "Почты": "",
                        "ИНН": f"Ошибка: {e}",
                    })
                progress.progress(i / max(1, len(domains)))
                time.sleep(0.01)

        summary_rows = [r.aggregate() for r in all_results]
        df_summary = pd.DataFrame(summary_rows)
        df_records = pd.DataFrame(records)

        st.subheader("Сводная таблица")
        st.dataframe(df_summary, use_container_width=True)

        st.subheader("Подробные записи")
        st.dataframe(df_records, use_container_width=True)

        output_xlsx = io.BytesIO()
        with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
            df_summary.to_excel(writer, index=False, sheet_name="Сводка")
            df_records.to_excel(writer, index=False, sheet_name="Записи")
        output_xlsx.seek(0)

        st.download_button(
            label="💾 Скачать результат (output.xlsx)",
            data=output_xlsx,
            file_name="output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        output_zip = io.BytesIO()
        with zipfile.ZipFile(output_zip, mode='w') as zf:
            zf.writestr('summary.csv', df_summary.to_csv(index=False, encoding='utf-8-sig'))
            zf.writestr('records.csv', df_records.to_csv(index=False, encoding='utf-8-sig'))
        output_zip.seek(0)

        st.download_button(
            label="💾 Скачать CSV (ZIP)",
            data=output_zip,
            file_name="output_csvs.zip",
            mime="application/zip",
        )

        st.success("Готово! 🎉")
