from __future__ import annotations
import time
from typing import Tuple
from urllib.parse import urlencode
import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

DEFAULT_HEADERS = {
    "User-Agent": "biomed-rag-agent/1.0 (contact: NCBI_EMAIL)",
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
}

def _http_get(url: str, params: dict, headers: dict) -> requests.Response:
    # NCBI prefers API key + email as params, not only headers
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 8))
def esearch(base: str, email: str, api_key: str, query: str) -> Tuple[int, str, str]:
    """
    Returns (count, webenv, query_key).
    Matches doc: db=pubmed, usehistory=y, retmax=0.
    """
    url = f"{base}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "usehistory": "y",
        "retmax": 0,
        "retmode": "xml",
        "api_key": api_key,
        "email": email,
    }
    r = _http_get(url, params, {**DEFAULT_HEADERS, "User-Agent": DEFAULT_HEADERS["User-Agent"].replace("NCBI_EMAIL", email)})
    xml = r.text

    # very light parsing to extract the three fields (works reliably for eutils XML)
    def _find(tag: str) -> str:
        start = xml.find(f"<{tag}>")
        if start < 0: return ""
        end = xml.find(f"</{tag}>", start)
        return xml[start + len(tag) + 2 : end].strip()

    count = int(_find("Count") or "0")
    webenv = _find("WebEnv")
    query_key = _find("QueryKey")
    if not webenv or not query_key:
        raise RuntimeError("ESearch did not return WebEnv/QueryKey. Query may be malformed.")
    return count, webenv, query_key

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 8))
def efetch_batch(base: str, email: str, api_key: str, webenv: str, query_key: str,
                 retstart: int, retmax: int = 10000) -> str:
    """
    Returns raw XML text for a batch of PubMedArticle records.
    Matches doc: retmode=xml, rettype=xml, batch of 10,000.
    """
    url = f"{base}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "xml",
        "rettype": "xml",
        "api_key": api_key,
        "email": email,
    }
    r = _http_get(url, params, {**DEFAULT_HEADERS, "User-Agent": DEFAULT_HEADERS["User-Agent"].replace("NCBI_EMAIL", email)})
    return r.text
