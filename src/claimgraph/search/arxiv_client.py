"""arXiv search through its Atom API, for the optional online search."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from .cache import cache_get, cache_put

BASE = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _text(node, path: str) -> str | None:
    found = node.find(path, _NS)
    return " ".join(found.text.split()) if found is not None and found.text else None


def parse_feed(xml_text: str) -> list[dict]:
    records = []
    for entry in ET.fromstring(xml_text).findall("atom:entry", _NS):
        published = _text(entry, "atom:published") or ""
        year = re.match(r"\d{4}", published)
        pdf = next((link.get("href") for link in entry.findall("atom:link", _NS) if link.get("title") == "pdf"), None)
        records.append(
            {
                "source_id": _text(entry, "atom:id") or "",
                "title": _text(entry, "atom:title") or "",
                "year": int(year.group(0)) if year else None,
                "doi": _text(entry, "arxiv:doi"),
                "abstract": _text(entry, "atom:summary"),
                "authors": ", ".join(filter(None, (_text(item, "atom:name") for item in entry.findall("atom:author", _NS))))[:500] or None,
                "venue": "arXiv",
                "citations": 0,
                "references": [],
                "pdf_url": pdf,
                "origin": "arxiv",
            }
        )
    return records


def search_arxiv(query: str, limit: int = 25) -> list[dict]:
    key = f"search:{query}:{limit}"
    cached = cache_get("arxiv", key)
    if cached is not None:
        return cached
    response = httpx.get(
        BASE, params={"search_query": f"all:{query}", "start": 0, "max_results": min(limit, 50)}, timeout=30, follow_redirects=True
    )
    response.raise_for_status()
    records = parse_feed(response.text)
    cache_put("arxiv", key, records)
    return records
