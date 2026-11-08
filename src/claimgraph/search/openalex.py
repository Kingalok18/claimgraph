"""OpenAlex works search, for the optional online search."""
from __future__ import annotations

import httpx

from ..config import get_settings
from .cache import cache_get, cache_put

BASE = "https://api.openalex.org/works"


def abstract_from_index(index: dict | None) -> str | None:
    """OpenAlex stores abstracts as word -> positions."""
    if not index:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        words.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(words)) or None


def to_record(work: dict) -> dict:
    return {
        "source_id": work.get("id", ""),
        "title": work.get("title") or work.get("display_name") or "",
        "year": work.get("publication_year"),
        "doi": (work.get("doi") or "").replace("https://doi.org/", "") or None,
        "abstract": abstract_from_index(work.get("abstract_inverted_index")),
        "authors": ", ".join(
            (item.get("author") or {}).get("display_name", "") for item in (work.get("authorships") or [])[:8]
        ) or None,
        "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "citations": work.get("cited_by_count") or 0,
        "references": work.get("referenced_works") or [],
        "pdf_url": (work.get("open_access") or {}).get("oa_url"),
        "origin": "openalex",
    }


def _get(params: dict) -> dict:
    mailto = get_settings().openalex_mailto
    if mailto:
        params = {**params, "mailto": mailto}
    response = httpx.get(BASE, params=params, timeout=30, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def search_works(query: str, limit: int = 25) -> list[dict]:
    key = f"search:{query}:{limit}"
    cached = cache_get("openalex", key)
    if cached is not None:
        return cached
    data = _get({"search": query, "per-page": min(limit, 50)})
    records = [to_record(work) for work in data.get("results", [])]
    cache_put("openalex", key, records)
    return records


def fetch_by_ids(work_ids: list[str]) -> list[dict]:
    """Fetch works by OpenAlex id, used to follow a paper's references."""
    short = [work_id.rsplit("/", 1)[-1] for work_id in work_ids][:50]
    if not short:
        return []
    key = "ids:" + "|".join(sorted(short))
    cached = cache_get("openalex", key)
    if cached is not None:
        return cached
    data = _get({"filter": "openalex:" + "|".join(short), "per-page": 50})
    records = [to_record(work) for work in data.get("results", [])]
    cache_put("openalex", key, records)
    return records
