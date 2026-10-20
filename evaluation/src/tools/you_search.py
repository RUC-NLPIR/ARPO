"""Optional You.com search backend for the evaluation agents.

Opt-in with ``--search_provider you``. Works without an API key:

* No key: ``GET https://api.you.com/v1/agents/search`` (free tier, 100
  searches a day, counted per IP).
* ``--ydc_api_key`` or ``YDC_API_KEY`` set: ``POST https://ydc-index.io/v1/search``
  with ``X-API-Key``, for higher limits.

Results are mapped to the same ``{"organic": [...]}`` shape the Bright Data
Bing zone returns, so the existing post-processing, caching and the SDS page
fetch/summarize path work unchanged.
"""

import os
import time
from typing import Dict, List, Optional

import requests

KEYLESS_URL = "https://api.you.com/v1/agents/search"
KEYED_URL = "https://ydc-index.io/v1/search"
USER_AGENT = "youdotcom-integration/ruc-nlpir-arpo"


def _to_organic(data: Dict) -> Dict:
    web = ((data or {}).get("results") or {}).get("web") or []
    organic: List[Dict] = []
    for item in web:
        snippets = [s for s in (item.get("snippets") or []) if s]
        description = item.get("description") or ""
        if snippets:
            joined = " ".join(snippets)
            description = f"{description} {joined}".strip() if description else joined
        organic.append(
            {
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "description": description,
                "datePublished": item.get("page_age") or "",
                "site_name": "",
            }
        )
    # "general" is kept so PreprocessCacheManager can store the result like a Bing one.
    return {"general": {}, "organic": organic}


def you_search_request(
    query: str,
    count: int = 10,
    api_key: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    session=requests,
) -> Optional[Dict]:
    """Run one You.com search. Returns ``{"organic": [...]}`` or ``None`` on failure."""
    api_key = api_key or os.environ.get("YDC_API_KEY") or None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(1, max_retries + 1):
        try:
            if api_key:
                headers["X-API-Key"] = api_key
                resp = session.post(
                    KEYED_URL,
                    headers=headers,
                    json={"query": query, "count": count},
                    timeout=timeout,
                )
            else:
                resp = session.get(
                    KEYLESS_URL,
                    headers=headers,
                    params={"query": query, "count": count},
                    timeout=timeout,
                )
        except requests.exceptions.RequestException as e:
            print(f"You.com search error (attempt {attempt}) for query: {query}: {e}")
            time.sleep(retry_delay)
            continue

        status = resp.status_code
        if status == 200:
            return _to_organic(resp.json())
        if status in (401, 403):
            print(f"You.com search: API key rejected (HTTP {status}). Check YDC_API_KEY.")
            return None
        if status == 402:
            print("You.com search: out of credits (HTTP 402).")
            return None
        if status == 429:
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else retry_delay * attempt
            except ValueError:
                wait = retry_delay * attempt
            hint = "" if api_key else " The keyless tier allows 100 searches a day; set YDC_API_KEY for more."
            print(f"You.com search: rate limited, retrying in {wait}s.{hint}")
            time.sleep(wait)
            continue
        print(f"You.com search: HTTP {status} (attempt {attempt}) for query: {query}")
        time.sleep(retry_delay)
    return None
