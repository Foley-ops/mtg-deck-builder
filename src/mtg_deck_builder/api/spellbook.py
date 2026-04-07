"""Commander Spellbook API."""

import requests

from mtg_deck_builder.cache import RateLimiter
from mtg_deck_builder.db import Database


def spellbook_search(query, db: Database, limit=100):
    cache_key = f"spellbook:{query}:{limit}"
    cached = db.get(cache_key)
    if cached is not None:
        return cached
    url = "https://backend.commanderspellbook.com/variants/"
    try:
        RateLimiter.wait("spellbook", 0.2)
        r = requests.get(url, params={"q": query, "limit": limit}, timeout=15)
        r.raise_for_status()
        data = r.json().get("results", [])
        db.put(cache_key, data)
        return data
    except Exception as e:
        print(f"    [Spellbook] {e}")
        return []
