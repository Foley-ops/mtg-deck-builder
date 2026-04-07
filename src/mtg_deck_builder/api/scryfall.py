"""Scryfall API."""

import hashlib
import json

import requests

from mtg_deck_builder.cache import RateLimiter
from mtg_deck_builder.db import Database


def _scryfall_get(url, db: Database, params=None):
    cache_key = f"scryfall:{url}:{json.dumps(params or {}, sort_keys=True)}"
    cached = db.get(cache_key)
    if cached is not None:
        return cached
    RateLimiter.wait("scryfall", 0.1)
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    db.put(cache_key, data)
    return data


def scryfall_search(query, db: Database, max_cards=250):
    """Paginated Scryfall search. Each page cached."""
    cards, url = [], "https://api.scryfall.com/cards/search"
    params = {"q": query, "order": "edhrec"}
    while url and len(cards) < max_cards:
        data = _scryfall_get(url, db, params)
        cards.extend(data.get("data", []))
        url = data.get("next_page")
        params = None
    return cards[:max_cards]


def scryfall_batch(card_names, db: Database):
    """Batch lookup via /cards/collection (75 cards/request)."""
    results = []
    unique = list(dict.fromkeys(card_names))
    chunks = [unique[i : i + 75] for i in range(0, len(unique), 75)]

    for chunk in chunks:
        cache_key = f"scryfall_batch:{hashlib.sha256(','.join(sorted(chunk)).encode()).hexdigest()[:16]}"
        cached = db.get(cache_key)
        if cached is not None:
            results.extend(cached)
            continue
        RateLimiter.wait("scryfall", 0.1)
        try:
            r = requests.post(
                "https://api.scryfall.com/cards/collection",
                json={"identifiers": [{"name": n} for n in chunk]},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            results.extend(data)
            db.put(cache_key, data)
        except Exception as e:
            print(f"    [Scryfall batch] chunk failed: {e}")
    return results


def fetch_game_changers(db: Database):
    """Fetch game changers from Scryfall (`is:gamechanger`)."""
    # Check DB first
    cached = db.get_game_changers()
    if cached is not None:
        print(f"  Game Changers from DB: {len(cached)}")
        return cached

    print("  Fetching current Game Changers from Scryfall...")
    try:
        results = scryfall_search("is:gamechanger", db, max_cards=200)
        names = {c["name"] for c in results}
        print(f"    -> {len(names)} Game Changers")
        db.save_game_changers(names)
        return names
    except Exception as e:
        print(f"    WARNING: Failed ({e}). Bracket GC checks disabled.")
        return set()
