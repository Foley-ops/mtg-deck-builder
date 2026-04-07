"""Scryfall API."""

import hashlib
import json
import time

import requests

from mtg_deck_builder.cache import RateLimiter
from mtg_deck_builder.db import Database

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


def _request_with_retry(method, url, retries=MAX_RETRIES, **kwargs):
    """GET or POST with exponential backoff on timeout/5xx."""
    kwargs.setdefault("timeout", 30)
    for attempt in range(retries):
        try:
            RateLimiter.wait("scryfall", 0.15)
            r = method(url, **kwargs)
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"    [retry {attempt+1}/{retries}] {e.__class__.__name__}, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code >= 500 and attempt < retries - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"    [retry {attempt+1}/{retries}] HTTP {e.response.status_code}, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def _scryfall_get(url, db: Database, params=None):
    cache_key = f"scryfall:{url}:{json.dumps(params or {}, sort_keys=True)}"
    cached = db.get(cache_key)
    if cached is not None:
        return cached
    r = _request_with_retry(requests.get, url, params=params)
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

    for i, chunk in enumerate(chunks):
        cache_key = f"scryfall_batch:{hashlib.sha256(','.join(sorted(chunk)).encode()).hexdigest()[:16]}"
        cached = db.get(cache_key)
        if cached is not None:
            results.extend(cached)
            continue
        try:
            r = _request_with_retry(
                requests.post,
                "https://api.scryfall.com/cards/collection",
                json={"identifiers": [{"name": n} for n in chunk]},
                timeout=45,
            )
            data = r.json().get("data", [])
            results.extend(data)
            db.put(cache_key, data)
        except Exception as e:
            print(f"    [Scryfall batch {i+1}/{len(chunks)}] failed after retries: {e}")
    return results


def fetch_game_changers(db: Database):
    """Fetch game changers from Scryfall (`is:gamechanger`)."""
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
