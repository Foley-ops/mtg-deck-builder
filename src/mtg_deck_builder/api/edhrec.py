"""EDHREC API."""

import requests

from mtg_deck_builder.cache import RateLimiter
from mtg_deck_builder.db import Database


def edhrec_json(path, db: Database):
    cache_key = f"edhrec:{path}"
    cached = db.get(cache_key)
    if cached is not None:
        return cached
    url = f"https://json.edhrec.com/pages/{path}.json"
    try:
        RateLimiter.wait("edhrec", 0.3)
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        db.put(cache_key, data)
        return data
    except Exception as e:
        print(f"    [EDHREC] {path}: {e}")
        return {}


def edhrec_extract_cards(data):
    cards = []
    container = data.get("container", data)
    jd = container.get("json_dict", {})
    for section in jd.get("cardlists", []):
        for cv in section.get("cardviews", []):
            cards.append(
                {
                    "name": cv.get("name", ""),
                    "synergy": cv.get("synergy", 0),
                    "num_decks": cv.get("num_decks", 0),
                    "potential_decks": cv.get("potential_decks", 1),
                }
            )
    return cards
