"""Commander lookup and slug generation."""

import re
from dataclasses import dataclass

from mtg_deck_builder.cache import RateLimiter
from mtg_deck_builder.db import Database

import requests


@dataclass
class Commander:
    name: str
    color_identity: set[str]
    edhrec_slug: str
    color_filter: str
    cmc: float
    type_line: str
    scryfall_data: dict


def name_to_edhrec_slug(name: str) -> str:
    """'Ms. Bumbleflower' -> 'ms-bumbleflower', etc."""
    slug = name.lower()
    slug = slug.replace(",", "").replace("'", "").replace("\u2019", "").replace(".", "")
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug


def colors_to_filter(colors: set[str]) -> str:
    """{"W", "U", "G"} -> "wug" """
    order = "WUBRG"
    return "".join(c.lower() for c in order if c in colors)


def resolve_commander(name: str, db: Database) -> Commander:
    """Look up a commander on Scryfall (fuzzy match). Caches in db."""
    cached = db.get_commander(name)
    if cached is not None:
        return cached

    cache_key = f"scryfall:named:{name}"
    data = db.get(cache_key)

    if data is None:
        RateLimiter.wait("scryfall", 0.1)
        r = requests.get(
            "https://api.scryfall.com/cards/named",
            params={"fuzzy": name},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        db.put(cache_key, data)

    canonical_name = data.get("name", name)
    color_identity = set(data.get("color_identity", []))
    type_line = data.get("type_line", "")
    cmc = data.get("cmc", 0.0)

    if "Legendary" not in type_line:
        print(f"  WARNING: '{canonical_name}' is not Legendary. Are you sure it's a commander?")

    slug = name_to_edhrec_slug(canonical_name)
    color_filter = colors_to_filter(color_identity)

    commander = Commander(
        name=canonical_name,
        color_identity=color_identity,
        edhrec_slug=slug,
        color_filter=color_filter,
        cmc=cmc,
        type_line=type_line,
        scryfall_data=data,
    )

    # Persist to DB
    db.save_commander(commander)

    return commander
