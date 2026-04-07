"""SQLite storage layer."""

import hashlib
import json
import sqlite3
import struct
import time
from pathlib import Path
from typing import Optional

import torch


SCHEMA_VERSION = 1

# Different TTLs for different data freshness needs
TTL_CARD_DATA = 7 * 24 * 3600       # 7 days -oracle text rarely changes
TTL_SYNERGY = 24 * 3600             # 24 hours -EDHREC synergy shifts
TTL_COMBOS = 24 * 3600              # 24 hours
TTL_COMMANDER = 30 * 24 * 3600      # 30 days -commander identity doesn't change
TTL_GAME_CHANGERS = 24 * 3600       # 24 hours -WotC can update the list
TTL_API_CACHE = 24 * 3600           # 24 hours -generic fallback


def _default_db_path():
    p = Path.home() / ".mtg_deck_builder"
    p.mkdir(exist_ok=True)
    return p / "deck_builder.db"


class Database:
    """Wraps a single SQLite file for all persistent data."""

    def __init__(self, db_path=None, force_refresh=False):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.force_refresh = force_refresh
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        # Counters for cache hit/miss stats
        self.hits = 0
        self.misses = 0

    def close(self):
        self.conn.close()

    def _init_schema(self):
        cur = self.conn.execute("PRAGMA user_version")
        version = cur.fetchone()[0]
        if version < SCHEMA_VERSION:
            self._create_tables()
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key   TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                fetched_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cards (
                name            TEXT PRIMARY KEY,
                cmc             REAL DEFAULT 0.0,
                type_line       TEXT DEFAULT '',
                oracle_text     TEXT DEFAULT '',
                colors          TEXT DEFAULT '[]',
                color_identity  TEXT DEFAULT '[]',
                edhrec_rank     INTEGER,
                price_usd       REAL,
                scryfall_data   TEXT,
                updated_at      REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS commanders (
                name            TEXT PRIMARY KEY,
                color_identity  TEXT NOT NULL,
                edhrec_slug     TEXT NOT NULL,
                color_filter    TEXT NOT NULL,
                cmc             REAL NOT NULL,
                type_line       TEXT NOT NULL,
                scryfall_data   TEXT NOT NULL,
                resolved_at     REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS synergies (
                commander_name  TEXT NOT NULL,
                card_name       TEXT NOT NULL,
                synergy_score   REAL DEFAULT 0.0,
                num_decks       INTEGER DEFAULT 0,
                potential_decks INTEGER DEFAULT 1,
                source_slug     TEXT NOT NULL,
                fetched_at      REAL NOT NULL,
                PRIMARY KEY (commander_name, card_name, source_slug)
            );

            CREATE TABLE IF NOT EXISTS combos (
                combo_id        TEXT PRIMARY KEY,
                description     TEXT DEFAULT '',
                card_names      TEXT NOT NULL,
                card_count      INTEGER NOT NULL,
                color_identity  TEXT DEFAULT '',
                fetched_at      REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS combo_cards (
                combo_id        TEXT NOT NULL,
                card_name       TEXT NOT NULL,
                PRIMARY KEY (combo_id, card_name),
                FOREIGN KEY (combo_id) REFERENCES combos(combo_id)
            );

            CREATE TABLE IF NOT EXISTS collection (
                name            TEXT PRIMARY KEY,
                quantity        INTEGER NOT NULL DEFAULT 1,
                foil            INTEGER DEFAULT 0,
                condition       TEXT DEFAULT 'NM',
                set_code        TEXT DEFAULT '',
                binder          TEXT DEFAULT '',
                imported_at     REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                commander_name  TEXT NOT NULL,
                card_name       TEXT NOT NULL,
                embedding       BLOB NOT NULL,
                train_epochs    INTEGER NOT NULL,
                model_hash      TEXT NOT NULL,
                created_at      REAL NOT NULL,
                PRIMARY KEY (commander_name, card_name)
            );

            CREATE TABLE IF NOT EXISTS game_changers (
                name            TEXT PRIMARY KEY,
                fetched_at      REAL NOT NULL
            );
        """)
        self.conn.commit()

    def _expired(self, fetched_at, ttl):
        if self.force_refresh:
            return True
        return (time.time() - fetched_at) > ttl

    # ----------------------------------------------------------------
    # Generic API cache (DiskCache drop-in replacement)
    # ----------------------------------------------------------------

    def get(self, key):
        """Alias for cache_get."""
        return self.cache_get(key)

    def put(self, key, data):
        """Alias for cache_put."""
        self.cache_put(key, data)

    def stats(self):
        """Alias for cache_stats."""
        return self.cache_stats()

    def cache_get(self, key: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT data, fetched_at FROM api_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None or self._expired(row["fetched_at"], TTL_API_CACHE):
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row["data"])

    def cache_put(self, key: str, data):
        self.conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, data, fetched_at) VALUES (?, ?, ?)",
            (key, json.dumps(data), time.time()),
        )
        self.conn.commit()

    def cache_stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM api_cache"
        ).fetchone()
        # Approximate DB size
        page_count = self.conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = self.conn.execute("PRAGMA page_size").fetchone()[0]
        size_mb = (page_count * page_size) / 1e6
        return {"entries": row["cnt"], "size_mb": size_mb}

    # ----------------------------------------------------------------
    # Cards
    # ----------------------------------------------------------------

    def upsert_card(self, name: str, scryfall_card: dict):
        self.conn.execute(
            """INSERT OR REPLACE INTO cards
               (name, cmc, type_line, oracle_text, colors, color_identity,
                edhrec_rank, price_usd, scryfall_data, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                scryfall_card.get("cmc", 0.0),
                scryfall_card.get("type_line", ""),
                scryfall_card.get("oracle_text", ""),
                json.dumps(scryfall_card.get("colors", [])),
                json.dumps(scryfall_card.get("color_identity", [])),
                scryfall_card.get("edhrec_rank"),
                _extract_price(scryfall_card),
                json.dumps(scryfall_card),
                time.time(),
            ),
        )
        self.conn.commit()

    def upsert_cards_batch(self, scryfall_cards: list[dict]):
        now = time.time()
        rows = []
        for sc in scryfall_cards:
            name = sc.get("name", "")
            if not name or "//" in name:
                continue
            rows.append((
                name,
                sc.get("cmc", 0.0),
                sc.get("type_line", ""),
                sc.get("oracle_text", ""),
                json.dumps(sc.get("colors", [])),
                json.dumps(sc.get("color_identity", [])),
                sc.get("edhrec_rank"),
                _extract_price(sc),
                json.dumps(sc),
                now,
            ))
        self.conn.executemany(
            """INSERT OR REPLACE INTO cards
               (name, cmc, type_line, oracle_text, colors, color_identity,
                edhrec_rank, price_usd, scryfall_data, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def get_card(self, name: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM cards WHERE name = ?", (name,)
        ).fetchone()
        if row is None or self._expired(row["updated_at"], TTL_CARD_DATA):
            return None
        return dict(row)

    def get_cards_needing_refresh(self, names: list[str]) -> list[str]:
        """Which of these names need a fresh Scryfall fetch?"""
        if not names:
            return []
        need = []
        # Check in batches to avoid SQLite variable limits
        for i in range(0, len(names), 500):
            batch = names[i:i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT name, updated_at FROM cards WHERE name IN ({placeholders})",
                batch,
            ).fetchall()
            found = {r["name"]: r["updated_at"] for r in rows}
            for name in batch:
                if name not in found or self._expired(found[name], TTL_CARD_DATA):
                    need.append(name)
        return need

    # ----------------------------------------------------------------
    # Commanders
    # ----------------------------------------------------------------

    def save_commander(self, commander):
        from mtg_deck_builder.commander import Commander
        self.conn.execute(
            """INSERT OR REPLACE INTO commanders
               (name, color_identity, edhrec_slug, color_filter, cmc, type_line,
                scryfall_data, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                commander.name,
                json.dumps(sorted(commander.color_identity)),
                commander.edhrec_slug,
                commander.color_filter,
                commander.cmc,
                commander.type_line,
                json.dumps(commander.scryfall_data),
                time.time(),
            ),
        )
        self.conn.commit()

    def get_commander(self, name: str):
        from mtg_deck_builder.commander import Commander
        row = self.conn.execute(
            "SELECT * FROM commanders WHERE name = ?", (name,)
        ).fetchone()
        if row is None or self._expired(row["resolved_at"], TTL_COMMANDER):
            return None
        return Commander(
            name=row["name"],
            color_identity=set(json.loads(row["color_identity"])),
            edhrec_slug=row["edhrec_slug"],
            color_filter=row["color_filter"],
            cmc=row["cmc"],
            type_line=row["type_line"],
            scryfall_data=json.loads(row["scryfall_data"]),
        )

    # ----------------------------------------------------------------
    # Synergies
    # ----------------------------------------------------------------

    def save_synergies(self, commander_name: str, cards: list[dict], source_slug: str):
        now = time.time()
        rows = [
            (commander_name, c["name"], c.get("synergy", 0), c.get("num_decks", 0),
             c.get("potential_decks", 1), source_slug, now)
            for c in cards if c.get("name")
        ]
        self.conn.executemany(
            """INSERT OR REPLACE INTO synergies
               (commander_name, card_name, synergy_score, num_decks,
                potential_decks, source_slug, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def get_synergies(self, commander_name: str) -> Optional[list[dict]]:
        rows = self.conn.execute(
            "SELECT * FROM synergies WHERE commander_name = ?", (commander_name,)
        ).fetchall()
        if not rows:
            return None
        # Check if any source is expired
        if any(self._expired(r["fetched_at"], TTL_SYNERGY) for r in rows):
            return None
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------
    # Combos
    # ----------------------------------------------------------------

    def save_combos(self, combos: list[dict], color_identity: str):
        now = time.time()
        for combo in combos:
            cid = combo.get("id", "")
            names = [u.get("card", {}).get("name", "") for u in combo.get("uses", [])]
            names = [n for n in names if n]
            self.conn.execute(
                """INSERT OR REPLACE INTO combos
                   (combo_id, description, card_names, card_count, color_identity, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cid, combo.get("description", ""), json.dumps(names),
                 len(names), color_identity, now),
            )
            for name in names:
                self.conn.execute(
                    "INSERT OR IGNORE INTO combo_cards (combo_id, card_name) VALUES (?, ?)",
                    (cid, name),
                )
        self.conn.commit()

    def get_combos(self, color_identity: str) -> Optional[list[dict]]:
        rows = self.conn.execute(
            "SELECT * FROM combos WHERE color_identity = ?", (color_identity,)
        ).fetchall()
        if not rows:
            return None
        if any(self._expired(r["fetched_at"], TTL_COMBOS) for r in rows):
            return None
        return [
            {"id": r["combo_id"], "cards": json.loads(r["card_names"]),
             "size": r["card_count"], "description": r["description"]}
            for r in rows
        ]

    # ----------------------------------------------------------------
    # Collection
    # ----------------------------------------------------------------

    def import_collection(self, collection: dict[str, int]):
        """Wipe and re-import. {name: quantity}."""
        now = time.time()
        with self.conn:
            self.conn.execute("DELETE FROM collection")
            self.conn.executemany(
                """INSERT INTO collection (name, quantity, imported_at)
                   VALUES (?, ?, ?)""",
                [(name, qty, now) for name, qty in collection.items()],
            )

    def get_collection(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT name, quantity FROM collection").fetchall()
        return {r["name"]: r["quantity"] for r in rows}

    def collection_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) as cnt FROM collection").fetchone()["cnt"]

    def clear_collection(self):
        self.conn.execute("DELETE FROM collection")
        self.conn.commit()

    # ----------------------------------------------------------------
    # Game Changers
    # ----------------------------------------------------------------

    def save_game_changers(self, names: set[str]):
        now = time.time()
        with self.conn:
            self.conn.execute("DELETE FROM game_changers")
            self.conn.executemany(
                "INSERT INTO game_changers (name, fetched_at) VALUES (?, ?)",
                [(n, now) for n in names],
            )

    def get_game_changers(self) -> Optional[set[str]]:
        rows = self.conn.execute("SELECT name, fetched_at FROM game_changers").fetchall()
        if not rows:
            return None
        if any(self._expired(r["fetched_at"], TTL_GAME_CHANGERS) for r in rows):
            return None
        return {r["name"] for r in rows}

    # ----------------------------------------------------------------
    # Embeddings
    # ----------------------------------------------------------------

    def save_embeddings(self, commander_name: str,
                        card_embeddings: dict[str, torch.Tensor],
                        train_epochs: int, model_hash: str):
        now = time.time()
        rows = []
        for card_name, emb in card_embeddings.items():
            blob = emb.cpu().numpy().tobytes()
            rows.append((commander_name, card_name, blob, train_epochs, model_hash, now))
        with self.conn:
            self.conn.execute(
                "DELETE FROM embeddings WHERE commander_name = ?",
                (commander_name,),
            )
            self.conn.executemany(
                """INSERT INTO embeddings
                   (commander_name, card_name, embedding, train_epochs, model_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def get_embeddings(self, commander_name: str,
                       model_hash: str) -> Optional[dict[str, torch.Tensor]]:
        rows = self.conn.execute(
            """SELECT card_name, embedding FROM embeddings
               WHERE commander_name = ? AND model_hash = ?""",
            (commander_name, model_hash),
        ).fetchall()
        if not rows:
            return None
        result = {}
        for r in rows:
            tensor = torch.frombuffer(bytearray(r["embedding"]), dtype=torch.float32).clone()
            result[r["card_name"]] = tensor
        return result

    # ----------------------------------------------------------------
    # Maintenance
    # ----------------------------------------------------------------

    def clear_all(self):
        """Wipe all tables."""
        for table in ["api_cache", "cards", "commanders", "synergies",
                      "combos", "combo_cards", "collection", "embeddings",
                      "game_changers"]:
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()


def _extract_price(sc: dict) -> Optional[float]:
    p = sc.get("prices", {})
    if p.get("usd"):
        try:
            return float(p["usd"])
        except (ValueError, TypeError):
            return None
    return None


def compute_model_hash(embed_dim, hidden_dim, num_heads, gnn_layers, epochs, card_count):
    """Hash of model config for cache invalidation."""
    s = f"{embed_dim}:{hidden_dim}:{num_heads}:{gnn_layers}:{epochs}:{card_count}"
    return hashlib.sha256(s.encode()).hexdigest()[:12]
