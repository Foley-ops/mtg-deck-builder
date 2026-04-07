"""Database tests."""

import tempfile
import os

import pytest
import torch

from mtg_deck_builder.db import Database, compute_model_hash
from mtg_deck_builder.commander import Commander


@pytest.fixture
def db(tmp_path):
    """Fresh database in a temp directory."""
    db_path = tmp_path / "test.db"
    d = Database(db_path=db_path)
    yield d
    d.close()


class TestApiCache:
    def test_put_and_get(self, db):
        db.put("key1", {"foo": "bar"})
        assert db.get("key1") == {"foo": "bar"}

    def test_get_missing_returns_none(self, db):
        assert db.get("nonexistent") is None

    def test_overwrite(self, db):
        db.put("key1", {"v": 1})
        db.put("key1", {"v": 2})
        assert db.get("key1") == {"v": 2}

    def test_stats(self, db):
        db.put("k1", {"a": 1})
        db.put("k2", {"b": 2})
        stats = db.stats()
        assert stats["entries"] == 2
        assert stats["size_mb"] >= 0

    def test_hit_miss_counters(self, db):
        db.put("k1", {"a": 1})
        db.get("k1")
        db.get("missing")
        assert db.hits == 1
        assert db.misses == 1


class TestForceRefresh:
    def test_force_refresh_bypasses_cache(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = Database(db_path=db_path)
        db.put("key1", {"foo": "bar"})
        db.close()

        db2 = Database(db_path=db_path, force_refresh=True)
        assert db2.get("key1") is None
        db2.close()


class TestCards:
    def test_upsert_and_get(self, db):
        sc = {"name": "Sol Ring", "cmc": 1.0, "type_line": "Artifact",
              "oracle_text": "{T}: Add {C}{C}.", "colors": [],
              "color_identity": [], "prices": {"usd": "2.00"}}
        db.upsert_card("Sol Ring", sc)
        card = db.get_card("Sol Ring")
        assert card is not None
        assert card["cmc"] == 1.0
        assert card["price_usd"] == 2.0

    def test_get_missing_card(self, db):
        assert db.get_card("Nonexistent Card") is None

    def test_batch_upsert(self, db):
        cards = [
            {"name": "Sol Ring", "cmc": 1.0, "type_line": "Artifact"},
            {"name": "Forest", "cmc": 0.0, "type_line": "Basic Land"},
        ]
        db.upsert_cards_batch(cards)
        assert db.get_card("Sol Ring") is not None
        assert db.get_card("Forest") is not None

    def test_cards_needing_refresh(self, db):
        db.upsert_card("Sol Ring", {"name": "Sol Ring"})
        need = db.get_cards_needing_refresh(["Sol Ring", "Missing Card"])
        assert "Missing Card" in need
        assert "Sol Ring" not in need


class TestCommanders:
    def test_save_and_get(self, db):
        cmd = Commander(
            name="Ms. Bumbleflower",
            color_identity={"W", "U", "G"},
            edhrec_slug="ms-bumbleflower",
            color_filter="wug",
            cmc=4.0,
            type_line="Legendary Creature",
            scryfall_data={"name": "Ms. Bumbleflower"},
        )
        db.save_commander(cmd)
        loaded = db.get_commander("Ms. Bumbleflower")
        assert loaded is not None
        assert loaded.name == "Ms. Bumbleflower"
        assert loaded.color_identity == {"W", "U", "G"}
        assert loaded.edhrec_slug == "ms-bumbleflower"

    def test_get_missing_commander(self, db):
        assert db.get_commander("Nobody") is None


class TestCollection:
    def test_import_and_get(self, db):
        db.import_collection({"Sol Ring": 1, "Forest": 10})
        coll = db.get_collection()
        assert coll["Sol Ring"] == 1
        assert coll["Forest"] == 10

    def test_import_replaces(self, db):
        db.import_collection({"Sol Ring": 1})
        db.import_collection({"Forest": 5})
        coll = db.get_collection()
        assert "Sol Ring" not in coll
        assert coll["Forest"] == 5

    def test_clear_collection(self, db):
        db.import_collection({"Sol Ring": 1})
        db.clear_collection()
        assert db.get_collection() == {}

    def test_collection_count(self, db):
        db.import_collection({"Sol Ring": 1, "Forest": 10, "Island": 5})
        assert db.collection_count() == 3


class TestGameChangers:
    def test_save_and_get(self, db):
        names = {"Rhystic Study", "Smothering Tithe", "Cyclonic Rift"}
        db.save_game_changers(names)
        loaded = db.get_game_changers()
        assert loaded == names

    def test_get_empty(self, db):
        assert db.get_game_changers() is None


class TestEmbeddings:
    def test_save_and_get(self, db):
        embs = {
            "Sol Ring": torch.randn(64),
            "Forest": torch.randn(64),
        }
        db.save_embeddings("Ms. Bumbleflower", embs, 200, "abc123")
        loaded = db.get_embeddings("Ms. Bumbleflower", "abc123")
        assert loaded is not None
        assert len(loaded) == 2
        assert torch.allclose(loaded["Sol Ring"], embs["Sol Ring"], atol=1e-6)

    def test_get_wrong_hash_returns_none(self, db):
        embs = {"Sol Ring": torch.randn(64)}
        db.save_embeddings("Ms. Bumbleflower", embs, 200, "abc123")
        assert db.get_embeddings("Ms. Bumbleflower", "different") is None

    def test_get_missing_returns_none(self, db):
        assert db.get_embeddings("Nobody", "abc") is None


class TestModelHash:
    def test_deterministic(self):
        h1 = compute_model_hash(64, 128, 4, 3, 200, 500)
        h2 = compute_model_hash(64, 128, 4, 3, 200, 500)
        assert h1 == h2

    def test_changes_with_params(self):
        h1 = compute_model_hash(64, 128, 4, 3, 200, 500)
        h2 = compute_model_hash(64, 128, 4, 3, 100, 500)
        assert h1 != h2


class TestClearAll:
    def test_clear_all(self, db):
        db.put("key", {"data": 1})
        db.import_collection({"Sol Ring": 1})
        db.save_game_changers({"Rhystic Study"})
        db.clear_all()
        assert db.get("key") is None
        assert db.get_collection() == {}
        assert db.get_game_changers() is None
