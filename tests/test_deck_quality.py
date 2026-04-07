"""Integration tests -- full pipeline deck legality checks."""

import pytest
from collections import Counter

from mtg_deck_builder.card import Card
from mtg_deck_builder.brackets import BRACKETS
from mtg_deck_builder.selector import DeckSelector
from mtg_deck_builder.config import EMBED_DIM, HIDDEN_DIM, NUM_HEADS, GNN_LAYERS
from mtg_deck_builder.model import CardGNN, SynergyPredictor, train_gnn


class FakeGraph:
    def __init__(self, cards, commander_name, combo_info=None, collection=None):
        self.cards = cards
        self.name_to_idx = {name: c.idx for name, c in cards.items()}
        self.combo_info = combo_info or {}
        self.collection = collection or {}
        self.game_changers = {n for n, c in cards.items() if c.is_game_changer}
        self.precon_cards = set()


class TestDeckQualityAllBrackets:
    """Run the full pipeline on a small graph and verify all invariants."""

    @pytest.fixture(autouse=True)
    def setup_decks(self, bant_commander, small_card_pool, small_graph_data):
        """Train model and generate decks for all 4 brackets."""
        data, ew = small_graph_data
        gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
        model = SynergyPredictor(gnn)
        train_gnn(model, data, ew, epochs=20, lr=1e-3)

        graph = FakeGraph(small_card_pool, bant_commander.name)
        selector = DeckSelector(bant_commander, graph, model, data)

        self.commander = bant_commander
        self.decks = {}
        for b in [1, 2, 3, 4]:
            self.decks[b] = selector.select(b)

    def test_all_decks_have_99_cards(self):
        for b, deck in self.decks.items():
            assert len(deck) == 99, f"Bracket {b}: {len(deck)} cards"

    def test_no_commander_in_any_deck(self):
        for b, deck in self.decks.items():
            names = [c.name for c in deck]
            assert self.commander.name not in names, f"Bracket {b}"

    def test_no_color_violations_in_any_deck(self):
        for b, deck in self.decks.items():
            for card in deck:
                if card.color_identity and card.type_line != "Basic Land":
                    assert set(card.color_identity).issubset(self.commander.color_identity), \
                        f"B{b}: {card.name} color identity {card.color_identity}"

    def test_singleton_enforced_in_all_decks(self):
        for b, deck in self.decks.items():
            nonbasics = [c.name for c in deck if c.type_line != "Basic Land"]
            dupes = [n for n, ct in Counter(nonbasics).items() if ct > 1]
            assert not dupes, f"B{b}: duplicates {dupes}"

    def test_game_changer_limits(self):
        for b, deck in self.decks.items():
            gc = [c for c in deck if c.is_game_changer]
            limit = BRACKETS[b].max_game_changers
            assert len(gc) <= limit, \
                f"B{b}: {len(gc)} game changers (limit {limit})"

    def test_mld_restrictions(self):
        for b in [1, 2, 3]:
            deck = self.decks[b]
            mld = [c for c in deck if c.is_mld]
            assert len(mld) == 0, f"B{b}: found {len(mld)} MLD cards"

    def test_extra_turn_restrictions(self):
        for b in [1, 2, 3]:
            deck = self.decks[b]
            et = [c for c in deck if c.is_extra_turn]
            assert len(et) <= 1, f"B{b}: {len(et)} extra turn cards"

    def test_higher_brackets_are_at_least_as_powerful(self):
        """Higher brackets should allow more nonbasic cards (fewer basics needed)."""
        basic_counts = {}
        for b, deck in self.decks.items():
            basic_counts[b] = sum(1 for c in deck if c.type_line == "Basic Land")
        # B4 should have <= basics than B1 (more nonbasics available)
        assert basic_counts[4] <= basic_counts[1]

    def test_basics_only_in_commander_colors(self):
        valid_basics = {"Plains", "Island", "Forest"}  # Bant
        for b, deck in self.decks.items():
            for card in deck:
                if card.type_line == "Basic Land":
                    assert card.name in valid_basics, \
                        f"B{b}: unexpected basic {card.name}"

    def test_prefer_owned_increases_owned_count(self, bant_commander,
                                                  small_card_pool, small_graph_data):
        """Decks with --prefer-owned should include more owned cards."""
        data, ew = small_graph_data
        gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
        model = SynergyPredictor(gnn)
        train_gnn(model, data, ew, epochs=20, lr=1e-3)

        collection = {"Normal0": 2, "Normal1": 1}
        graph = FakeGraph(small_card_pool, bant_commander.name, collection=collection)
        selector = DeckSelector(bant_commander, graph, model, data)

        deck_normal = selector.select(3, prefer_owned=False)
        deck_owned = selector.select(3, prefer_owned=True)

        owned_normal = sum(1 for c in deck_normal if c.owned_qty > 0)
        owned_boosted = sum(1 for c in deck_owned if c.owned_qty > 0)
        # With prefer_owned, should have at least as many owned cards
        assert owned_boosted >= owned_normal
