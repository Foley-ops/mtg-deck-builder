"""Selector tests -- bracket constraints, deck legality."""

import pytest
import torch
from collections import Counter
from unittest.mock import MagicMock

from mtg_deck_builder.card import Card
from mtg_deck_builder.brackets import BRACKETS
from mtg_deck_builder.selector import DeckSelector
from mtg_deck_builder.config import EMBED_DIM, HIDDEN_DIM, NUM_HEADS, GNN_LAYERS
from mtg_deck_builder.model import CardGNN, SynergyPredictor

from tests.conftest import make_card


class FakeGraph:
    """Minimal graph-like object for selector testing."""

    def __init__(self, cards, commander_name, combo_info=None, collection=None):
        self.cards = cards
        self.name_to_idx = {name: c.idx for name, c in cards.items()}
        self.combo_info = combo_info or {}
        self.collection = collection or {}
        self.game_changers = {n for n, c in cards.items() if c.is_game_changer}
        self.precon_cards = set()


def _build_selector(bant_commander, small_card_pool, small_graph_data, trained_model):
    """Wire up a DeckSelector from test fixtures."""
    graph = FakeGraph(small_card_pool, bant_commander.name)
    data, _ = small_graph_data
    return DeckSelector(bant_commander, graph, trained_model, data)


class TestDeckSize:
    def test_deck_has_99_cards(self, bant_commander, small_card_pool,
                                small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        for bracket in [1, 2, 3, 4]:
            deck = selector.select(bracket)
            assert len(deck) == 99, f"Bracket {bracket}: got {len(deck)} cards"


class TestCommanderNotIn99:
    def test_commander_excluded(self, bant_commander, small_card_pool,
                                 small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        for bracket in [1, 2, 3, 4]:
            deck = selector.select(bracket)
            deck_names = [c.name for c in deck]
            assert bant_commander.name not in deck_names


class TestColorIdentity:
    def test_no_off_color_cards(self, bant_commander, small_card_pool,
                                 small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        for bracket in [1, 2, 3, 4]:
            deck = selector.select(bracket)
            for card in deck:
                if card.color_identity and card.type_line != "Basic Land":
                    assert set(card.color_identity).issubset(bant_commander.color_identity), \
                        f"B{bracket}: {card.name} has {card.color_identity}"


class TestSingleton:
    def test_no_duplicate_nonbasics(self, bant_commander, small_card_pool,
                                     small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        for bracket in [1, 2, 3, 4]:
            deck = selector.select(bracket)
            nonbasics = [c.name for c in deck if c.type_line != "Basic Land"]
            assert len(nonbasics) == len(set(nonbasics)), \
                f"B{bracket}: duplicate nonbasics found"


class TestBracketGameChangers:
    def test_bracket_1_no_game_changers(self, bant_commander, small_card_pool,
                                         small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        deck = selector.select(1)
        gc = [c for c in deck if c.is_game_changer]
        assert len(gc) == 0

    def test_bracket_2_no_game_changers(self, bant_commander, small_card_pool,
                                         small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        deck = selector.select(2)
        gc = [c for c in deck if c.is_game_changer]
        assert len(gc) == 0

    def test_bracket_3_max_3_game_changers(self, bant_commander, small_card_pool,
                                            small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        deck = selector.select(3)
        gc = [c for c in deck if c.is_game_changer]
        assert len(gc) <= 3


class TestBracketMLD:
    def test_bracket_1_no_mld(self, bant_commander, small_card_pool,
                               small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        deck = selector.select(1)
        mld = [c for c in deck if c.is_mld]
        assert len(mld) == 0

    def test_bracket_4_allows_mld(self, bant_commander, small_card_pool,
                                   small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        deck = selector.select(4)
        # Bracket 4 allows MLD — just verify the deck is valid
        assert len(deck) == 99


class TestBracketExtraTurns:
    def test_bracket_1_limited_extra_turns(self, bant_commander, small_card_pool,
                                            small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        deck = selector.select(1)
        et = [c for c in deck if c.is_extra_turn]
        assert len(et) <= 1


class TestBasicLandDistribution:
    def test_basics_fill_remaining_slots(self, bant_commander, small_card_pool,
                                          small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        deck = selector.select(1)
        basics = [c for c in deck if c.type_line == "Basic Land"]
        nonbasics = [c for c in deck if c.type_line != "Basic Land"]
        assert len(basics) + len(nonbasics) == 99

    def test_basics_use_commander_colors(self, bant_commander, small_card_pool,
                                          small_graph_data, trained_model):
        selector = _build_selector(bant_commander, small_card_pool,
                                   small_graph_data, trained_model)
        deck = selector.select(1)
        basic_names = {c.name for c in deck if c.type_line == "Basic Land"}
        # Bant should use Plains, Island, Forest
        assert basic_names.issubset({"Plains", "Island", "Forest"})
