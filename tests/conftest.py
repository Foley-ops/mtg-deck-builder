"""Test fixtures."""

import pytest
import torch
from torch_geometric.data import Data

from mtg_deck_builder.card import Card
from mtg_deck_builder.commander import Commander
from mtg_deck_builder.config import EMBED_DIM, HIDDEN_DIM, NUM_HEADS, GNN_LAYERS
from mtg_deck_builder.model import CardGNN, SynergyPredictor, train_gnn


def make_card(name, idx=-1, **kwargs):
    """Factory for test cards with sensible defaults."""
    defaults = dict(
        cmc=3.0,
        type_line="Creature",
        color_identity=["W", "U", "G"],
        synergy_score=0.5,
        inclusion_rate=30.0,
    )
    defaults.update(kwargs)
    return Card(name=name, idx=idx, **defaults)


@pytest.fixture
def bant_commander():
    return Commander(
        name="Ms. Bumbleflower",
        color_identity={"W", "U", "G"},
        edhrec_slug="ms-bumbleflower",
        color_filter="wug",
        cmc=4.0,
        type_line="Legendary Creature — Faerie Noble",
        scryfall_data={},
    )


@pytest.fixture
def five_color_commander():
    return Commander(
        name="Kenrith, the Returned King",
        color_identity={"W", "U", "B", "R", "G"},
        edhrec_slug="kenrith-the-returned-king",
        color_filter="wubrg",
        cmc=5.0,
        type_line="Legendary Creature — Human Noble",
        scryfall_data={},
    )


@pytest.fixture
def small_card_pool():
    """30 cards with varied properties for testing deck selection."""
    cards = {}

    # Commander
    cmd = make_card("Ms. Bumbleflower", idx=0, cmc=4.0,
                    type_line="Legendary Creature", synergy_score=0.0)
    cards[cmd.name] = cmd

    # 5 game changers
    for i in range(5):
        c = make_card(f"GameChanger{i}", idx=i + 1,
                      is_game_changer=True, synergy_score=0.8 - i * 0.1)
        cards[c.name] = c

    # 3 extra turn cards
    for i in range(3):
        c = make_card(f"ExtraTurn{i}", idx=i + 6,
                      is_extra_turn=True, synergy_score=0.6,
                      oracle_text="Take an extra turn after this one.")
        cards[c.name] = c

    # 2 MLD cards
    for i in range(2):
        c = make_card(f"MLD{i}", idx=i + 9,
                      is_mld=True, synergy_score=0.4,
                      oracle_text="Destroy all lands.")
        cards[c.name] = c

    # 2 off-color cards (should never be selected for Bant)
    for i in range(2):
        c = make_card(f"OffColor{i}", idx=i + 11,
                      color_identity=["B", "R"], synergy_score=0.9)
        cards[c.name] = c

    # 16 normal cards
    for i in range(16):
        c = make_card(f"Normal{i}", idx=i + 13,
                      synergy_score=0.5 - i * 0.02,
                      type_line=["Creature", "Instant", "Sorcery", "Enchantment"][i % 4])
        cards[c.name] = c

    # 2 owned cards
    cards["Normal0"].owned_qty = 2
    cards["Normal1"].owned_qty = 1

    return cards


@pytest.fixture
def small_graph_data(small_card_pool):
    """Build a simple PyG Data object from the small card pool."""
    n = len(small_card_pool)
    x = torch.zeros(n, Card.NODE_FEAT_DIM)
    edges = []

    for card in small_card_pool.values():
        if card.idx >= 0:
            x[card.idx] = torch.tensor(card.feature_vector)
        # Connect every card to commander
        if card.idx > 0:
            w = abs(card.synergy_score) + card.inclusion_rate / 100.0
            edges.append((0, card.idx, w))
            edges.append((card.idx, 0, w))

    if edges:
        ei = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
        ew = torch.tensor([e[2] for e in edges], dtype=torch.float)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
        ew = torch.zeros(0)

    return Data(x=x, edge_index=ei), ew


@pytest.fixture
def trained_model(small_graph_data):
    """A model trained for 20 epochs on the small graph."""
    data, ew = small_graph_data
    gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
    model = SynergyPredictor(gnn)
    train_gnn(model, data, ew, epochs=20, lr=1e-3)
    return model
