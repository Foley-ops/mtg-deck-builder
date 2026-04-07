"""GNN model tests."""

import torch

from mtg_deck_builder.card import Card
from mtg_deck_builder.config import EMBED_DIM, HIDDEN_DIM, NUM_HEADS, GNN_LAYERS
from mtg_deck_builder.model import CardGNN, SynergyPredictor, train_gnn


class TestCardGNN:
    def test_output_shape(self, small_graph_data):
        data, _ = small_graph_data
        gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
        emb = gnn(data.x, data.edge_index)
        assert emb.shape == (data.x.size(0), EMBED_DIM)

    def test_embeddings_are_l2_normalized(self, small_graph_data):
        data, _ = small_graph_data
        gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
        emb = gnn(data.x, data.edge_index)
        norms = torch.norm(emb, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestSynergyPredictor:
    def test_score_shape(self, small_graph_data):
        data, _ = small_graph_data
        gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
        model = SynergyPredictor(gnn)
        emb = model.get_embeddings(data.x, data.edge_index)
        scores = model.score_deck_candidates(emb, commander_idx=0)
        assert scores.shape == (data.x.size(0),)

    def test_edge_prediction_shape(self, small_graph_data):
        data, _ = small_graph_data
        gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
        model = SynergyPredictor(gnn)
        emb = model.get_embeddings(data.x, data.edge_index)
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 3])
        pred = model.predict_edge(emb, src, dst)
        assert pred.shape == (3,)


class TestTraining:
    def test_loss_decreases(self, small_graph_data):
        data, ew = small_graph_data
        gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
        model = SynergyPredictor(gnn)
        losses = train_gnn(model, data, ew, epochs=50, lr=1e-3)
        assert losses[-1] < losses[0]
