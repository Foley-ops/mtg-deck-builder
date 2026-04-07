"""Config."""

import random

import numpy as np
import torch


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()

# GNN architecture
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_HEADS = 4
GNN_LAYERS = 3

# deck construction
DECK_SIZE = 99
MAX_NONLAND = 74

# scoring weights
SYNERGY_WEIGHT = 0.6
GNN_WEIGHT = 0.4
INCLUSION_RATE_FACTOR = 0.5
PRECON_BOOST = 0.2
OWNED_BASE_BOOST = 0.3
OWNED_COVERAGE_DAMPEN = 0.7
OWNED_BRACKET_SCALE = {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.25}

# graph construction
COMBO_EDGE_WEIGHT = 5.0
MIN_SYNERGY_EDGE_WEIGHT = 0.001
SCRYFALL_STAPLE_LIMIT = 200
SCRYFALL_SET_LIMIT = 150

# reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# color -> basic land
COLOR_TO_BASIC = {
    "W": "Plains",
    "U": "Island",
    "B": "Swamp",
    "R": "Mountain",
    "G": "Forest",
}
