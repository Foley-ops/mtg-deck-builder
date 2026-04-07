"""Card dataclass."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Card:
    name: str
    idx: int = -1
    cmc: float = 0.0
    type_line: str = ""
    oracle_text: str = ""
    colors: list = field(default_factory=list)
    color_identity: list = field(default_factory=list)
    edhrec_rank: Optional[int] = None
    synergy_score: float = 0.0
    inclusion_rate: float = 0.0
    is_game_changer: bool = False
    is_land: bool = False
    is_extra_turn: bool = False
    is_mld: bool = False
    price_usd: Optional[float] = None
    combo_ids: list = field(default_factory=list)
    owned_qty: int = 0
    source: str = ""

    @property
    def feature_vector(self):
        """Hand-crafted node features (18-dim)."""
        ci = set(self.color_identity)
        return [
            # color identity (5)
            1.0 if "W" in ci else 0.0,
            1.0 if "U" in ci else 0.0,
            1.0 if "B" in ci else 0.0,
            1.0 if "R" in ci else 0.0,
            1.0 if "G" in ci else 0.0,
            # type (6)
            1.0 if "creature" in self.type_line.lower() else 0.0,
            1.0 if "instant" in self.type_line.lower() else 0.0,
            1.0 if "sorcery" in self.type_line.lower() else 0.0,
            1.0 if "artifact" in self.type_line.lower() else 0.0,
            1.0 if "enchantment" in self.type_line.lower() else 0.0,
            1.0 if "land" in self.type_line.lower() else 0.0,
            # numeric (7)
            self.cmc / 10.0,
            self.synergy_score,
            min(self.inclusion_rate / 100.0, 1.0),
            1.0 if self.is_game_changer else 0.0,
            1.0 if self.is_extra_turn else 0.0,
            1.0 if self.is_mld else 0.0,
            1.0 if self.owned_qty > 0 else 0.0,
        ]

    NODE_FEAT_DIM = 18
