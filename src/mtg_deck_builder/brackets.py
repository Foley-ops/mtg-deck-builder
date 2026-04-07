"""Bracket rules."""

from dataclasses import dataclass


@dataclass
class BracketRules:
    bracket: int
    max_game_changers: int
    allow_2card_combos: bool
    allow_early_2card_combos: bool
    allow_extra_turn_chains: bool
    allow_mass_land_denial: bool
    prefer_precon_cards: bool
    description: str


BRACKETS = {
    1: BracketRules(
        1, 0, False, False, False, False, True,
        "Exhibition - themed/casual, no GC, no combos, precon-favored",
    ),
    2: BracketRules(
        2, 0, False, False, False, False, False,
        "Core - precon-level, no GC, no combos",
    ),
    3: BracketRules(
        3, 3, True, False, False, False, False,
        "Upgraded - <=3 GC, no early combos, no MLD",
    ),
    4: BracketRules(
        4, 999, True, True, True, True, False,
        "Optimized - unrestricted",
    ),
}

# combos where all pieces cost this or less are "early"
EARLY_COMBO_CMC_THRESHOLD = 3
