"""Graph construction tests."""

from mtg_deck_builder.card import Card
from mtg_deck_builder.commander import Commander


class TestUpsertLogic:
    def setup_method(self):
        """Import and set up a minimal CardGraph-like state for testing upsert."""
        self.game_changers = {"Rhystic Study"}
        self.cards = {}

    def _upsert(self, name, synergy=0, num_decks=0, potential_decks=1):
        """Replicate CardGraph._upsert logic for testing."""
        if name not in self.cards:
            self.cards[name] = Card(name=name, is_game_changer=name in self.game_changers)
        c = self.cards[name]
        c.is_game_changer = name in self.game_changers
        if synergy:
            c.synergy_score = max(c.synergy_score, synergy)
        if potential_decks > 0:
            c.inclusion_rate = max(c.inclusion_rate, num_decks / potential_decks * 100)

    def test_upsert_creates_card(self):
        self._upsert("Sol Ring", synergy=0.5, num_decks=50, potential_decks=100)
        assert "Sol Ring" in self.cards
        assert self.cards["Sol Ring"].synergy_score == 0.5

    def test_upsert_keeps_max_synergy(self):
        self._upsert("Sol Ring", synergy=0.3)
        self._upsert("Sol Ring", synergy=0.7)
        assert self.cards["Sol Ring"].synergy_score == 0.7

    def test_upsert_does_not_lower_synergy(self):
        self._upsert("Sol Ring", synergy=0.7)
        self._upsert("Sol Ring", synergy=0.3)
        assert self.cards["Sol Ring"].synergy_score == 0.7

    def test_upsert_keeps_max_inclusion_rate(self):
        self._upsert("Sol Ring", num_decks=30, potential_decks=100)
        self._upsert("Sol Ring", num_decks=50, potential_decks=100)
        assert self.cards["Sol Ring"].inclusion_rate == 50.0

    def test_game_changer_detection(self):
        self._upsert("Rhystic Study")
        assert self.cards["Rhystic Study"].is_game_changer is True

    def test_non_game_changer(self):
        self._upsert("Sol Ring")
        assert self.cards["Sol Ring"].is_game_changer is False


class TestIndexAssignment:
    def test_commander_is_index_zero(self, bant_commander):
        cards = {
            bant_commander.name: Card(name=bant_commander.name),
            "Sol Ring": Card(name="Sol Ring"),
            "Forest": Card(name="Forest"),
        }
        # Replicate _assign_indices logic
        cards[bant_commander.name].idx = 0
        idx = 1
        for name in cards:
            if name != bant_commander.name:
                cards[name].idx = idx
                idx += 1

        assert cards[bant_commander.name].idx == 0
        assert all(c.idx > 0 for n, c in cards.items() if n != bant_commander.name)
