"""Card feature vector tests."""

from mtg_deck_builder.card import Card


class TestFeatureVector:
    def test_dimension_is_18(self):
        c = Card(name="Test Card")
        assert len(c.feature_vector) == 18
        assert Card.NODE_FEAT_DIM == 18

    def test_color_encoding_bant(self):
        c = Card(name="Bant Card", color_identity=["W", "U", "G"])
        fv = c.feature_vector
        assert fv[0] == 1.0  # W
        assert fv[1] == 1.0  # U
        assert fv[2] == 0.0  # B
        assert fv[3] == 0.0  # R
        assert fv[4] == 1.0  # G

    def test_color_encoding_colorless(self):
        c = Card(name="Sol Ring", color_identity=[])
        fv = c.feature_vector
        assert fv[0:5] == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_type_encoding_creature(self):
        c = Card(name="Test", type_line="Creature — Human Wizard")
        fv = c.feature_vector
        assert fv[5] == 1.0   # creature
        assert fv[6] == 0.0   # instant
        assert fv[7] == 0.0   # sorcery

    def test_type_encoding_artifact_creature(self):
        c = Card(name="Test", type_line="Artifact Creature — Golem")
        fv = c.feature_vector
        assert fv[5] == 1.0   # creature
        assert fv[8] == 1.0   # artifact

    def test_land_flag(self):
        c = Card(name="Test", type_line="Land")
        fv = c.feature_vector
        assert fv[10] == 1.0  # land

    def test_cmc_normalized(self):
        c = Card(name="Test", cmc=5.0)
        fv = c.feature_vector
        assert fv[11] == 0.5  # 5.0 / 10.0

    def test_synergy_score_direct(self):
        c = Card(name="Test", synergy_score=0.75)
        fv = c.feature_vector
        assert fv[12] == 0.75

    def test_inclusion_rate_capped(self):
        c = Card(name="Test", inclusion_rate=150.0)
        fv = c.feature_vector
        assert fv[13] == 1.0  # capped at 1.0

    def test_game_changer_flag(self):
        c = Card(name="Test", is_game_changer=True)
        assert c.feature_vector[14] == 1.0

    def test_owned_flag(self):
        c = Card(name="Test", owned_qty=3)
        assert c.feature_vector[17] == 1.0

    def test_owned_flag_zero(self):
        c = Card(name="Test", owned_qty=0)
        assert c.feature_vector[17] == 0.0
