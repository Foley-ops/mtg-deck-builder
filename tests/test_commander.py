"""Slug generation and color filter tests."""

import pytest

from mtg_deck_builder.commander import name_to_edhrec_slug, colors_to_filter


class TestNameToEdhrecSlug:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Ms. Bumbleflower", "ms-bumbleflower"),
            ("Atraxa, Praetors' Voice", "atraxa-praetors-voice"),
            ("Kenrith, the Returned King", "kenrith-the-returned-king"),
            ("K'rrik, Son of Yawgmoth", "krrik-son-of-yawgmoth"),
            ("Korvold, Fae-Cursed King", "korvold-fae-cursed-king"),
            ("Urza, Lord High Artificer", "urza-lord-high-artificer"),
            # Smart quotes (sometimes appear in copy-paste)
            ("K\u2019rrik, Son of Yawgmoth", "krrik-son-of-yawgmoth"),
            # Extra whitespace
            ("  Ms.  Bumbleflower  ", "ms-bumbleflower"),
        ],
    )
    def test_slug_generation(self, name, expected):
        assert name_to_edhrec_slug(name) == expected

    def test_slug_is_lowercase(self):
        slug = name_to_edhrec_slug("SHOUTING COMMANDER")
        assert slug == slug.lower()

    def test_slug_has_no_spaces(self):
        slug = name_to_edhrec_slug("Some Commander Name")
        assert " " not in slug


class TestColorsToFilter:
    @pytest.mark.parametrize(
        "colors, expected",
        [
            ({"W", "U", "G"}, "wug"),
            ({"W", "U", "B", "R", "G"}, "wubrg"),
            ({"B", "R"}, "br"),
            ({"G"}, "g"),
            ({"W"}, "w"),
            (set(), ""),
        ],
    )
    def test_color_filter(self, colors, expected):
        assert colors_to_filter(colors) == expected

    def test_color_order_is_wubrg(self):
        result = colors_to_filter({"G", "R", "B", "U", "W"})
        assert result == "wubrg"
