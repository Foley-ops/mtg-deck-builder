"""Deck selector."""

from collections import defaultdict

import torch

from mtg_deck_builder.brackets import BRACKETS, EARLY_COMBO_CMC_THRESHOLD
from mtg_deck_builder.card import Card
from mtg_deck_builder.commander import Commander
from mtg_deck_builder.config import (
    COLOR_TO_BASIC, DECK_SIZE, DEVICE, GNN_WEIGHT, INCLUSION_RATE_FACTOR,
    MAX_NONLAND, OWNED_BASE_BOOST, OWNED_BRACKET_SCALE, OWNED_COVERAGE_DAMPEN,
    PRECON_BOOST, SYNERGY_WEIGHT,
)


class DeckSelector:
    def __init__(self, commander: Commander, graph, model, data):
        self.commander = commander
        self.graph = graph
        self.model = model
        self.data = data

    @torch.no_grad()
    def select(self, bracket, prefer_owned=False):
        """Select a deck. Returns (deck, scores_array)."""
        rules = BRACKETS[bracket]

        self.model.eval()
        self.model.to(DEVICE)
        d = self.data.to(DEVICE)
        emb = self.model.get_embeddings(d.x, d.edge_index)
        gnn_scores = (
            self.model.score_deck_candidates(
                emb, self.graph.name_to_idx[self.commander.name]
            )
            .cpu()
            .numpy()
        )

        # normalize GNN scores to [0, 1]
        gnn_min, gnn_max = gnn_scores.min(), gnn_scores.max()
        if gnn_max > gnn_min:
            gnn_norm = (gnn_scores - gnn_min) / (gnn_max - gnn_min)
        else:
            gnn_norm = gnn_scores * 0.0

        # blend GNN with EDHREC synergy + inclusion rate
        scores = gnn_norm.copy()
        for c in self.graph.cards.values():
            if 0 <= c.idx < len(scores):
                synergy_signal = (c.synergy_score
                                  + (c.inclusion_rate / 100.0) * INCLUSION_RATE_FACTOR)
                scores[c.idx] = (GNN_WEIGHT * gnn_norm[c.idx]
                                 + SYNERGY_WEIGHT * synergy_signal)

        # B1 favors precon cards
        if rules.prefer_precon_cards and self.graph.precon_cards:
            for c in self.graph.cards.values():
                if c.name in self.graph.precon_cards and 0 <= c.idx < len(scores):
                    scores[c.idx] += PRECON_BOOST

        if prefer_owned and self.graph.collection:
            owned_in_pool = sum(1 for c in self.graph.cards.values() if c.owned_qty > 0)
            total = len(self.graph.cards)
            coverage = owned_in_pool / max(total, 1)
            base_boost = OWNED_BASE_BOOST * (1.0 - coverage * OWNED_COVERAGE_DAMPEN)
            bracket_scale = OWNED_BRACKET_SCALE.get(bracket, 0.5)
            boost = base_boost * bracket_scale
            for c in self.graph.cards.values():
                if c.owned_qty > 0 and 0 <= c.idx < len(scores):
                    scores[c.idx] += boost

        scored = [
            (c, scores[c.idx])
            for c in self.graph.cards.values()
            if c.name != self.commander.name and 0 <= c.idx < len(scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        deck, gc_count, et_count, deck_names = [], 0, 0, set()
        for card, _ in scored:
            if len(deck) >= MAX_NONLAND:
                break
            if card.color_identity and not set(card.color_identity).issubset(
                self.commander.color_identity
            ):
                continue
            if card.is_game_changer:
                if gc_count >= rules.max_game_changers:
                    continue
                gc_count += 1
            if card.is_extra_turn:
                if not rules.allow_extra_turn_chains and et_count >= 1:
                    continue
                et_count += 1
            if card.is_mld and not rules.allow_mass_land_denial:
                continue
            if not rules.allow_2card_combos:
                bad = False
                for cid in card.combo_ids:
                    info = self.graph.combo_info.get(cid, {})
                    if info.get("size") == 2:
                        if (set(info["cards"]) - {card.name}) & (
                            deck_names | {self.commander.name}
                        ):
                            bad = True
                            break
                if bad:
                    continue
            elif rules.allow_2card_combos and not rules.allow_early_2card_combos:
                # B3: allow 2-card combos but block early ones (all pieces CMC <= 3)
                bad = False
                for cid in card.combo_ids:
                    info = self.graph.combo_info.get(cid, {})
                    if info.get("size") == 2:
                        partner_names = set(info["cards"]) - {card.name}
                        in_deck = partner_names & (deck_names | {self.commander.name})
                        if in_deck:
                            combo_cards_data = [
                                self.graph.cards.get(n) for n in info["cards"]
                            ]
                            if all(c and c.cmc <= EARLY_COMBO_CMC_THRESHOLD
                                   for c in combo_cards_data):
                                bad = True
                                break
                if bad:
                    continue
            deck.append(card)
            deck_names.add(card.name)

        # fill with basics
        need = DECK_SIZE - len(deck)
        basics = self._distribute_basics(need)
        for bname, count in basics:
            for _ in range(count):
                if len(deck) < DECK_SIZE:
                    deck.append(
                        Card(
                            name=bname,
                            type_line="Basic Land",
                            is_land=True,
                            source="basic",
                        )
                    )
        deck = deck[:DECK_SIZE]

        # compact terminal summary
        nl = [c for c in deck if not c.is_land]
        gc = [c for c in deck if c.is_game_changer]
        price = sum(c.price_usd for c in deck if c.price_usd)
        cmc_avg = sum(c.cmc for c in nl) / max(len(nl), 1)
        buy_count = 0
        if self.graph.collection:
            buy_count = sum(1 for c in deck if c.owned_qty == 0 and c.type_line != "Basic Land")
        print(f"  B{bracket}: {len(deck)} cards | GC: {len(gc)}/{rules.max_game_changers} | "
              f"CMC: {cmc_avg:.2f} | ${price:.0f} | Buy: {buy_count}")

        return deck, scores

    def _distribute_basics(self, need):
        """Split basics across commander colors."""
        colors = sorted(self.commander.color_identity, key="WUBRG".index)
        if not colors:
            return [("Wastes", need)]

        n_colors = len(colors)
        per_color = need // n_colors
        remainder = need % n_colors

        basics = []
        for i, color in enumerate(colors):
            count = per_color + (1 if i < remainder else 0)
            basics.append((COLOR_TO_BASIC[color], count))
        return basics
