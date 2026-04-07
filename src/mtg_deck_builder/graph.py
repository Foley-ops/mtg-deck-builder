"""Card graph construction."""

import math
from collections import defaultdict

import torch
from torch_geometric.data import Data

from mtg_deck_builder.card import Card
from mtg_deck_builder.db import Database
from mtg_deck_builder.commander import Commander
from mtg_deck_builder.api.scryfall import scryfall_search, scryfall_batch, fetch_game_changers
from mtg_deck_builder.api.edhrec import edhrec_json, edhrec_extract_cards
from mtg_deck_builder.api.spellbook import spellbook_search


BRACKET_SLUGS = ["exhibition", "core", "upgraded", "optimized"]


class CardGraph:
    def __init__(self, commander: Commander, db: Database, collection=None):
        self.commander = commander
        self.db = db
        self.cards: dict[str, Card] = {}
        self.name_to_idx: dict[str, int] = {}
        self.edges: list[tuple[int, int, float]] = []
        self.precon_cards: set[str] = set()
        self.combo_info: dict[str, dict] = {}
        self.card_combos: dict[str, list] = defaultdict(list)
        self.game_changers: set[str] = set()
        self.collection = collection or {}

    def load_all_data(self):
        cmd = self.commander
        print("=" * 60)
        print("MTG DECK OPTIMIZER v3 -- GNN EDITION")
        print(f"Commander: {cmd.name} ({cmd.color_filter.upper()}) | Slug: {cmd.edhrec_slug}")
        if self.collection:
            print(f"Collection: {len(self.collection)} unique cards")
        cs = self.db.stats()
        print(f"Cache: {cs['entries']} entries ({cs['size_mb']:.1f} MB)")
        print("=" * 60)

        print("\n[0/5] Game Changers...")
        self.game_changers = fetch_game_changers(self.db)

        print("\n[1/5] EDHREC...")
        self._load_edhrec()

        print("\n[2/5] Scryfall...")
        self._load_scryfall()

        print("\n[3/5] Spellbook...")
        self._load_spellbook()

        print("\n[4/5] Batch enrich...")
        self._batch_enrich()

        self._apply_collection()

        print("\n[5/5] Building graph...")
        self._assign_indices()
        self._build_edges()

        print(f"\n  Graph: {len(self.cards)} nodes, {len(self.edges)} edges")
        print(f"  Game Changers: {len(self.game_changers)} | Precon: {len(self.precon_cards)}")
        print(f"  Combos: {len(self.combo_info)}")
        print(f"  Cache: {self.db.hits} hits / {self.db.misses} misses")
        if self.collection:
            owned = sum(1 for c in self.cards.values() if c.owned_qty > 0)
            print(f"  Owned in pool: {owned}")

    def _load_edhrec(self):
        slug = self.commander.edhrec_slug
        # Main page + bracket pages
        slugs = [f"commanders/{slug}"]
        for bracket_slug in BRACKET_SLUGS:
            slugs.append(f"commanders/{slug}/{bracket_slug}")

        for s in slugs:
            label = s.split("/")[-1]
            print(f"  {label}...", end=" ")
            recs = edhrec_extract_cards(edhrec_json(s, self.db))
            print(f"{len(recs)} cards")
            for c in recs:
                if c["name"] and c["name"] != self.commander.name:
                    self._upsert(c["name"], c["synergy"], c["num_decks"], c["potential_decks"])

        # try to find precon data if there is one
        self._try_load_precon()

    def _try_load_precon(self):
        """Best-effort precon fetch from EDHREC."""
        slug = self.commander.edhrec_slug

        # Try the commander's EDHREC page to discover precon links
        main_data = edhrec_json(f"commanders/{slug}", self.db)
        precon_slugs = self._discover_precon_slugs(main_data, slug)

        if not precon_slugs:
            print("  precon... none found (OK)")
            return

        for precon_slug in precon_slugs:
            print(f"  precon ({precon_slug})...", end=" ")
            pcards = edhrec_extract_cards(edhrec_json(precon_slug, self.db))
            if pcards:
                print(f"{len(pcards)} cards")
                for c in pcards:
                    if c["name"]:
                        self.precon_cards.add(c["name"])
                        if c["name"] != self.commander.name:
                            self._upsert(c["name"])
                return
        print("  precon... 0")

    def _discover_precon_slugs(self, main_data, slug):
        precon_slugs = []
        raw = str(main_data)

        # Search the entire data structure for precon hrefs
        # EDHREC nests these in navigation groups -> items arrays
        self._find_precon_hrefs(main_data, precon_slugs)

        return precon_slugs

    def _find_precon_hrefs(self, obj, results):
        if isinstance(obj, dict):
            href = obj.get("href", "")
            if isinstance(href, str) and "/precon/" in href:
                clean = href.strip("/").replace(".json", "")
                if clean not in results:
                    results.append(clean)
            for v in obj.values():
                self._find_precon_hrefs(v, results)
        elif isinstance(obj, list):
            for item in obj:
                self._find_precon_hrefs(item, results)

    def _load_scryfall(self):
        cf = self.commander.color_filter
        print(f"  Staples (id<={cf})...", end=" ")
        try:
            results = scryfall_search(
                f"format:commander legal:commander (id<={cf}) -t:basic game:paper",
                self.db,
                200,
            )
            print(f"{len(results)} cards")
            self.db.upsert_cards_batch(results)
            for sc in results:
                self._ingest_scryfall(sc)
        except Exception as e:
            print(f"FAILED: {e}")

        # Try to find the commander's set for precon cards
        set_code = self.commander.scryfall_data.get("set", "")
        if set_code:
            print(f"  Set {set_code.upper()}...", end=" ")
            try:
                set_cards = scryfall_search(f"set:{set_code}", self.db, 150)
                print(f"{len(set_cards)} cards")
                self.db.upsert_cards_batch(set_cards)
                for sc in set_cards:
                    name = sc.get("name", "")
                    if name:
                        self.precon_cards.add(name)
                    self._ingest_scryfall(sc)
            except Exception as e:
                print(f"FAILED: {e}")

    def _load_spellbook(self):
        cmd = self.commander
        combos = []
        for q in [f'card:"{cmd.name}" ci:{cmd.color_filter}', f"ci:{cmd.color_filter}"]:
            print(f"  {q[:50]}...", end=" ")
            r = spellbook_search(q, self.db, 100)
            print(f"{len(r)} combos")
            combos.extend(r)
        for combo in combos:
            cid = combo.get("id", "")
            names = [u.get("card", {}).get("name", "") for u in combo.get("uses", [])]
            names = [n for n in names if n]
            self.combo_info[cid] = {
                "cards": names,
                "size": len(names),
                "description": combo.get("description", ""),
            }
            for n in names:
                self.card_combos[n].append(cid)
                if n != cmd.name:
                    self._upsert(n)

    def _batch_enrich(self):
        need = [
            n for n, c in self.cards.items()
            if not c.type_line and n != self.commander.name
        ]
        if not need:
            print("  All enriched!")
            return

        # Check DB for cards we already have stored
        from json import loads as json_loads
        db_need = self.db.get_cards_needing_refresh(need)
        db_have = set(need) - set(db_need)
        if db_have:
            print(f"  {len(db_have)} cards from DB, ", end="")
            for name in db_have:
                row = self.db.get_card(name)
                if row:
                    sc = json_loads(row["scryfall_data"]) if row["scryfall_data"] else {}
                    if sc:
                        self._ingest_scryfall(sc)
            # Recalculate what still needs API fetch
            need = [
                n for n, c in self.cards.items()
                if not c.type_line and n != self.commander.name
            ]

        if not need:
            print("all enriched!")
            return
        print(f"  {len(need)} cards -> ~{math.ceil(len(need)/75)} batch calls")
        results = scryfall_batch(need, self.db)
        # Persist to DB
        self.db.upsert_cards_batch(results)
        for sc in results:
            self._ingest_scryfall(sc)
        still = sum(
            1 for n, c in self.cards.items()
            if not c.type_line and n != self.commander.name
        )
        print(f"  Done ({still} still missing)")

    def _apply_collection(self):
        if not self.collection:
            return
        print("\n  Applying collection...")
        for name, qty in self.collection.items():
            if name in self.cards:
                self.cards[name].owned_qty = qty
            elif name != self.commander.name:
                self._upsert(name)
                self.cards[name].owned_qty = qty

    def _assign_indices(self):
        cmd = self.commander
        if cmd.name not in self.cards:
            self.cards[cmd.name] = Card(
                name=cmd.name,
                cmc=cmd.cmc,
                type_line=cmd.type_line,
                color_identity=list(cmd.color_identity),
                source="commander",
            )
        self.cards[cmd.name].idx = 0
        self.name_to_idx[cmd.name] = 0
        for i, name in enumerate(n for n in self.cards if n != cmd.name):
            self.cards[name].idx = i + 1
            self.name_to_idx[name] = i + 1

    def _build_edges(self):
        edge_set = set()
        syn_cards = [c for c in self.cards.values() if c.synergy_score > 0]

        for i, c1 in enumerate(syn_cards):
            for j, c2 in enumerate(syn_cards):
                if i >= j or c1.idx == c2.idx:
                    continue
                w = (
                    math.sqrt(abs(c1.synergy_score * c2.synergy_score))
                    * min(c1.inclusion_rate, c2.inclusion_rate)
                    / 100.0
                )
                if w > 0.001:
                    edge_set.add((c1.idx, c2.idx, w))
                    edge_set.add((c2.idx, c1.idx, w))

        for info in self.combo_info.values():
            ns = info["cards"]
            for i, n1 in enumerate(ns):
                for j, n2 in enumerate(ns):
                    if i >= j:
                        continue
                    if n1 in self.name_to_idx and n2 in self.name_to_idx:
                        edge_set.add((self.name_to_idx[n1], self.name_to_idx[n2], 5.0))
                        edge_set.add((self.name_to_idx[n2], self.name_to_idx[n1], 5.0))

        for card in self.cards.values():
            if card.idx != 0 and card.synergy_score != 0:
                w = abs(card.synergy_score) + card.inclusion_rate / 100.0
                edge_set.add((0, card.idx, w))
                edge_set.add((card.idx, 0, w))

        self.edges = list(edge_set)
        print(f"  {len(self.edges)} directed edges")

    def _upsert(self, name, synergy=0, num_decks=0, potential_decks=1):
        if name not in self.cards:
            self.cards[name] = Card(name=name, is_game_changer=name in self.game_changers)
        c = self.cards[name]
        c.is_game_changer = name in self.game_changers
        if synergy:
            c.synergy_score = max(c.synergy_score, synergy)
        if potential_decks > 0:
            c.inclusion_rate = max(c.inclusion_rate, num_decks / potential_decks * 100)

    def _ingest_scryfall(self, sc):
        name = sc.get("name", "")
        if not name or "//" in name:
            return
        if name not in self.cards:
            self.cards[name] = Card(name=name, is_game_changer=name in self.game_changers)
        c = self.cards[name]
        c.is_game_changer = name in self.game_changers
        c.cmc = sc.get("cmc", 0)
        c.type_line = sc.get("type_line", "")
        c.oracle_text = sc.get("oracle_text", "")
        c.colors = sc.get("colors", [])
        c.color_identity = sc.get("color_identity", [])
        c.edhrec_rank = sc.get("edhrec_rank")
        c.is_land = "Land" in c.type_line
        ot = (c.oracle_text or "").lower()
        c.is_extra_turn = "extra turn" in ot or "additional turn" in ot
        c.is_mld = any(
            p in ot
            for p in [
                "destroy all lands",
                "destroy all nonbasic",
                "each player sacrifices all",
            ]
        )
        p = sc.get("prices", {})
        if p.get("usd"):
            try:
                c.price_usd = float(p["usd"])
            except (ValueError, TypeError):
                pass

    def to_pyg(self):
        n = len(self.cards)
        x = torch.zeros(n, Card.NODE_FEAT_DIM)
        for c in self.cards.values():
            if c.idx >= 0:
                x[c.idx] = torch.tensor(c.feature_vector)
        if self.edges:
            ei = torch.tensor(
                [[e[0] for e in self.edges], [e[1] for e in self.edges]],
                dtype=torch.long,
            )
            ew = torch.tensor([e[2] for e in self.edges], dtype=torch.float)
        else:
            ei = (
                torch.tensor([[i, i] for i in range(n)], dtype=torch.long)
                .t()
                .contiguous()
            )
            ew = torch.ones(n)
        return Data(x=x, edge_index=ei), ew
