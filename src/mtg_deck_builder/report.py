"""Output report generation -- markdown, csv, and txt decklists."""

import csv
from collections import defaultdict
from pathlib import Path


def generate_reports(commander, graph, decks, scores_by_bracket, embeddings,
                     output_dir=".", file_slug=None):
    """Write all output files for a run. Returns list of paths written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = file_slug or commander.edhrec_slug
    written = []

    # individual deck files (txt + csv per bracket)
    for b, deck in decks.items():
        txt_path = output_dir / f"{slug}_b{b}.txt"
        csv_path = output_dir / f"{slug}_b{b}.csv"
        _write_decklist_txt(txt_path, commander, deck, b)
        _write_decklist_csv(csv_path, commander, deck, b)
        written.extend([txt_path, csv_path])

    # combined markdown report
    md_path = output_dir / f"{slug}_report.md"
    _write_markdown_report(md_path, commander, graph, decks, scores_by_bracket, embeddings)
    written.append(md_path)

    return written


def _write_decklist_txt(path, commander, deck, bracket):
    from mtg_deck_builder.brackets import BRACKETS
    rules = BRACKETS[bracket]
    bc = defaultdict(int)
    with open(path, "w") as f:
        f.write(f"// {commander.name} - Bracket {bracket}\n")
        f.write(f"// {rules.description}\n")
        f.write(f"// Commander: {commander.name}\n\n")
        for c in sorted(deck, key=lambda c: (c.is_land, c.cmc, c.name)):
            if c.type_line == "Basic Land":
                bc[c.name] += 1
            else:
                f.write(f"1 {c.name}\n")
        for n, ct in sorted(bc.items()):
            f.write(f"{ct} {n}\n")


def _write_decklist_csv(path, commander, deck, bracket):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "name", "type", "cmc", "owned", "price_usd"])
        bc = defaultdict(int)
        for c in sorted(deck, key=lambda c: (c.is_land, c.cmc, c.name)):
            if c.type_line == "Basic Land":
                bc[c.name] += 1
            else:
                writer.writerow([
                    1, c.name, c.type_line, c.cmc,
                    "yes" if c.owned_qty > 0 else "no",
                    f"{c.price_usd:.2f}" if c.price_usd else "",
                ])
        for n, ct in sorted(bc.items()):
            writer.writerow([ct, n, "Basic Land", 0, "", ""])


def _write_markdown_report(path, commander, graph, decks, scores_by_bracket, embeddings):
    from mtg_deck_builder.brackets import BRACKETS

    lines = []
    w = lines.append

    # title
    w(f"# {commander.name} - Deck Report\n")

    # table of contents
    w("## Contents\n")
    w("- [Overview](#overview)")
    brackets = sorted(decks.keys())
    for b in brackets:
        w(f"- [Bracket {b} - {BRACKETS[b].description.split(' - ')[0]}](#bracket-{b})")
    if len(decks) > 1:
        w("- [Bracket Comparison](#bracket-comparison)")
    w("- [Embedding Analysis](#embedding-analysis)")
    w("")

    # overview
    w("## Overview\n")
    w(f"| | |")
    w(f"|---|---|")
    w(f"| Commander | {commander.name} |")
    w(f"| Colors | {', '.join(sorted(commander.color_identity))} |")
    w(f"| Card Pool | {len(graph.cards)} cards |")
    w(f"| Game Changers | {len(graph.game_changers)} in format |")
    w(f"| Combos Tracked | {len(graph.combo_info)} |")
    if graph.collection:
        owned = sum(1 for c in graph.cards.values() if c.owned_qty > 0)
        w(f"| Cards Owned | {owned} / {len(graph.cards)} |")
    w("")

    # each bracket
    for b in brackets:
        deck = decks[b]
        scores = scores_by_bracket[b]
        rules = BRACKETS[b]
        nl = [c for c in deck if not c.is_land]
        gc = [c for c in deck if c.is_game_changer]
        basics = [c for c in deck if c.type_line == "Basic Land"]
        price = sum(c.price_usd for c in deck if c.price_usd)
        cmc = sum(c.cmc for c in nl) / max(len(nl), 1)

        w(f"## Bracket {b}\n")
        w(f"**{rules.description}**\n")

        # stats table
        w("| Stat | Value |")
        w("|------|-------|")
        w(f"| Cards | {len(deck)} + commander |")
        w(f"| Game Changers | {len(gc)} / {rules.max_game_changers} |")
        w(f"| Avg CMC | {cmc:.2f} |")
        w(f"| Est. Price | ${price:.2f} |")
        w(f"| Basic Lands | {len(basics)} |")
        if graph.collection:
            buy = [c for c in deck if c.owned_qty == 0 and c.type_line != "Basic Land"]
            buy_cost = sum(c.price_usd for c in buy if c.price_usd)
            w(f"| Need to Buy | {len(buy)} cards (${buy_cost:.2f}) |")
        w("")

        # game changers
        if gc:
            w("**Game Changers:**\n")
            for c in gc:
                p = f" (${c.price_usd:.2f})" if c.price_usd else ""
                w(f"- {c.name}{p}")
            w("")

        # top 25
        w("### Top 25 Cards\n")
        w("| # | Card | Score | Synergy | Owned | Price |")
        w("|---|------|-------|---------|-------|-------|")
        nl.sort(key=lambda c: scores[c.idx] if 0 <= c.idx < len(scores) else 0, reverse=True)
        for i, c in enumerate(nl[:25], 1):
            s = scores[c.idx] if 0 <= c.idx < len(scores) else 0
            own = "yes" if c.owned_qty > 0 else ""
            gc_tag = " (GC)" if c.is_game_changer else ""
            p = f"${c.price_usd:.2f}" if c.price_usd else ""
            w(f"| {i} | {c.name}{gc_tag} | {s:.4f} | {c.synergy_score:.2f} | {own} | {p} |")
        w("")

        # full decklist
        w("### Decklist\n")
        w("```")
        w(f"Commander: {commander.name}")
        w("")
        bc = defaultdict(int)
        for c in sorted(deck, key=lambda c: (c.is_land, c.cmc, c.name)):
            if c.type_line == "Basic Land":
                bc[c.name] += 1
            else:
                own = " [OWN]" if c.owned_qty > 0 else ""
                w(f"1 {c.name}{own}")
        for n, ct in sorted(bc.items()):
            w(f"{ct} {n}")
        w("```\n")

    # bracket comparison
    if len(decks) > 1:
        w("## Bracket Comparison\n")
        sb = sorted(decks.keys())
        for i in range(1, len(sb)):
            prev_names = {c.name for c in decks[sb[i - 1]]}
            curr_names = {c.name for c in decks[sb[i]]}
            added = curr_names - prev_names
            removed = prev_names - curr_names

            w(f"### B{sb[i-1]} -> B{sb[i]}: +{len(added)} / -{len(removed)}\n")
            if added:
                w("**Added:**\n")
                for n in sorted(added):
                    gc_tag = " (GC)" if n in graph.game_changers else ""
                    w(f"- {n}{gc_tag}")
                w("")
            if removed:
                w("**Removed:**\n")
                for n in sorted(removed):
                    w(f"- {n}")
                w("")

    # embedding analysis
    w("## Embedding Analysis\n")
    w("Cards most similar to commander in GNN embedding space:\n")
    w("| Rank | Card | Similarity |")
    w("|------|------|------------|")
    if embeddings is not None:
        import torch
        cmd_name = commander.name
        cmd_idx = graph.name_to_idx.get(cmd_name, 0)
        cmd_e = embeddings[cmd_idx]
        sims = torch.mv(embeddings, cmd_e)
        topk_vals, topk_idx = torch.topk(sims, min(15, len(sims)))
        for rank, (s, idx) in enumerate(zip(topk_vals, topk_idx), 1):
            name = [n for n, c in graph.cards.items() if c.idx == idx.item()]
            name = name[0] if name else "?"
            gc_tag = " (GC)" if name in graph.game_changers else ""
            w(f"| {rank} | {name}{gc_tag} | {s:.4f} |")
    w("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
