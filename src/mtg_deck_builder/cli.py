"""CLI."""

import argparse
from collections import defaultdict

import torch

from mtg_deck_builder.brackets import BRACKETS
from mtg_deck_builder.card import Card
from mtg_deck_builder.collection import generate_template_csv, load_collection_csv
from mtg_deck_builder.commander import name_to_edhrec_slug, resolve_commander
from mtg_deck_builder.config import (
    DEVICE,
    EMBED_DIM,
    GNN_LAYERS,
    HIDDEN_DIM,
    NUM_HEADS,
)
from mtg_deck_builder.db import Database, compute_model_hash
from mtg_deck_builder.graph import CardGraph
from mtg_deck_builder.model import CardGNN, SynergyPredictor, train_gnn
from mtg_deck_builder.selector import DeckSelector


def main():
    p = argparse.ArgumentParser(description="MTG GNN Deck Optimizer")
    p.add_argument(
        "--commander",
        type=str,
        default="Ms. Bumbleflower",
        help="Commander name (looked up on Scryfall with fuzzy matching)",
    )
    p.add_argument("--train-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bracket", type=int, default=0, help="0=all")
    p.add_argument(
        "--archetype",
        type=int,
        default=0,
        help="0=group_hug 1=counters 2=wheels 3=cantrips",
    )
    p.add_argument("--collection", type=str, help="Path to collection CSV (imports and uses)")
    p.add_argument(
        "--prefer-owned", action="store_true", help="Boost owned cards in scoring"
    )
    p.add_argument(
        "--generate-template",
        action="store_true",
        help="Write collection_template.csv and exit",
    )
    p.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    p.add_argument("--refresh", action="store_true", help="Force re-fetch all API data")
    p.add_argument("--clear-collection", action="store_true", help="Remove stored collection")
    p.add_argument("--force-train", action="store_true", help="Force re-training even if embeddings are cached")
    args = p.parse_args()

    if args.generate_template:
        generate_template_csv()
        return

    db = Database(db_path=args.db, force_refresh=args.refresh)

    if args.clear_collection:
        db.clear_collection()
        print("Collection cleared.")
        db.close()
        return

    # Resolve commander
    print(f"\nResolving commander: {args.commander}")
    commander = resolve_commander(args.commander, db)
    print(f"  -> {commander.name} ({', '.join(sorted(commander.color_identity))})")
    print(f"  -> EDHREC slug: {commander.edhrec_slug}")

    # Collection: import from CSV if provided, otherwise use stored
    collection = None
    if args.collection:
        collection = load_collection_csv(args.collection)
        if collection:
            db.import_collection(collection)
            print(f"  Collection saved to DB ({len(collection)} cards)")
    else:
        stored = db.get_collection()
        if stored:
            collection = stored
            print(f"  Using stored collection ({len(collection)} cards)")

    # Build card graph
    graph = CardGraph(commander, db, collection)
    graph.load_all_data()
    if len(graph.cards) < 20:
        print("\nERROR: too few cards. Check network.")
        db.close()
        return

    data, ew = graph.to_pyg()
    print(
        f"\nGraph: {data.x.size(0)} nodes | {data.edge_index.size(1)} edges | {data.x.size(1)} features"
    )

    # Check for cached embeddings
    model_hash = compute_model_hash(
        EMBED_DIM, HIDDEN_DIM, NUM_HEADS, GNN_LAYERS,
        args.train_epochs, len(graph.cards),
    )

    gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
    model = SynergyPredictor(gnn)

    cached_emb = None if args.force_train else db.get_embeddings(commander.name, model_hash)

    if cached_emb is not None:
        print(f"\n  Using cached embeddings ({len(cached_emb)} cards, hash={model_hash})")
    else:
        print(f"\n{'='*60}\nTRAINING ({args.train_epochs} epochs, {DEVICE})\n{'='*60}")
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        losses = train_gnn(model, data, ew, args.train_epochs, args.lr)
        print(f"  Final loss: {losses[-1]:.4f}")

        # Cache embeddings
        model.eval()
        with torch.no_grad():
            emb = model.get_embeddings(
                data.to(DEVICE).x, data.to(DEVICE).edge_index
            ).cpu()
        card_embeddings = {}
        for card_name, card in graph.cards.items():
            if 0 <= card.idx < emb.size(0):
                card_embeddings[card_name] = emb[card.idx]
        db.save_embeddings(commander.name, card_embeddings, args.train_epochs, model_hash)
        print(f"  Embeddings cached ({len(card_embeddings)} cards, hash={model_hash})")

    # Select decks
    selector = DeckSelector(commander, graph, model, data)
    brackets = [args.bracket] if args.bracket else [1, 2, 3, 4]
    decks = {}
    file_slug = name_to_edhrec_slug(commander.name)

    for b in brackets:
        deck = selector.select(b, args.archetype, args.prefer_owned)
        decks[b] = deck
        fname = f"{file_slug}_gnn_b{b}.txt"
        with open(fname, "w") as f:
            f.write(f"// {commander.name} - Bracket {b} (GNN)\n")
            f.write(f"// {BRACKETS[b].description}\n// Commander: {commander.name}\n\n")
            bc = defaultdict(int)
            for c in sorted(deck, key=lambda c: (c.is_land, c.cmc, c.name)):
                if c.type_line == "Basic Land":
                    bc[c.name] += 1
                else:
                    f.write(f"1 {c.name}\n")
            for n, ct in sorted(bc.items()):
                f.write(f"{ct} {n}\n")
        print(f"  Saved: {fname}")

    if len(decks) > 1:
        print(f"\n{'='*60}\nBRACKET DIFF\n{'='*60}")
        sb = sorted(decks.keys())
        for i in range(1, len(sb)):
            pn = {c.name for c in decks[sb[i - 1]]}
            cn = {c.name for c in decks[sb[i]]}
            added, removed = cn - pn, pn - cn
            print(f"\n  B{sb[i-1]} -> B{sb[i]}: +{len(added)} / -{len(removed)}")
            for n in sorted(added)[:10]:
                gc = " GC" if n in graph.game_changers else ""
                print(f"    + {n}{gc}")
            for n in sorted(removed)[:10]:
                print(f"    - {n}")

    print(f"\n{'='*60}\nEMBEDDING ANALYSIS\n{'='*60}")
    model.eval()
    with torch.no_grad():
        emb = model.get_embeddings(data.to(DEVICE).x, data.to(DEVICE).edge_index).cpu()
    cmd_e = emb[graph.name_to_idx[commander.name]]
    sims = torch.mv(emb, cmd_e)
    for s, idx in zip(*torch.topk(sims, min(15, len(sims)))):
        name = [n for n, c in graph.cards.items() if c.idx == idx.item()]
        name = name[0] if name else "?"
        gc = " GC" if name in graph.game_changers else ""
        print(f"  {s:.4f}  {name}{gc}")

    db_stats = db.stats()
    print(f"\nDONE | DB: {db.db_path} ({db_stats['size_mb']:.1f} MB)")
    print(f"     | Cache: {db.hits} hits / {db.misses} misses")
    db.close()


if __name__ == "__main__":
    main()
