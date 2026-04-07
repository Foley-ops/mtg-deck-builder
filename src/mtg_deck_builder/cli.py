"""CLI."""

import argparse

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
from mtg_deck_builder.report import generate_reports
from mtg_deck_builder.selector import DeckSelector


def main():
    p = argparse.ArgumentParser(description="MTG GNN Deck Optimizer")
    p.add_argument(
        "--commander", type=str, default="Ms. Bumbleflower",
        help="Commander name (Scryfall fuzzy matching)",
    )
    p.add_argument("--train-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bracket", type=int, default=0, help="0=all")
    p.add_argument("--archetype", type=int, default=0,
                    help="0=group_hug 1=counters 2=wheels 3=cantrips")
    p.add_argument("--collection", type=str, help="Path to collection CSV")
    p.add_argument("--prefer-owned", action="store_true", help="Boost owned cards")
    p.add_argument("--generate-template", action="store_true",
                    help="Write collection_template.csv and exit")
    p.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    p.add_argument("--output-dir", type=str, default="./output", help="Output directory for reports")
    p.add_argument("--refresh", action="store_true", help="Force re-fetch all API data")
    p.add_argument("--clear-collection", action="store_true", help="Remove stored collection")
    p.add_argument("--force-train", action="store_true", help="Force re-training")
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

    # resolve commander
    print(f"\nResolving commander: {args.commander}")
    commander = resolve_commander(args.commander, db)
    print(f"  {commander.name} ({', '.join(sorted(commander.color_identity))})")

    # collection
    collection = None
    if args.collection:
        collection = load_collection_csv(args.collection)
        if collection:
            db.import_collection(collection)
    else:
        stored = db.get_collection()
        if stored:
            collection = stored
            print(f"  Collection: {len(collection)} cards (from db)")

    # build graph
    graph = CardGraph(commander, db, collection)
    graph.load_all_data()
    if len(graph.cards) < 20:
        print("\nERROR: too few cards. Check network.")
        db.close()
        return

    data, ew = graph.to_pyg()
    print(f"\n  Graph: {data.x.size(0)} nodes, {data.edge_index.size(1)} edges")

    # training / cached embeddings
    model_hash = compute_model_hash(
        EMBED_DIM, HIDDEN_DIM, NUM_HEADS, GNN_LAYERS,
        args.train_epochs, len(graph.cards),
    )
    gnn = CardGNN(Card.NODE_FEAT_DIM, HIDDEN_DIM, EMBED_DIM, NUM_HEADS, GNN_LAYERS)
    model = SynergyPredictor(gnn)

    cached_emb = None if args.force_train else db.get_embeddings(commander.name, model_hash)

    if cached_emb is not None:
        print(f"  Using cached embeddings ({len(cached_emb)} cards)")
    else:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n  Training GNN ({param_count:,} params, {args.train_epochs} epochs, {DEVICE})")
        losses = train_gnn(model, data, ew, args.train_epochs, args.lr)
        print(f"  Final loss: {losses[-1]:.4f}")

        # cache embeddings
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

    # select decks
    brackets = [args.bracket] if args.bracket else [1, 2, 3, 4]
    decks = {}
    scores_by_bracket = {}

    selector = DeckSelector(commander, graph, model, data)
    print(f"\n  Selecting decks:")
    for b in brackets:
        deck, scores = selector.select(b, args.archetype, args.prefer_owned)
        decks[b] = deck
        scores_by_bracket[b] = scores

    # get embeddings for report
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(
            data.to(DEVICE).x, data.to(DEVICE).edge_index
        ).cpu()

    # generate all output files
    file_slug = name_to_edhrec_slug(commander.name)
    written = generate_reports(
        commander, graph, decks, scores_by_bracket, embeddings,
        output_dir=args.output_dir, file_slug=file_slug,
    )

    print(f"\n  Output files:")
    for path in written:
        print(f"    {path}")

    db_stats = db.stats()
    print(f"\n  DB: {db.db_path} ({db_stats['size_mb']:.1f} MB)"
          f" | Cache: {db.hits} hits / {db.misses} misses\n")
    db.close()


if __name__ == "__main__":
    main()
