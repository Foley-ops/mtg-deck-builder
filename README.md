# MTG Commander Deck Optimizer (GNN)

Uses a GATv2 graph neural network to build optimized Commander decklists. You give it a commander name, it pulls card data from Scryfall/EDHREC/Commander Spellbook, trains a GNN on card co-occurrence, and spits out bracket-legal 99-card decklists.

Works with any commander, any color identity. Optionally takes your collection CSV to prioritize cards you already own.

## Setup

```bash
pip install torch torch-geometric requests numpy
pip install -e ".[dev]"
```

Automatically uses CUDA (NVIDIA), MPS (Apple Silicon), or CPU -- whatever's available.

For CPU-only installs:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric requests numpy
pip install -e ".[dev]"
```

Docker:
```bash
docker build -t mtg-deck-builder .
docker run --rm mtg-deck-builder python -m mtg_deck_builder --commander "Ms. Bumbleflower"
```

## Usage

```bash
# all 4 brackets
mtg-deck-builder --commander "Ms. Bumbleflower"

# specific bracket
mtg-deck-builder --commander "Atraxa, Praetors' Voice" --bracket 3

# with your collection (saved to local db, reused on future runs)
mtg-deck-builder --commander "Ms. Bumbleflower" --collection my_cards.csv --prefer-owned

# archetypes: 0=group_hug, 1=counters, 2=wheels, 3=cantrips
mtg-deck-builder --commander "Ms. Bumbleflower" --archetype 1

# force re-fetch from APIs
mtg-deck-builder --commander "Ms. Bumbleflower" --refresh

# force retrain (ignore cached embeddings)
mtg-deck-builder --commander "Ms. Bumbleflower" --force-train
```

Commander names use Scryfall fuzzy matching so you don't need exact spelling.

## How it works

**Data**: ~13 API calls on first run. Scryfall for card data + game changers, EDHREC for synergy/inclusion scores per commander, Commander Spellbook for combo detection. Everything cached in a local SQLite db after the first fetch.

**Graph**: Cards are nodes. Edges come from synergy co-occurrence (geometric mean of synergy scores * min inclusion rate), combo relationships (weight 5.0), and commander affinity. Each node has an 18-dim feature vector (color identity, card type, CMC, synergy, inclusion rate, various flags).

**Model**: GATv2 with 3 layers, 4 attention heads, residual connections, and layer norm. Trained with weighted link prediction (skip-gram style) - positive edges get high dot-product similarity, random negatives get pushed apart. Archetype conditioning via learned embeddings combined with the commander embedding through a gating mechanism.

**Scoring**: Final card scores blend GNN embedding similarity (40%) with raw EDHREC synergy + inclusion rate (60%). Pure GNN scores alone aren't reliable enough yet.

**Deck selection**: Greedy - take highest-scored cards that pass bracket constraints, fill remaining slots with basics split across commander colors.

## Bracket rules

These are hard filters, not learned:

| Bracket | Game Changers | 2-Card Combos | Extra Turns | MLD |
|---------|--------------|---------------|-------------|-----|
| 1 (Exhibition) | 0 | No | No | No |
| 2 (Core) | 0 | No | No | No |
| 3 (Upgraded) | <=3 | Yes (no early) | No | No |
| 4 (Optimized) | No limit | Yes | Yes | Yes |

Game changers list is pulled live from Scryfall (`is:gamechanger`) so it stays current.

B1 also biases toward precon cards for a more casual feel.

## Collection

Supports ManaBox CSV exports directly. Column names are auto-detected (handles `name`/`Card`/`card_name` and `quantity`/`Qty`/`count`/`amount`).

```bash
mtg-deck-builder --generate-template    # see the expected format
mtg-deck-builder --collection cards.csv  # imports + saves to db
mtg-deck-builder --clear-collection      # wipe stored collection
```

Once imported, your collection persists in the db. Future runs automatically use it - you don't need to pass `--collection` again.

## Local database

SQLite at `~/.mtg_deck_builder/deck_builder.db`. Stores card data (7d TTL), synergy scores (24h), combos (24h), commander lookups (30d), your collection (permanent), and trained embeddings (invalidated on hyperparameter changes). Second run for the same commander skips training and most API calls.

## Output

Each run generates three file types per bracket, plus a combined report:

- `<commander>_b1.txt` through `_b4.txt` -- plain decklists, importable into ManaBox/Archidekt/Moxfield
- `<commander>_b1.csv` through `_b4.csv` -- full card data with type, CMC, owned status, and price
- `<commander>_report.md` -- markdown report with table of contents, stats tables, top 25 rankings, full decklists, bracket diffs, and embedding analysis

Use `--output-dir ./my_folder` to control where files go.

Terminal output is kept compact -- one summary line per bracket with card count, game changers, CMC, price, and buy count.

## Tests

```bash
pytest  # 95 tests
```

Covers slug generation, feature vectors, CSV parsing, db operations, bracket constraints, deck legality invariants (99 cards, singleton, color identity, no commander in 99), GNN shapes, and a full integration test.

## Known limitations

- Archetype embeddings are unsupervised - they'd be better with labeled theme data from EDHREC
- No mana curve or creature count constraints in the selector
- No budget cap option (would be a knapsack problem on top of synergy scores)
- Partner commanders with combined EDHREC pages aren't handled yet

## License

MIT
