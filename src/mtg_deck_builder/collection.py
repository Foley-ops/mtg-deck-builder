"""Collection CSV loading."""

import csv
from collections import defaultdict


COLLECTION_TEMPLATE = """\
name,quantity,foil,condition,set_code,binder
Sol Ring,1,false,NM,C21,Precon Cards
Swords to Plowshares,2,false,NM,MH3,Mix
Rhystic Study,1,false,NM,MH3,ATLA
Cyclonic Rift,1,true,NM,MM3,ATLA
Ms. Bumbleflower,1,false,NM,BLC,Precon Cards
Mr. Foxglove,1,false,NM,BLC,Precon Cards
Selvala Explorer Returned,1,false,NM,BLC,Precon Cards
Kwain Itinerant Meddler,1,false,NM,BLC,Precon Cards
Rites of Flourishing,1,false,NM,BLC,Precon Cards
Kalonian Hydra,1,false,NM,BLC,Precon Cards
Managorger Hydra,1,false,NM,BLC,Precon Cards
Body of Knowledge,1,false,NM,BLC,Precon Cards
Psychosis Crawler,1,false,NM,BLC,Precon Cards
Triskaidekaphile,1,false,NM,BLC,Precon Cards
Forest,15,false,NM,,Precon Cards
Plains,10,false,NM,,Precon Cards
Island,10,false,NM,,Precon Cards
"""


def generate_template_csv(path="collection_template.csv"):
    with open(path, "w", newline="") as f:
        f.write(COLLECTION_TEMPLATE)
    print(f"\n  Template saved: {path}")
    print()
    print("  Columns:")
    print("    name       (required) exact English card name")
    print("    quantity   (required) how many you own")
    print("    foil       (optional) true / false")
    print("    condition  (optional) NM / LP / MP / HP / DMG")
    print("    set_code   (optional) 3-letter set code, helps pricing")
    print("    binder     (optional) which binder/deck it lives in")
    print()
    print("  Compatible with ManaBox CSV exports (auto-detects columns).")
    print("  Export from ManaBox: Collection > ... menu > Export > CSV")


def load_collection_csv(path):
    """Load collection CSV -> {card_name: total_quantity}."""
    collection = defaultdict(int)
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"  WARNING: empty CSV {path}")
            return {}

        field_map = {}
        for fn in reader.fieldnames:
            fl = fn.strip().lower().replace(" ", "_")
            if fl in ("name", "card", "card_name", "cardname"):
                field_map["name"] = fn
            elif fl in ("quantity", "qty", "count", "amount"):
                field_map["quantity"] = fn

        if "name" not in field_map:
            print(f"  ERROR: no 'name' column found. Got: {reader.fieldnames}")
            return {}

        for row in reader:
            name = row.get(field_map["name"], "").strip()
            qty_raw = row.get(field_map.get("quantity", ""), "1").strip()
            try:
                qty = int(qty_raw) if qty_raw else 1
            except ValueError:
                qty = 1
            if name:
                collection[name] += qty

    print(f"  {len(collection)} unique cards ({sum(collection.values())} total)")
    return dict(collection)
