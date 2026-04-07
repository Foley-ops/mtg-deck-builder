"""Collection CSV tests."""

import tempfile
import os

import pytest

from mtg_deck_builder.collection import load_collection_csv


def _write_csv(content):
    """Write CSV content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class TestLoadCollectionCSV:
    def test_standard_format(self):
        path = _write_csv("name,quantity\nSol Ring,1\nForest,10\n")
        try:
            result = load_collection_csv(path)
            assert result["Sol Ring"] == 1
            assert result["Forest"] == 10
        finally:
            os.unlink(path)

    def test_manabox_column_names(self):
        path = _write_csv("Card,Qty,Set\nSol Ring,2,C21\nPath to Exile,1,MH3\n")
        try:
            result = load_collection_csv(path)
            assert result["Sol Ring"] == 2
            assert result["Path to Exile"] == 1
        finally:
            os.unlink(path)

    def test_card_name_column(self):
        path = _write_csv("card_name,amount\nLightning Bolt,4\n")
        try:
            result = load_collection_csv(path)
            assert result["Lightning Bolt"] == 4
        finally:
            os.unlink(path)

    def test_duplicate_names_sum_quantities(self):
        path = _write_csv("name,quantity\nSol Ring,1\nSol Ring,2\n")
        try:
            result = load_collection_csv(path)
            assert result["Sol Ring"] == 3
        finally:
            os.unlink(path)

    def test_missing_quantity_defaults_to_1(self):
        path = _write_csv("name\nSol Ring\nForest\n")
        try:
            result = load_collection_csv(path)
            assert result["Sol Ring"] == 1
        finally:
            os.unlink(path)

    def test_no_name_column_returns_empty(self):
        path = _write_csv("id,count\n1,5\n")
        try:
            result = load_collection_csv(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_whitespace_stripped(self):
        path = _write_csv("name,quantity\n  Sol Ring  , 3 \n")
        try:
            result = load_collection_csv(path)
            assert result["Sol Ring"] == 3
        finally:
            os.unlink(path)

    def test_bom_handling(self):
        path = _write_csv("\ufeffname,quantity\nSol Ring,1\n")
        try:
            result = load_collection_csv(path)
            assert result["Sol Ring"] == 1
        finally:
            os.unlink(path)
