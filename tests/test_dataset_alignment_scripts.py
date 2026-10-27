from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from scripts.fill_missing_esm import collect_missing_outputs, output_paths
from scripts.select_pythia_by_comddg_pickle import select_items


def _record(length: int = 3) -> dict:
    features = np.zeros((length, 28), dtype=np.float32)
    features[:, 0] = 1
    return {"node_features": features}


def _item(ddg: float = 1.0) -> dict:
    return {
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_A1V.pdb",
        "wild_type": _record(),
        "mutant": _record(),
        "ddg": ddg,
    }


def test_standard_esm_outputs_are_nested_and_deduplicated(tmp_path: Path):
    item = _item()
    input_path = tmp_path / "samples.pkl"
    with input_path.open("wb") as handle:
        pickle.dump([item, item], handle)

    paths = output_paths(item, tmp_path / "esm", "standard")
    assert paths == (
        tmp_path / "esm/1ABC_A_wild_type/esm2.pt",
        tmp_path / "esm/1ABC_A_mutant_A1V/esm2.pt",
    )
    outputs = collect_missing_outputs(
        [input_path], tmp_path / "esm", layout="standard"
    )
    assert sum(len(records) for records in outputs.values()) == 2


def test_select_items_follows_baseline_order_and_keeps_pythia_graphs():
    first = _item(1.0)
    second = {
        **_item(2.0),
        "mutant_name": "1ABC_A_mutant_A2G.pdb",
    }
    pythia_first = {**first, "wild_type": _record(2), "structure_scope": "patch"}
    pythia_second = {**second, "wild_type": _record(4), "structure_scope": "patch"}

    selected, missing = select_items(
        [second, first], [pythia_first, pythia_second]
    )
    assert not missing
    assert [item["ddg"] for item in selected] == [2.0, 1.0]
    assert [len(item["wild_type"]["node_features"]) for item in selected] == [4, 2]
    assert all(item["structure_scope"] == "patch" for item in selected)
