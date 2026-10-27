from __future__ import annotations

import csv
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


RECORD_RE = re.compile(r"^(.+)_([^_]+)_(wild_type|mutant_(.+))$")
CHAINLESS_RECORD_RE = re.compile(r"^(.+)_(wild_type|wild|mutant_(.+))$")
MUTATION_RE = re.compile(r"^([A-Z])(-?\d+[A-Z]?)([A-Z])$")


def clean_record_name(value: Any) -> str:
    name = Path(str(value)).name
    return name[:-4] if name.lower().endswith(".pdb") else name


def parse_record_name(value: Any) -> dict[str, str]:
    name = clean_record_name(value)
    match = RECORD_RE.match(name)
    if match is None:
        chainless = CHAINLESS_RECORD_RE.match(name)
        if chainless is None:
            return {
                "record_name": name,
                "protein_id": "",
                "chain": "",
                "mutation": "",
                "record_kind": "",
            }
        protein_id, kind, mutation = chainless.groups()
        return {
            "record_name": name,
            "protein_id": protein_id,
            "chain": "",
            "mutation": mutation or "",
            "record_kind": "wild_type" if kind in {"wild_type", "wild"} else "mutant",
        }
    protein_id, chain, kind, mutation = match.groups()
    return {
        "record_name": name,
        "protein_id": protein_id,
        "chain": chain,
        "mutation": mutation or "",
        "record_kind": "wild_type" if kind == "wild_type" else "mutant",
    }


def mutation_from_row(row: dict[str, Any]) -> str:
    mutation = str(row.get("Mutation_PDB", "")).strip()
    if mutation:
        return mutation
    return (
        f"{str(row.get('WT', '')).strip()}"
        f"{str(row.get('POS', '')).strip()}"
        f"{str(row.get('MT', '')).strip()}"
    )


def ddg_from_row(row: dict[str, Any]) -> float:
    for name in ("DDGexp", "ddG", "ddg", "DDG"):
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise ValueError("baseline row has no DDG column")


def canonical_mutation_id(
    dataset: str,
    source_row_index: int,
    protein_id: str,
    chain: str,
    mutation: str,
) -> str:
    """Return a unique, baseline-row anchored mutation identifier."""

    return (
        f"{dataset.lower()}:{source_row_index:06d}:"
        f"{protein_id.upper()}:{chain}:{mutation.upper()}"
    )


def baseline_rows(path: str | Path, dataset: str) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for index, row in enumerate(rows):
        protein_id = str(row.get("PDB Id", "")).strip()
        chain = str(row.get("Mutated Chain", "")).strip()
        mutation = mutation_from_row(row)
        mutation_aliases = list(
            dict.fromkeys(
                value
                for value in (
                    mutation,
                    str(row.get("Mutation_UNP", "")).strip(),
                    str(row.get("Mut_seq", "")).strip(),
                )
                if value
            )
        )
        result.append(
            {
                "dataset": dataset.lower(),
                "source_row_index": index,
                "protein_id": protein_id,
                "chain": chain,
                "mutation": mutation,
                "mutation_aliases": mutation_aliases,
                "ddg": ddg_from_row(row),
                "mutation_id": canonical_mutation_id(
                    dataset, index, protein_id, chain, mutation
                ),
            }
        )
    return result


def _rounded_ddg(value: Any) -> str:
    return f"{float(value):.6g}"


def match_items_to_baseline(
    items: Iterable[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any] | None]:
    """Match legacy pickle items to baseline rows without assuming ordering."""

    exact: dict[tuple[str, str, str, str], deque[dict[str, Any]]] = defaultdict(
        deque
    )
    identity: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(
        deque
    )
    relaxed_identity: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(
        deque
    )
    mutation_ddg_identity: dict[
        tuple[str, str], deque[dict[str, Any]]
    ] = defaultdict(deque)
    for row in rows:
        key = (row["protein_id"], row["chain"], row["mutation"])
        exact[(*key, _rounded_ddg(row["ddg"]))].append(row)
        identity[key].append(row)
        for mutation_alias in row.get("mutation_aliases", [row["mutation"]]):
            relaxed_identity[(row["protein_id"], mutation_alias)].append(row)
            mutation_ddg_identity[
                (mutation_alias, _rounded_ddg(abs(float(row["ddg"]))))
            ].append(row)

    used: set[str] = set()
    matched: list[dict[str, Any] | None] = []
    for item in items:
        parts = parse_record_name(item.get("mutant_name", ""))
        key = (parts["protein_id"], parts["chain"], parts["mutation"])
        candidates = exact.get((*key, _rounded_ddg(item.get("ddg", 0))))
        row = None
        while candidates:
            candidate = candidates.popleft()
            if candidate["mutation_id"] not in used:
                row = candidate
                break
        if row is None:
            while identity.get(key):
                candidate = identity[key].popleft()
                if candidate["mutation_id"] not in used:
                    row = candidate
                    break
        if row is None:
            relaxed_key = (parts["protein_id"], parts["mutation"])
            while relaxed_identity.get(relaxed_key):
                candidate = relaxed_identity[relaxed_key].popleft()
                if candidate["mutation_id"] not in used:
                    row = candidate
                    break
        if row is None:
            mutation_ddg_key = (
                parts["mutation"],
                _rounded_ddg(abs(float(item.get("ddg", 0)))),
            )
            while mutation_ddg_identity.get(mutation_ddg_key):
                candidate = mutation_ddg_identity[mutation_ddg_key].popleft()
                if candidate["mutation_id"] not in used:
                    row = candidate
                    break
        if row is not None:
            used.add(row["mutation_id"])
        matched.append(row)
    return matched


def parse_mutation(value: str) -> tuple[str, str, str]:
    match = MUTATION_RE.match(value.upper())
    if match is None:
        raise ValueError(f"unsupported mutation notation: {value!r}")
    return match.group(1), match.group(2), match.group(3)


def read_allowed_mutation_ids(
    path: str | Path,
    dataset: str | None = None,
) -> set[str]:
    """Read success/common-subset IDs from a CSV manifest."""

    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "mutation_id" not in rows[0]:
        raise ValueError(f"manifest has no mutation_id column: {path}")
    result = set()
    for row in rows:
        if dataset and row.get("dataset", "").lower() not in {"", dataset.lower()}:
            continue
        if row.get("status", "success").lower() not in {
            "success",
            "common",
            "included",
        }:
            continue
        result.add(row["mutation_id"])
    return result
