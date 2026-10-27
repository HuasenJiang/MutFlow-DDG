from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import gzip
import json
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_BASE_URL = "https://files.rcsb.org/download"
USER_AGENT = "MutFlow-DDG/0.1 RCSB BioA downloader"
PDB_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")


@dataclasses.dataclass(frozen=True)
class DownloadTarget:
    pdb_id: str
    assembly_id: int
    output_dir: Path
    formats: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class DownloadResult:
    pdb_id: str
    assembly_id: int
    status: str
    path: str
    url: str
    file_format: str | None = None
    error: str | None = None


def _terminal(attribute: str, operator: str, value: Any | None = None) -> dict[str, Any]:
    parameters: dict[str, Any] = {"attribute": attribute, "operator": operator}
    if value is not None:
        parameters["value"] = value
    return {"type": "terminal", "service": "text", "parameters": parameters}


def build_bioa_query(args: argparse.Namespace) -> dict[str, Any]:
    nodes = [
        _terminal("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein"),
        _terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        _terminal(
            "rcsb_accession_info.initial_release_date",
            "less",
            args.release_before,
        ),
        _terminal(
            "rcsb_entry_info.resolution_combined",
            "less_or_equal",
            args.max_resolution,
        ),
        _terminal("refine.ls_R_factor_R_work", "less_or_equal", args.max_r_work),
    ]
    if args.sequence_length_attribute.lower() != "none":
        nodes.append(
            _terminal(
                args.sequence_length_attribute,
                "range",
                {
                    "from": args.min_length,
                    "to": args.max_length,
                    "include_lower": True,
                    "include_upper": True,
                },
            )
        )
    query: dict[str, Any] = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        },
        "return_type": "polymer_entity",
        "request_options": {
            "results_content_type": ["experimental"],
            "results_verbosity": "compact",
            "return_all_hits": True,
        },
    }
    if not args.no_sequence_clustering:
        query["request_options"]["group_by"] = {
            "aggregation_method": "sequence_identity",
            "similarity_cutoff": args.sequence_identity_cutoff,
        }
        query["request_options"]["group_by_return_type"] = "representatives"
    return query


def _without_sequence_length_filter(
    query: dict[str, Any],
    sequence_length_attribute: str,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(query))
    nodes = copied.get("query", {}).get("nodes", [])
    copied["query"]["nodes"] = [
        node
        for node in nodes
        if node.get("parameters", {}).get("attribute") != sequence_length_attribute
    ]
    return copied


def _without_sequence_clustering(query: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(query))
    request_options = copied.get("request_options", {})
    request_options.pop("group_by", None)
    request_options.pop("group_by_return_type", None)
    return copied


def _query_variants(args: argparse.Namespace, query: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [query]
    if args.sequence_length_attribute.lower() != "none":
        variants.append(_without_sequence_length_filter(query, args.sequence_length_attribute))
    if not args.no_sequence_clustering:
        variants.append(_without_sequence_clustering(query))
        if args.sequence_length_attribute.lower() != "none":
            variants.append(
                _without_sequence_clustering(
                    _without_sequence_length_filter(
                        query,
                        args.sequence_length_attribute,
                    )
                )
            )

    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for variant in variants:
        key = json.dumps(variant, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(variant)
    return deduplicated


def _query_has_sequence_length_filter(
    query: dict[str, Any] | None,
    sequence_length_attribute: str,
) -> bool:
    if query is None or sequence_length_attribute.lower() == "none":
        return False
    nodes = query.get("query", {}).get("nodes", [])
    return any(
        node.get("parameters", {}).get("attribute") == sequence_length_attribute
        for node in nodes
    )


def _query_has_sequence_clustering(query: dict[str, Any] | None) -> bool:
    if query is None:
        return False
    return "group_by" in query.get("request_options", {})


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read()
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RCSB search failed with HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"RCSB search failed: {exc}") from exc
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def _extract_identifier(hit: Any) -> str | None:
    if isinstance(hit, str):
        return hit
    if isinstance(hit, dict):
        identifier = hit.get("identifier")
        if isinstance(identifier, str):
            return identifier
    return None


def _extract_result_ids(response: dict[str, Any]) -> list[str]:
    result_set = response.get("result_set") or []
    identifiers = [
        identifier
        for identifier in (_extract_identifier(hit) for hit in result_set)
        if identifier is not None
    ]
    if identifiers:
        return identifiers

    group_ids: list[str] = []
    for group in response.get("group_set") or []:
        for hit in group.get("result_set") or []:
            identifier = _extract_identifier(hit)
            if identifier is not None:
                group_ids.append(identifier)
    return group_ids


def _pdb_id_from_search_identifier(identifier: str) -> str | None:
    pdb_id = re.split(r"[_\-.]", identifier, maxsplit=1)[0].upper()
    if PDB_ID_RE.fullmatch(pdb_id):
        return pdb_id
    return None


def query_bioa_pdb_ids(
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any], dict[str, Any], list[str]]:
    query = build_bioa_query(args)
    query_errors: list[str] = []
    response: dict[str, Any] | None = None
    for index, candidate in enumerate(_query_variants(args, query)):
        if index > 0 and not args.allow_search_fallbacks:
            break
        if args.progress != "none":
            message = "Querying RCSB Search API"
            if index > 0:
                message += f" with fallback #{index}"
            print(f"{message}...", flush=True)
        try:
            response = post_json(args.search_url, candidate, args.timeout)
            query = candidate
            break
        except RuntimeError as exc:
            query_errors.append(str(exc))
    if response is None:
        joined = "\n\n".join(query_errors)
        raise RuntimeError(f"RCSB search failed for all query variants:\n{joined}")
    identifiers = _extract_result_ids(response)
    pdb_ids: list[str] = []
    seen: set[str] = set()
    for identifier in identifiers:
        pdb_id = _pdb_id_from_search_identifier(identifier)
        if pdb_id is None or pdb_id in seen:
            continue
        seen.add(pdb_id)
        pdb_ids.append(pdb_id)
    if args.progress != "none":
        print(f"RCSB query returned {len(pdb_ids)} unique PDB IDs.", flush=True)
    return pdb_ids, query, response, query_errors


def _clean_id_token(value: str) -> str | None:
    token = value.strip().strip(",;")
    if not token:
        return None
    token = Path(token).name
    stem = Path(token).stem
    for candidate in (token, stem):
        if PDB_ID_RE.fullmatch(candidate):
            return candidate.upper()
        match = re.fullmatch(r"(?:pdb)?([A-Za-z0-9]{4})(?:[_\-.].*)?", candidate)
        if match:
            return match.group(1).upper()
    return None


def read_pdb_ids(path: Path) -> list[str]:
    pdb_ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        for raw_token in re.split(r"[\s,;]+", line):
            pdb_id = _clean_id_token(raw_token)
            if pdb_id is None or pdb_id in seen:
                continue
            seen.add(pdb_id)
            pdb_ids.append(pdb_id)
    return pdb_ids


def parse_pdb_ids(value: str) -> list[str]:
    path = Path(value).expanduser()
    if path.exists():
        return read_pdb_ids(path)
    pdb_ids: list[str] = []
    seen: set[str] = set()
    for raw_token in re.split(r"[\s,;]+", value):
        pdb_id = _clean_id_token(raw_token)
        if pdb_id is None:
            raise ValueError(
                f"Invalid PDB ID token {raw_token!r}; pass a file path or "
                "comma-separated 4-character IDs"
            )
        if pdb_id not in seen:
            seen.add(pdb_id)
            pdb_ids.append(pdb_id)
    return pdb_ids


def read_exclude_ids(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            print(
                f"warning: exclude list does not exist, continuing: {path}",
                file=sys.stderr,
            )
            continue
        excluded.update(read_pdb_ids(path))
    return excluded


def read_failed_ids(report_path: Path) -> set[str]:
    failed: set[str] = set()
    if not report_path.exists():
        return failed
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("status") not in {None, "failed"}:
            continue
        pdb_id = _clean_id_token(str(item.get("pdb_id", "")))
        if pdb_id is not None:
            failed.add(pdb_id)
    return failed


def _download_url(base_url: str, pdb_id: str, assembly_id: int, fmt: str) -> str:
    if fmt == "pdb":
        return f"{base_url.rstrip('/')}/{pdb_id}.pdb{assembly_id}.gz"
    if fmt == "cif":
        return f"{base_url.rstrip('/')}/{pdb_id}-assembly{assembly_id}.cif"
    raise ValueError(f"unsupported format: {fmt}")


def _output_path(out: Path, pdb_id: str, assembly_id: int, fmt: str) -> Path:
    suffix = "pdb" if fmt == "pdb" else "cif"
    return out / "assemblies" / f"{pdb_id}_assembly{assembly_id}.{suffix}"


def _download_formats(args: argparse.Namespace) -> tuple[str, ...]:
    if args.format == "auto":
        return ("pdb", "cif")
    return (args.format,)


def _candidate_downloads(
    target: DownloadTarget,
    base_url: str,
) -> list[tuple[str, str, Path]]:
    return [
        (
            fmt,
            _download_url(base_url, target.pdb_id, target.assembly_id, fmt),
            _output_path(target.output_dir, target.pdb_id, target.assembly_id, fmt),
        )
        for fmt in target.formats
    ]


def build_download_targets(args: argparse.Namespace, pdb_ids: list[str]) -> list[DownloadTarget]:
    output_dir = Path(args.out)
    formats = _download_formats(args)
    return [
        DownloadTarget(
            pdb_id=pdb_id,
            assembly_id=args.assembly_id,
            output_dir=output_dir,
            formats=formats,
        )
        for pdb_id in pdb_ids
    ]


def _copy_response_to_file(response: Any, output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            if fmt == "pdb":
                with gzip.GzipFile(fileobj=response) as gzip_handle:
                    shutil.copyfileobj(gzip_handle, handle, length=1024 * 1024)
            else:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        if tmp_path.stat().st_size == 0:
            raise RuntimeError("downloaded file is empty")
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def download_one(
    target: DownloadTarget,
    download_base_url: str,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> DownloadResult:
    candidates = _candidate_downloads(target, download_base_url)
    if not overwrite:
        for fmt, url, output_path in candidates:
            if output_path.exists() and output_path.stat().st_size > 0:
                return DownloadResult(
                    pdb_id=target.pdb_id,
                    assembly_id=target.assembly_id,
                    status="existing",
                    path=str(output_path),
                    url=url,
                    file_format=fmt,
                )

    last_error: str | None = None
    attempted: list[str] = []
    for fmt, url, output_path in candidates:
        attempted.append(url)
        for attempt in range(1, retries + 1):
            try:
                req = request.Request(
                    url,
                    headers={"User-Agent": USER_AGENT},
                )
                with request.urlopen(req, timeout=timeout) as response:
                    _copy_response_to_file(response, output_path, fmt)
                return DownloadResult(
                    pdb_id=target.pdb_id,
                    assembly_id=target.assembly_id,
                    status="downloaded",
                    path=str(output_path),
                    url=url,
                    file_format=fmt,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt < retries:
                    time.sleep(min(2**attempt, 30))
    if len(attempted) > 1 and last_error is not None:
        last_error = (
            f"{last_error}; attempted {len(attempted)} URLs: "
            + ", ".join(attempted)
        )
    return DownloadResult(
        pdb_id=target.pdb_id,
        assembly_id=target.assembly_id,
        status="failed",
        path=str(candidates[-1][2] if candidates else target.output_dir),
        url=candidates[-1][1] if candidates else "",
        file_format=candidates[-1][0] if candidates else None,
        error=last_error,
    )


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def split_train_validation(
    paths: list[str],
    validation_size: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    validation_size = min(max(validation_size, 0), len(shuffled))
    validation = sorted(shuffled[:validation_size])
    train = sorted(shuffled[validation_size:])
    return train, validation


def write_manifests(
    args: argparse.Namespace,
    query: dict[str, Any] | None,
    response: dict[str, Any] | None,
    query_errors: list[str],
    selected_pdb_ids: list[str],
    targets: list[DownloadTarget],
    results: list[DownloadResult],
    excluded_count: int,
) -> dict[str, str]:
    out = Path(args.out)
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    selected_files = [
        result.path
        for result in results
        if result.status in {"downloaded", "existing", "dry_run"}
    ]
    train_files, validation_files = split_train_validation(
        selected_files,
        args.validation_size,
        args.seed,
    )

    paths = {
        "pdb_ids": str(manifest_dir / "rcsb_bioa_pdb_ids.txt"),
        "assemblies": str(manifest_dir / "rcsb_bioa_assemblies.txt"),
        "files": str(manifest_dir / "rcsb_bioa_files.txt"),
        "train_files": str(manifest_dir / "rcsb_bioa_train_files.txt"),
        "validation_files": str(manifest_dir / "rcsb_bioa_validation_files.txt"),
        "query_json": str(manifest_dir / "rcsb_bioa_query.json"),
        "search_response_json": str(manifest_dir / "rcsb_bioa_search_response.json"),
        "download_report_json": str(manifest_dir / "download_report.json"),
        "failed_downloads_jsonl": str(manifest_dir / "failed_downloads.jsonl"),
    }
    write_lines(Path(paths["pdb_ids"]), selected_pdb_ids)
    write_lines(
        Path(paths["assemblies"]),
        [f"{target.pdb_id}-{target.assembly_id}" for target in targets],
    )
    write_lines(Path(paths["files"]), sorted(selected_files))
    write_lines(Path(paths["train_files"]), train_files)
    write_lines(Path(paths["validation_files"]), validation_files)

    if query is not None:
        Path(paths["query_json"]).write_text(json.dumps(query, indent=2), encoding="utf-8")
    if response is not None:
        Path(paths["search_response_json"]).write_text(
            json.dumps(response, indent=2),
            encoding="utf-8",
        )

    failed = [dataclasses.asdict(result) for result in results if result.status == "failed"]
    Path(paths["failed_downloads_jsonl"]).write_text(
        "".join(json.dumps(item) + "\n" for item in failed),
        encoding="utf-8",
    )

    counts = Counter(result.status for result in results)
    report = {
        "out": str(out),
        "format": args.format,
        "assembly_id": args.assembly_id,
        "query_used_sequence_length_filter": _query_has_sequence_length_filter(
            query,
            args.sequence_length_attribute,
        ),
        "query_used_sequence_clustering": _query_has_sequence_clustering(query),
        "query_fallback_errors": query_errors,
        "selected_pdb_ids": len(selected_pdb_ids),
        "excluded_pdb_ids": excluded_count,
        "downloads": dict(sorted(counts.items())),
        "validation_size": len(validation_files),
        "train_size": len(train_files),
        "manifests": paths,
    }
    Path(paths["download_report_json"]).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return paths


def run_downloads(args: argparse.Namespace, targets: list[DownloadTarget]) -> list[DownloadResult]:
    journal_path = Path(args.out) / "manifests" / "download_results.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        if args.progress != "none":
            print(f"Dry run: prepared {len(targets)} download targets.", flush=True)
        results = []
        for target in targets:
            candidates = _candidate_downloads(target, args.download_base_url)
            fmt, url, output_path = candidates[0]
            results.append(
                DownloadResult(
                    pdb_id=target.pdb_id,
                    assembly_id=target.assembly_id,
                    status="dry_run",
                    path=str(output_path),
                    url=url,
                    file_format=fmt,
                )
            )
        return results

    journal_handle = journal_path.open("w", encoding="utf-8")
    if args.progress != "none":
        print(
            f"Downloading {len(targets)} RCSB assembly files with "
            f"{args.workers} workers...",
            flush=True,
        )
    results: list[DownloadResult] = []
    counts: Counter[str] = Counter()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        progress_mode = args.progress
        if progress_mode == "bar" and tqdm is None:
            print(
                "tqdm is not installed; falling back to JSON progress. "
                "Install tqdm for a progress bar.",
                flush=True,
            )
            progress_mode = "json"

        progress_bar = None
        if progress_mode == "bar":
            progress_bar = tqdm(
                total=len(targets),
                desc="RCSB assemblies",
                unit="file",
                dynamic_ncols=True,
            )
        try:
            target_iter = iter(targets)
            max_pending = max(args.workers * args.prefetch, args.workers)
            pending: set[futures.Future[DownloadResult]] = set()

            def submit_more() -> None:
                while len(pending) < max_pending:
                    try:
                        target = next(target_iter)
                    except StopIteration:
                        break
                    pending.add(
                        executor.submit(
                            download_one,
                            target,
                            args.download_base_url,
                            args.overwrite,
                            args.timeout,
                            args.retries,
                        )
                    )

            submit_more()
            completed = 0
            while pending:
                done, pending = futures.wait(
                    pending,
                    return_when=futures.FIRST_COMPLETED,
                )
                for future in done:
                    completed += 1
                    result = future.result()
                    results.append(result)
                    journal_handle.write(json.dumps(dataclasses.asdict(result)) + "\n")
                    journal_handle.flush()
                    counts[result.status] += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            downloaded=counts.get("downloaded", 0),
                            existing=counts.get("existing", 0),
                            failed=counts.get("failed", 0),
                        )
                    elif (
                        progress_mode == "json"
                        and args.progress_interval > 0
                        and completed % args.progress_interval == 0
                    ):
                        print(
                            json.dumps(
                                {
                                    "completed": completed,
                                    "total": len(targets),
                                    "status": dict(sorted(counts.items())),
                                }
                            ),
                            flush=True,
                        )
                submit_more()
        finally:
            if progress_bar is not None:
                progress_bar.close()
            journal_handle.close()
        if progress_mode == "json" and args.progress_interval > 0:
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "total": len(targets),
                        "status": dict(sorted(counts.items())),
                    }
                ),
                flush=True,
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download an RCSB BioA-like biological assembly set for MutFlow-DDG "
            "structure pretraining."
        )
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for raw files and manifests.",
    )
    parser.add_argument(
        "--exclude-pdb-ids",
        action="append",
        default=[],
        help="File containing PDB IDs to exclude. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pdb-ids",
        default=None,
        help="Optional PDB ID file or comma-separated IDs. When set, skips the RCSB search.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after filtering.")
    parser.add_argument("--assembly-id", type=int, default=1)
    parser.add_argument(
        "--format",
        choices=["auto", "pdb", "cif"],
        default="auto",
        help=(
            "Download format. auto tries PDB first and falls back to mmCIF."
        ),
    )
    parser.add_argument("--release-before", default="2020-05-01")
    parser.add_argument("--max-resolution", type=float, default=2.5)
    parser.add_argument("--max-r-work", type=float, default=0.25)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=10000)
    parser.add_argument(
        "--sequence-length-attribute",
        default="entity_poly.rcsb_sample_sequence_length",
        help=(
            "RCSB Search API attribute used for polymer length filtering. "
            "Use 'none' to disable the length filter."
        ),
    )
    parser.add_argument(
        "--allow-search-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If RCSB rejects the strict query, retry less strict variants "
            "without sequence-length filtering and/or sequence clustering."
        ),
    )
    parser.add_argument(
        "--sequence-identity-cutoff",
        type=int,
        choices=[30, 50, 70, 90, 95, 100],
        default=50,
    )
    parser.add_argument(
        "--no-sequence-clustering",
        action="store_true",
        help="Disable RCSB sequence-identity representative grouping.",
    )
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--prefetch",
        type=int,
        default=4,
        help="Maximum queued download tasks per worker.",
    )
    parser.add_argument(
        "--progress",
        choices=["bar", "json", "none"],
        default="bar",
        help="Progress display mode. Default: tqdm progress bar.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Only retry PDB IDs listed in manifests/failed_downloads.jsonl.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--search-url", default=SEARCH_URL)
    parser.add_argument("--download-base-url", default=DOWNLOAD_BASE_URL)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.assembly_id < 1:
        parser.error("--assembly-id must be >= 1")
    if args.retries < 1:
        parser.error("--retries must be >= 1")
    if args.prefetch < 1:
        parser.error("--prefetch must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()
    args.out = str(out)
    (out / "assemblies").mkdir(parents=True, exist_ok=True)

    excluded = read_exclude_ids(args.exclude_pdb_ids)
    query: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    query_errors: list[str] = []
    if args.pdb_ids:
        pdb_ids = parse_pdb_ids(args.pdb_ids)
    else:
        pdb_ids, query, response, query_errors = query_bioa_pdb_ids(args)

    before_exclusion = len(pdb_ids)
    pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id not in excluded]
    excluded_count = before_exclusion - len(pdb_ids)
    if args.retry_failed_only:
        failed_report_path = out / "manifests" / "failed_downloads.jsonl"
        failed_ids = read_failed_ids(failed_report_path)
        if not failed_ids:
            failed_report_path = out / "manifests" / "download_results.jsonl"
            failed_ids = read_failed_ids(failed_report_path)
        pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id in failed_ids]
        if args.progress != "none":
            print(
                f"Retrying {len(pdb_ids)} PDB IDs from {failed_report_path}.",
                flush=True,
            )
    if args.limit is not None:
        pdb_ids = pdb_ids[: args.limit]
    if args.progress != "none":
        print(
            f"Selected {len(pdb_ids)} PDB IDs after exclusions "
            f"({excluded_count} excluded).",
            flush=True,
        )

    targets = build_download_targets(args, pdb_ids)
    results = run_downloads(args, targets)
    results = sorted(results, key=lambda item: (item.pdb_id, item.assembly_id))
    paths = write_manifests(
        args=args,
        query=query,
        response=response,
        query_errors=query_errors,
        selected_pdb_ids=pdb_ids,
        targets=targets,
        results=results,
        excluded_count=excluded_count,
    )

    counts = Counter(result.status for result in results)
    summary = {
        "selected_pdb_ids": len(pdb_ids),
        "excluded_pdb_ids": excluded_count,
        "downloads": dict(sorted(counts.items())),
        "files_manifest": paths["files"],
        "train_manifest": paths["train_files"],
        "validation_manifest": paths["validation_files"],
        "report": paths["download_report_json"],
    }
    print(json.dumps(summary, indent=2))
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
