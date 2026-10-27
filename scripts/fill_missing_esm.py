from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Iterable

import torch

from mutflow_ddg.device import describe_device, resolve_device


AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


@dataclass(frozen=True)
class OutputRecord:
    path: Path
    metadata: dict[str, Any]


def clean_name(value: Any) -> str:
    name = Path(str(value)).name
    return name[:-4] if name.lower().endswith(".pdb") else name


def sequence_from_record(record: dict[str, Any]) -> str:
    features = torch.as_tensor(record["node_features"])
    if features.ndim != 2 or features.shape[1] < 20:
        raise ValueError("Expected a 20-column residue one-hot prefix")
    residue_features = features[:, :20]
    indices = residue_features.argmax(dim=-1)
    present = residue_features.max(dim=-1).values > 0
    return "".join(
        AA_ORDER[index] if valid else "X"
        for index, valid in zip(indices.tolist(), present.tolist())
    )


def paired_flat_names(item: dict[str, Any]) -> tuple[str, str]:
    mutant_name = clean_name(item["mutant_name"])
    if "_mutant_" not in mutant_name:
        raise ValueError(
            "Cannot derive paired-flat WT name from mutant name "
            f"{mutant_name!r}"
        )
    return mutant_name.replace("_mutant_", "_wild_", 1), mutant_name


def output_paths(
    item: dict[str, Any],
    output_dir: Path,
    layout: str,
) -> tuple[Path, Path]:
    if layout == "paired_flat":
        names = paired_flat_names(item)
        return tuple(output_dir / f"{name}_esm2.pt" for name in names)
    if layout == "standard":
        return tuple(
            output_dir / clean_name(item[f"{side}_name"]) / "esm2.pt"
            for side in ("wild_type", "mutant")
        )
    raise ValueError(f"Unsupported ESM cache layout: {layout!r}")


def metadata_for(
    item: dict[str, Any],
    side: str,
    output_name: str,
) -> dict[str, Any]:
    mutation = output_name.rsplit("_", 1)[-1]
    return {
        "mutation": mutation,
        "type": "wild" if side == "wild_type" else "mutant",
        "ddg": float(item["ddg"]),
        "model": "esm2_t33_650M_UR50D",
        "original_name": item[f"{side}_name"],
    }


def collect_missing_outputs(
    input_paths: Iterable[Path],
    output_dir: Path,
    existing_dirs: Iterable[Path] = (),
    layout: str = "paired_flat",
) -> dict[str, list[OutputRecord]]:
    outputs_by_sequence: dict[str, list[OutputRecord]] = defaultdict(list)
    scheduled_paths: set[Path] = set()
    for input_path in input_paths:
        with input_path.open("rb") as handle:
            items = pickle.load(handle)
        for item in items:
            paths = output_paths(item, output_dir, layout)
            existing_paths = [
                output_paths(item, directory, layout)
                for directory in existing_dirs
            ]
            for side_index, (side, output_path) in enumerate(
                zip(("wild_type", "mutant"), paths)
            ):
                if output_path in scheduled_paths:
                    continue
                if output_path.is_file() or any(
                    candidates[side_index].is_file()
                    for candidates in existing_paths
                ):
                    continue
                scheduled_paths.add(output_path)
                sequence = sequence_from_record(item[side])
                outputs_by_sequence[sequence].append(
                    OutputRecord(
                        path=output_path,
                        metadata=metadata_for(
                            item,
                            side,
                            clean_name(item[f"{side}_name"]),
                        ),
                    )
                )
    return outputs_by_sequence


def token_batches(
    sequences: Iterable[str],
    max_tokens: int,
) -> Iterable[list[str]]:
    batch: list[str] = []
    max_length = 0
    for sequence in sorted(sequences, key=len):
        next_max_length = max(max_length, len(sequence))
        if batch and next_max_length * (len(batch) + 1) > max_tokens:
            yield batch
            batch = []
            max_length = 0
        batch.append(sequence)
        max_length = max(max_length, len(sequence))
    if batch:
        yield batch


def save_representation(
    output: OutputRecord,
    sequence: str,
    representation: torch.Tensor,
) -> None:
    payload = {
        "features": representation.float().cpu().numpy(),
        "sequence": sequence,
        "metadata": output.metadata,
    }
    output.path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output.path.with_suffix(output.path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output.path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing paired-flat ESM2 caches from CoMDDG pickles"
    )
    parser.add_argument("--input-pkl", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--layout",
        choices=("standard", "paired_flat"),
        default="paired_flat",
        help="Cache naming convention (default: paired_flat).",
    )
    parser.add_argument(
        "--existing-dir",
        action="append",
        default=[],
        help="Fallback cache roots whose existing files should not be regenerated.",
    )
    parser.add_argument(
        "--model-path",
        default=(
            str(Path(torch.hub.get_dir()) / "checkpoints" / "esm2_t33_650M_UR50D.pt")
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_by_sequence = collect_missing_outputs(
        [Path(path) for path in args.input_pkl],
        output_dir,
        [Path(path) for path in args.existing_dir],
        layout=args.layout,
    )
    total_files = sum(map(len, outputs_by_sequence.values()))
    print(
        f"Missing files: {total_files}; "
        f"unique sequences: {len(outputs_by_sequence)}"
    )
    if not outputs_by_sequence:
        return

    try:
        import esm
    except ImportError as error:
        raise ImportError(
            "Install fair-esm or run this script in an environment that "
            "provides the `esm` package"
        ) from error

    device = resolve_device(args.device)
    print(f"Loading local ESM2 model on {describe_device(device)}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(
        args.model_path
    )
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    completed_files = 0
    batches = list(token_batches(outputs_by_sequence, args.max_tokens))
    with torch.inference_mode():
        for batch_index, sequences in enumerate(batches, start=1):
            data = [
                (f"sequence_{index}", sequence)
                for index, sequence in enumerate(sequences)
            ]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                result = model(
                    tokens,
                    repr_layers=[33],
                    return_contacts=False,
                )
            representations = result["representations"][33]
            for row, sequence in enumerate(sequences):
                representation = representations[row, 1 : len(sequence) + 1]
                for output in outputs_by_sequence[sequence]:
                    save_representation(output, sequence, representation)
                    completed_files += 1
            print(
                f"batch={batch_index}/{len(batches)} "
                f"files={completed_files}/{total_files}"
            )


if __name__ == "__main__":
    main()
