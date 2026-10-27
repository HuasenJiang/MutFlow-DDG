from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from mutflow_ddg.device import describe_device, resolve_device


AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


def clean_name(value) -> str:
    name = Path(str(value)).name
    return name[:-4] if name.lower().endswith(".pdb") else name


def sequence_from_record(record) -> str:
    features = torch.as_tensor(record["node_features"])
    if features.ndim != 2 or features.shape[1] < 20:
        raise ValueError("Expected a 20-column residue one-hot prefix")
    indices = features[:, :20].argmax(dim=-1)
    present = features[:, :20].max(dim=-1).values > 0
    return "".join(
        AA_ORDER[index] if valid else "X"
        for index, valid in zip(indices.tolist(), present.tolist())
    )


def window_starts(length: int, window: int, overlap: int) -> list[int]:
    if length <= window:
        return [0]
    stride = window - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than the window length")
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def embed_sequence(sequence, tokenizer, model, device, overlap: int):
    max_residues = int(model.config.max_position_embeddings) - 2
    hidden_size = int(model.config.hidden_size)
    output = torch.zeros((len(sequence), hidden_size), dtype=torch.float32)
    counts = torch.zeros((len(sequence), 1), dtype=torch.float32)
    for start in window_starts(len(sequence), max_residues, overlap):
        fragment = sequence[start : start + max_residues]
        tokens = tokenizer(
            fragment,
            return_tensors="pt",
            add_special_tokens=True,
        )
        tokens = {name: value.to(device) for name, value in tokens.items()}
        hidden = model(**tokens).last_hidden_state[0, 1:-1].float().cpu()
        output[start : start + len(fragment)] += hidden
        counts[start : start + len(fragment)] += 1
    return output / counts.clamp_min(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-residue ESM2 representations for CoMDDG pickles"
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-name",
        default="facebook/esm2_t33_650M_UR50D",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "Install the ESM extraction dependency with "
            "`pip install -e '.[esm]'`"
        ) from error

    device = resolve_device(args.device)
    print(f"Using device: {describe_device(device)}")
    with Path(args.input_pkl).open("rb") as handle:
        items = pickle.load(handle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).eval().to(device)

    records = {}
    for item in items:
        for side, name_key in (
            ("wild_type", "wild_type_name"),
            ("mutant", "mutant_name"),
        ):
            records.setdefault(clean_name(item[name_key]), item[side])
    total = len(records)
    for index, (name, record) in enumerate(records.items(), start=1):
        target_dir = output_dir / name
        target = target_dir / "esm2.pt"
        if target.exists() and not args.overwrite:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        sequence = sequence_from_record(record)
        representation = embed_sequence(
            sequence,
            tokenizer=tokenizer,
            model=model,
            device=device,
            overlap=args.overlap,
        )
        torch.save(representation, target)
        if index % 25 == 0 or index == total:
            print(f"processed={index}/{total}")


if __name__ == "__main__":
    main()
