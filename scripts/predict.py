"""Single-checkpoint CSV inference on prepared, trusted paired graph inputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from mutflow_ddg.config import load_config
from mutflow_ddg.data import CoMDDGPairDataset
from mutflow_ddg.device import describe_device, resolve_device
from mutflow_ddg.engine import load_torch_checkpoint
from mutflow_ddg.models import MutFlowDDG
from scripts.train import build_loader, validate_dataset_dimensions


FIELDS = [
    'dataset', 'mutation_id', 'direction', 'target', 'prediction', 'uncertainty',
    'p_stabilizing', 'p_neutral', 'p_destabilizing', 'rank_score',
]


def prediction_rows(batch, output, *, dataset_name, direction, offset=0, unlabeled=False):
    means = output['ddg'].detach().cpu().tolist()
    sigmas = torch.exp(0.5 * output['log_variance']).detach().cpu().tolist()
    probabilities = output['class_logits'].softmax(-1).detach().cpu().tolist()
    scores = output['rank_score'].detach().cpu().tolist()
    targets = batch.ddg.detach().cpu().tolist()
    identifiers = batch.mutation_ids
    if identifiers is None:
        identifiers = [f'{dataset_name}:input:{offset + i}' for i in range(len(means))]
    if len(identifiers) != len(means):
        raise ValueError('Prediction and mutation-ID counts differ')
    for i, value in enumerate(means):
        yield {
            'dataset': dataset_name, 'mutation_id': identifiers[i],
            'direction': direction, 'target': '' if unlabeled else targets[i],
            'prediction': value, 'uncertainty': sigmas[i],
            'p_stabilizing': probabilities[i][0], 'p_neutral': probabilities[i][1],
            'p_destabilizing': probabilities[i][2], 'rank_score': scores[i],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/train.yaml'))
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Trusted trained checkpoint; legacy loading can execute code.')
    parser.add_argument('--input-pkl', type=Path, required=True,
                        help='Trusted, prepared paired-graph pickle.')
    parser.add_argument('--esm-cache-dir', type=Path)
    parser.add_argument('--esm-layout', choices=['standard', 'paired_flat'], default='standard')
    parser.add_argument('--dataset-name', default='prediction')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--unlabeled', action='store_true')
    parser.add_argument('--force', action='store_true', help='Allow replacing the requested CSV.')
    args = parser.parse_args()
    for path in [args.config, args.checkpoint, args.input_pkl]:
        if not path.is_file():
            parser.error(f'Required file is missing: {path}')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.output.exists() and not args.force:
        parser.error('Output already exists; choose another path or pass --force')

    config = load_config(args.config)
    config.data.batch_size = args.batch_size
    sequence = config.model.branch_mode in {'sequence', 'both'}
    uses_esm = sequence and config.data.sequence_feature_source in {'esm', 'esm_pssm'}
    if uses_esm and args.esm_cache_dir is None:
        parser.error('--esm-cache-dir is required for this model')
    dataset = CoMDDGPairDataset(
        args.input_pkl,
        esm_cache_dir=args.esm_cache_dir if uses_esm else None,
        require_esm=uses_esm,
        esm_layout=args.esm_layout,
        sequence_feature_source=config.data.sequence_feature_source if sequence else 'none',
        sequence_feature_dim=config.model.sequence_dim if sequence else None,
        mutation_feature_mode=config.model.mutation_feature_mode,
        dataset_name=args.dataset_name,
    )
    if not len(dataset):
        parser.error('Input dataset is empty')
    validate_dataset_dimensions(dataset, config)
    loader = build_loader(dataset, config, shuffle=False, uses_sequence=sequence, seed=config.seed)
    device = resolve_device(args.device)
    checkpoint = load_torch_checkpoint(args.checkpoint, 'cpu')
    model = MutFlowDDG(config.model)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f'Using device: {describe_device(device)}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = offset = 0
    with args.output.open('w', newline='', encoding='utf-8') as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for batch in loader:
            batch = batch.to(device)
            variants = [('direct', batch)]
            if args.reverse:
                variants.append(('reverse', batch.reversed()))
            for direction, current in variants:
                rows = list(prediction_rows(
                    current, model(current), dataset_name=args.dataset_name,
                    direction=direction, offset=offset, unlabeled=args.unlabeled,
                ))
                writer.writerows(rows)
                count += len(rows)
            offset += batch.ddg.numel()
    print(f'Wrote {count} prediction rows to {args.output}')


if __name__ == '__main__':
    main()
