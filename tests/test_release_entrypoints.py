from pathlib import Path

import torch

from mutflow_ddg.config import load_config
from mutflow_ddg.data.synthetic import make_synthetic_batch
from scripts.check_data import required_paths
from scripts.predict import FIELDS, prediction_rows


def test_public_pretraining_explicitly_has_four_objectives():
    config = load_config('configs/pretrain.yaml')
    pre = config.pretraining
    assert pre.objective == 'flow'
    assert pre.mask_ratio == 0.85
    assert pre.random_replace_ratio == 0.15
    assert pre.central_weight == 1.0
    assert pre.flow_weight == 1.0
    assert pre.transport_weight == 0.1
    assert pre.contrastive_weight == 0.2
    assert pre.replacement_weight == pre.geometry_weight == pre.contact_weight == 0
    assert pre.adaptive_loss_weighting is False


def test_public_downstream_matches_five_benchmark_scope():
    config = load_config('configs/train.yaml')
    assert {test.name for test in config.data.external_tests} == {
        's250', 's350', 's605', 's1925', 'ssym',
    }
    assert config.model.antisymmetric_output is False
    assert config.model.transfer_pretrained_flow is False
    assert config.model.structure_checkpoint == load_config(
        'configs/pretrain.yaml'
    ).pretraining.checkpoint_path
    assert not any(path.is_absolute() for _, path, _ in required_paths(config, 'train'))


def test_prediction_rows_export_uncertainty_and_preserve_identifiers():
    batch = make_synthetic_batch(batch_size=2)
    batch.mutation_ids = ['a', 'b']
    output = {
        'ddg': torch.tensor([1.0, -2.0]),
        'log_variance': torch.tensor([0.0, 2.0]),
        'class_logits': torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        'rank_score': torch.tensor([0.5, -0.5]),
    }
    rows = list(prediction_rows(batch, output, dataset_name='toy', direction='direct'))
    assert [row['mutation_id'] for row in rows] == ['a', 'b']
    assert rows[0]['prediction'] == 1.0
    assert rows[0]['uncertainty'] == 1.0
    assert abs(rows[1]['uncertainty'] - torch.exp(torch.tensor(1.0)).item()) < 1e-6
    assert set(rows[0]) == set(FIELDS)
    assert abs(sum(rows[0][key] for key in ['p_stabilizing', 'p_neutral', 'p_destabilizing']) - 1) < 1e-6
    unlabeled = list(prediction_rows(
        batch, output, dataset_name='toy', direction='direct', unlabeled=True,
    ))
    assert all(row['target'] == '' for row in unlabeled)


def test_prediction_rows_label_reverse_targets_correctly():
    batch = make_synthetic_batch(batch_size=1)
    output = {'ddg': torch.tensor([0.2]), 'log_variance': torch.tensor([0.0]),
              'class_logits': torch.zeros(1, 3), 'rank_score': torch.zeros(1)}
    row = next(prediction_rows(
        batch.reversed(), output, dataset_name='toy', direction='reverse',
    ))
    assert row['target'] == -batch.ddg.item()
    assert row['direction'] == 'reverse'


def test_prediction_cli_loads_checkpoint_and_exports_both_directions(tmp_path):
    import csv
    import pickle
    import subprocess
    import sys

    import numpy as np
    import yaml

    from mutflow_ddg.models import MutFlowDDG

    configuration = {
        'data': {'sequence_feature_source': 'pssm', 'require_esm': False},
        'model': {
            'branch_mode': 'both', 'node_dim': 67, 'sequence_dim': 41,
            'edge_dim': 16, 'hidden_dim': 32, 'latent_dim': 32,
            'num_structure_layers': 1, 'num_sequence_layers': 1,
            'num_attention_heads': 4, 'flow_steps': 2, 'dropout': 0.0,
        },
    }
    config_path = tmp_path / 'toy.yaml'
    config_path.write_text(yaml.safe_dump(configuration))
    config = load_config(config_path)
    model = MutFlowDDG(config.model)
    checkpoint = tmp_path / 'toy.pt'
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, checkpoint)
    features = np.zeros((4, 67), dtype=np.float32)
    features[np.arange(4), np.arange(4)] = 1
    record = {
        'node_features': features, 'edge_index': np.empty((2, 0), dtype=np.int64),
        'edge_features': np.empty((0, 16), dtype=np.float32), 'mutation_pos': 1,
    }
    pairs = [{
        'wild_type': record, 'mutant': record,
        'wild_type_name': '1ABC_A_wild_type.pdb',
        'mutant_name': '1ABC_A_mutant_R2A.pdb', 'ddg': 1.25,
    }]
    input_path = tmp_path / 'pairs.pkl'
    with input_path.open('wb') as handle:
        pickle.dump(pairs, handle)
    output = tmp_path / 'predictions.csv'
    completed = subprocess.run([
        sys.executable, '-m', 'scripts.predict', '--config', str(config_path),
        '--checkpoint', str(checkpoint), '--input-pkl', str(input_path),
        '--output', str(output), '--device', 'cpu', '--reverse',
    ], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row['direction'] for row in rows] == ['direct', 'reverse']
    assert [float(row['target']) for row in rows] == [1.25, -1.25]
    assert all(np.isfinite(float(row['prediction'])) for row in rows)
