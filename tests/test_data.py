from pathlib import Path

import numpy as np
import pytest
import torch

from mutflow_ddg.data import StructureGraphDataset
from mutflow_ddg.data.comddg import (
    CoMDDGPairDataset,
    ESMFeatureStore,
    collate_graph_pairs,
    convert_comddg_item,
)
from scripts.convert_cath_pdb_to_graphs import (
    PYTHIA_EDGE_DIM,
    PYTHIA_NODE_DIM,
    pythia_knn_graph,
    pythia_node_features,
)
from scripts.generate_pythia_structure_from_coordinate_cache import (
    convert_record as convert_pythia_coordinate_record,
)
from scripts.train import AttentionBudgetBatchSampler, resolve_resume_path


def graph_record(length=4, node_dim=67, edge_dim=16):
    node_features = np.zeros((length, node_dim), dtype=np.float32)
    node_features[np.arange(length), np.arange(length)] = 1
    return {
        "node_features": node_features,
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "edge_features": np.empty((0, edge_dim), dtype=np.float32),
        "mutation_pos": 1,
    }


def test_comddg_item_loads_aligned_esm_cache(tmp_path: Path):
    item = {
        "wild_type": graph_record(),
        "mutant": graph_record(),
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_R2A.pdb",
        "ddg": 1.25,
    }
    for name in ("1ABC_A_wild_type", "1ABC_A_mutant_R2A"):
        folder = tmp_path / name
        folder.mkdir()
        torch.save(torch.randn(4, 1280), folder / "esm2.pt")
    sample = convert_comddg_item(
        item,
        esm_store=ESMFeatureStore(tmp_path),
        require_esm=True,
    )
    assert sample.wild.sequence_features.shape == (4, 1280)
    assert sample.wild.residue_type.tolist() == [0, 1, 2, 3]


def test_comddg_item_splits_full_sequence_from_structure_patch(
    tmp_path: Path,
):
    wild_patch = graph_record(length=3)
    mutant_patch = graph_record(length=3)
    wild_patch["mutation_pos"] = 1
    mutant_patch["mutation_pos"] = 1
    wild_sequence = graph_record(length=6)
    mutant_sequence = graph_record(length=6)
    wild_sequence["mutation_pos"] = 4
    mutant_sequence["mutation_pos"] = 4
    item = {
        "wild_type": wild_patch,
        "mutant": mutant_patch,
        "wild_type_sequence": wild_sequence,
        "mutant_sequence": mutant_sequence,
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_R5A.pdb",
        "ddg": 1.25,
    }
    for name in ("1ABC_A_wild_type", "1ABC_A_mutant_R5A"):
        folder = tmp_path / name
        folder.mkdir()
        torch.save(torch.randn(6, 1280), folder / "esm2.pt")
    data_path = tmp_path / "dataset.pkl"
    with data_path.open("wb") as handle:
        import pickle

        pickle.dump([item], handle)

    dataset = CoMDDGPairDataset(
        data_path,
        esm_cache_dir=tmp_path,
        require_esm=True,
    )
    sample = dataset[0]
    batch = collate_graph_pairs([sample])

    assert dataset.pair_length_at(0) == 6
    assert sample.wild.x.shape[0] == 3
    assert sample.wild.sequence_features is None
    assert sample.sequence_wild.sequence_features.shape == (6, 1280)
    assert sample.wild.mutation_index.tolist() == [1]
    assert sample.sequence_wild.mutation_index.tolist() == [4]
    assert batch.wild.x.shape[0] == 3
    assert batch.sequence_wild.x.shape[0] == 6
    assert batch.sequence_wild.sequence_features.shape == (6, 1280)


def test_comddg_item_can_ignore_reference_node_features(tmp_path: Path):
    item = {
        "wild_type": graph_record(),
        "mutant": graph_record(),
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_R2A.pdb",
        "ddg": 1.25,
    }
    for name in ("1ABC_A_wild_type", "1ABC_A_mutant_R2A"):
        folder = tmp_path / name
        folder.mkdir()
        torch.save(torch.randn(4, 1280), folder / "esm2.pt")

    sample = convert_comddg_item(
        item,
        esm_store=ESMFeatureStore(tmp_path),
        require_esm=True,
        ignore_node_features=True,
    )

    assert sample.wild.x.shape == (4, 1)
    assert torch.count_nonzero(sample.wild.x) == 0
    assert sample.wild.sequence_features.shape == (4, 1280)
    assert sample.wild.residue_type is None


def test_comddg_item_uses_reference_pssm_node_features():
    wild_record = graph_record()
    mutant_record = graph_record()
    wild_record["node_features"][:, 20:41] = np.arange(21)
    mutant_record["node_features"][:, 20:41] = np.arange(21) + 100
    item = {
        "wild_type": wild_record,
        "mutant": mutant_record,
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_R2A.pdb",
        "ddg": 1.25,
    }

    sample = convert_comddg_item(
        item,
        sequence_feature_source="pssm",
        sequence_feature_dim=41,
        ignore_node_features=True,
    )

    assert sample.wild.sequence_features.shape == (4, 41)
    assert sample.mutant.sequence_features.shape == (4, 41)
    assert torch.equal(
        sample.wild.sequence_features,
        torch.tensor(wild_record["node_features"][:, :41]),
    )
    assert torch.equal(
        sample.mutant.sequence_features,
        torch.tensor(mutant_record["node_features"][:, :41]),
    )
    assert sample.wild.x.shape == (4, 1)
    assert torch.count_nonzero(sample.wild.x) == 0


def test_comddg_item_combines_esm_and_pssm_features(tmp_path: Path):
    wild_record = graph_record()
    mutant_record = graph_record()
    wild_record["node_features"][:, 20:41] = np.arange(21)
    mutant_record["node_features"][:, 20:41] = np.arange(21) + 100
    item = {
        "wild_type": wild_record,
        "mutant": mutant_record,
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_R2A.pdb",
        "ddg": 1.25,
    }
    for index, name in enumerate(
        ("1ABC_A_wild_type", "1ABC_A_mutant_R2A"), start=1
    ):
        folder = tmp_path / name
        folder.mkdir()
        torch.save(
            torch.full((4, 1280), float(index)),
            folder / "esm2.pt",
        )

    sample = convert_comddg_item(
        item,
        esm_store=ESMFeatureStore(tmp_path),
        require_esm=True,
        sequence_feature_source="esm_pssm",
        sequence_feature_dim=1321,
        ignore_node_features=True,
    )

    assert sample.wild.sequence_features.shape == (4, 1321)
    assert sample.mutant.sequence_features.shape == (4, 1321)
    assert torch.all(sample.wild.sequence_features[:, :1280] == 1)
    assert torch.all(sample.mutant.sequence_features[:, :1280] == 2)
    assert torch.equal(
        sample.wild.sequence_features[:, 1280:],
        torch.tensor(wild_record["node_features"][:, :41]),
    )
    assert torch.equal(
        sample.mutant.sequence_features[:, 1280:],
        torch.tensor(mutant_record["node_features"][:, :41]),
    )


def test_pssm_features_reject_insufficient_node_dimension():
    item = {
        "wild_type": graph_record(),
        "mutant": graph_record(),
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_R2A.pdb",
        "ddg": 1.25,
    }

    with pytest.raises(ValueError, match="requires at least 68"):
        convert_comddg_item(
            item,
            sequence_feature_source="pssm",
            sequence_feature_dim=68,
        )


def test_comddg_item_can_normalize_ddg_sign():
    item = {
        "wild_type": graph_record(),
        "mutant": graph_record(),
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_R2A.pdb",
        "ddg": -1.25,
    }

    sample = convert_comddg_item(item, ddg_multiplier=-1)

    assert sample.ddg.item() == 1.25


def test_comddg_item_pads_esm_for_unknown_graph_residue(tmp_path: Path):
    item = {
        "wild_type": graph_record(length=5),
        "mutant": graph_record(length=5),
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_R2A.pdb",
        "ddg": 1.25,
    }
    for side in ("wild_type", "mutant"):
        item[side]["node_features"][-1, :20] = 0
    for name in ("1ABC_A_wild_type", "1ABC_A_mutant_R2A"):
        folder = tmp_path / name
        folder.mkdir()
        torch.save(torch.randn(4, 1280), folder / "esm2.pt")

    sample = convert_comddg_item(
        item,
        esm_store=ESMFeatureStore(tmp_path),
        require_esm=True,
    )

    assert sample.wild.sequence_features.shape == (5, 1280)
    assert torch.count_nonzero(sample.wild.sequence_features[-1]) == 0


def test_comddg_item_rejects_unexplained_esm_mismatch(tmp_path: Path):
    item = {
        "wild_type": graph_record(length=5),
        "mutant": graph_record(length=5),
        "wild_type_name": "1ABC_A_wild_type.pdb",
        "mutant_name": "1ABC_A_mutant_R2A.pdb",
        "ddg": 1.25,
    }
    for name in ("1ABC_A_wild_type", "1ABC_A_mutant_R2A"):
        folder = tmp_path / name
        folder.mkdir()
        torch.save(torch.randn(4, 1280), folder / "esm2.pt")

    with pytest.raises(ValueError, match="residue count mismatch"):
        convert_comddg_item(
            item,
            esm_store=ESMFeatureStore(tmp_path),
            require_esm=True,
        )


def test_paired_flat_esm_cache_and_node_padding(tmp_path: Path):
    wild_record = graph_record()
    mutant_record = graph_record()
    wild_record["node_features"] = wild_record["node_features"][:, :41]
    mutant_record["node_features"] = mutant_record["node_features"][:, :41]
    item = {
        "wild_type": wild_record,
        "mutant": mutant_record,
        "wild_type_name": "P12345_wild_type",
        "mutant_name": "P12345_mutant_A2G",
        "ddg": -0.75,
    }
    data_path = tmp_path / "dataset.pkl"
    with data_path.open("wb") as handle:
        import pickle

        pickle.dump([item], handle)

    torch.save(
        {"features": np.ones((4, 1280), dtype=np.float32)},
        tmp_path / "P12345_wild_A2G_esm2.pt",
    )
    torch.save(
        {"features": np.full((4, 1280), 2.0, dtype=np.float32)},
        tmp_path / "P12345_mutant_A2G_esm2.pt",
    )

    dataset = CoMDDGPairDataset(
        data_path,
        esm_cache_dir=tmp_path,
        require_esm=True,
        esm_layout="paired_flat",
        node_feature_dim=67,
    )
    sample = dataset[0]

    assert sample.wild.x.shape == (4, 67)
    assert torch.count_nonzero(sample.wild.x[:, 41:]) == 0
    assert sample.wild.sequence_features.shape == (4, 1280)
    assert sample.wild.sequence_features[0, 0].item() == 1.0
    assert sample.mutant.sequence_features[0, 0].item() == 2.0


def test_dataset_can_skip_samples_with_missing_esm(tmp_path: Path):
    complete = {
        "wild_type": graph_record(),
        "mutant": graph_record(),
        "wild_type_name": "P12345_wild_type",
        "mutant_name": "P12345_mutant_A2G",
        "ddg": -0.75,
    }
    missing = {
        "wild_type": graph_record(),
        "mutant": graph_record(),
        "wild_type_name": "P67890_wild_type",
        "mutant_name": "P67890_mutant_A2G",
        "ddg": 0.5,
    }
    data_path = tmp_path / "dataset.pkl"
    with data_path.open("wb") as handle:
        import pickle

        pickle.dump([complete, missing], handle)

    torch.save(
        torch.ones(4, 1280),
        tmp_path / "P12345_wild_A2G_esm2.pt",
    )
    torch.save(
        torch.ones(4, 1280),
        tmp_path / "P12345_mutant_A2G_esm2.pt",
    )

    dataset = CoMDDGPairDataset(
        data_path,
        esm_cache_dir=tmp_path,
        require_esm=True,
        esm_layout="paired_flat",
        skip_missing_esm=True,
    )

    assert len(dataset) == 1
    assert dataset.original_size == 2
    assert dataset.skipped_missing_esm == 1
    assert dataset[0].mutation_name == "P12345_mutant_A2G"


def test_attention_budget_batch_sampler_is_length_aware():
    lengths = [100, 1000, 120, 1100, 130]
    sampler = AttentionBudgetBatchSampler(
        lengths,
        max_batch_size=3,
        max_attention_elements=100_000,
    )

    batches = list(sampler)

    assert batches[:2] == [[0, 2, 4], [1]]
    assert batches[2] == [3]
    for batch in batches:
        max_length = max(lengths[index] for index in batch)
        assert len(batch) == 1 or len(batch) * max_length**2 <= 100_000


def test_auto_resume_prefers_last_checkpoint(tmp_path: Path):
    best_path = tmp_path / "best.pt"
    last_path = tmp_path / "last.pt"
    best_path.touch()

    assert resolve_resume_path("auto", tmp_path) == str(best_path)

    last_path.touch()
    assert resolve_resume_path("auto", tmp_path) == str(last_path)


def test_structure_graph_dataset_reads_sharded_directory(tmp_path: Path):
    import pickle

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    records = []
    for index in range(3):
        record = graph_record(
            length=5,
            node_dim=PYTHIA_NODE_DIM,
            edge_dim=PYTHIA_EDGE_DIM,
        )
        record["name"] = f"domain_{index}"
        records.append(record)
    with (shard_dir / "graphs_0000.pkl").open("wb") as handle:
        pickle.dump(records[:2], handle)
    with (shard_dir / "graphs_0001.pkl").open("wb") as handle:
        pickle.dump(records[2:], handle)

    dataset = StructureGraphDataset(shard_dir, deduplicate_proteins=False)
    graph = dataset[2]

    assert len(dataset) == 3
    assert graph.x.shape == (5, PYTHIA_NODE_DIM)
    assert graph.residue_type.tolist() == [0, 1, 2, 3, 4]


def test_structure_graph_dataset_crops_local_ca_radius(
    tmp_path: Path, monkeypatch
):
    import pickle

    record = graph_record(
        length=6,
        node_dim=PYTHIA_NODE_DIM,
        edge_dim=PYTHIA_EDGE_DIM,
    )
    record["name"] = "local_crop_case"
    record["ca_coords"] = np.asarray(
        [[float(index * 4), 0.0, 0.0] for index in range(6)],
        dtype=np.float32,
    )
    src = np.arange(5, dtype=np.int64)
    dst = src + 1
    record["edge_index"] = np.stack(
        [np.concatenate([src, dst]), np.concatenate([dst, src])],
        axis=0,
    )
    record["edge_features"] = np.ones((10, PYTHIA_EDGE_DIM), dtype=np.float32)
    path = tmp_path / "graphs.pkl"
    with path.open("wb") as handle:
        pickle.dump([record], handle)

    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.tensor([2], dtype=torch.long),
    )
    dataset = StructureGraphDataset(
        path,
        deduplicate_proteins=False,
        local_crop_radius=4.1,
    )

    graph = dataset[0]

    assert graph.residue_type.tolist() == [1, 2, 3]
    assert graph.mutation_index.tolist() == [1]
    assert graph.edge_index.max().item() < graph.x.shape[0]
    assert graph.edge_attr.shape[0] == 4


def test_structure_graph_dataset_crops_local_ca_neighbors(
    tmp_path: Path, monkeypatch
):
    import pickle

    record = graph_record(
        length=6,
        node_dim=PYTHIA_NODE_DIM,
        edge_dim=PYTHIA_EDGE_DIM,
    )
    record["name"] = "local_knn_case"
    record["ca_coords"] = np.asarray(
        [[float(index * 3), 0.0, 0.0] for index in range(6)],
        dtype=np.float32,
    )
    atom5 = np.repeat(record["ca_coords"][:, None, :], 5, axis=1)
    chain = np.zeros(6, dtype=np.int64)
    resseq = np.arange(6, dtype=np.int64)
    record["edge_index"], record["edge_features"] = pythia_knn_graph(
        atom5,
        chain,
        resseq,
        k_neighbors=2,
    )
    path = tmp_path / "graphs.pkl"
    with path.open("wb") as handle:
        pickle.dump([record], handle)

    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.tensor([2], dtype=torch.long),
    )
    dataset = StructureGraphDataset(
        path,
        deduplicate_proteins=False,
        local_crop_neighbors=2,
    )

    graph = dataset[0]

    assert graph.x.shape == (3, PYTHIA_NODE_DIM)
    assert graph.residue_type.tolist() == [1, 2, 3]
    assert graph.mutation_index.tolist() == [1]
    assert graph.edge_attr.shape[1] == PYTHIA_EDGE_DIM


def test_pythia_graph_feature_builders_match_pythia_dimensions():
    aa = np.asarray([0, 5, 19], dtype=np.int64)
    angles = np.asarray(
        [[0.0, np.pi / 2.0, np.pi], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=np.float32,
    )
    node_features = pythia_node_features(aa, angles)

    assert node_features.shape == (3, PYTHIA_NODE_DIM)
    assert node_features[0, 0] == 1.0
    assert node_features[1, 5] == 1.0
    assert node_features[2, 19] == 1.0
    assert np.all(node_features[:, 20:22] == 0)

    atom5 = np.zeros((4, 5, 3), dtype=np.float32)
    atom5[:, :, 0] = np.arange(4, dtype=np.float32)[:, None] * 3.0
    atom5[:, :, 1] = np.arange(5, dtype=np.float32)[None, :]
    chain = np.asarray([0, 0, 1, 1], dtype=np.int64)
    resseq = np.asarray([1, 2, 1, 2], dtype=np.int64)
    edge_index, edge_attr = pythia_knn_graph(atom5, chain, resseq, k_neighbors=2)

    assert edge_index.shape == (2, 8)
    assert edge_attr.shape == (8, PYTHIA_EDGE_DIM)
    target_zero = edge_index[1] == 0
    assert edge_index[0, target_zero].tolist() == [1, 2]
    assert edge_attr[target_zero, -1].tolist() == [1.0, 0.0]


def test_pythia_coordinate_cache_conversion_preserves_ddg_graph_contract():
    length = 35
    record = graph_record(length=length, node_dim=67, edge_dim=16)
    record["mutation_pos"] = 7
    pos14 = np.zeros((length, 14, 3), dtype=np.float32)
    for index in range(length):
        base = float(index) * 3.8
        pos14[index, 0] = [base - 1.2, 0.1, 0.0]
        pos14[index, 1] = [base, 0.0, 0.0]
        pos14[index, 2] = [base + 1.3, 0.1, 0.0]
        pos14[index, 3] = [base + 1.8, 1.0, 0.0]
    pos14_mask = np.zeros((length, 14, 3), dtype=bool)
    pos14_mask[:, :4, :] = True
    coord = {
        "pos14": pos14,
        "pos14_mask": pos14_mask,
        "resseq": np.arange(1, length + 1, dtype=np.int64),
    }

    converted = convert_pythia_coordinate_record(record, coord, k_neighbors=4)

    assert converted["node_features"].shape == (length, PYTHIA_NODE_DIM)
    assert converted["edge_features"].shape == (length * 4, PYTHIA_EDGE_DIM)
    assert converted["edge_index"].shape == (2, length * 4)
    assert converted["mutation_pos"] == 7
    assert converted["residue_type"].tolist()[:5] == [0, 1, 2, 3, 4]
    assert converted["ca_coords"].shape == (length, 3)
