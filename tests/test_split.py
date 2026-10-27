from sklearn.model_selection import train_test_split
import torch

from mutflow_ddg.data.schema import GraphBatch, MutationSample
from mutflow_ddg.data.split import (
    ReverseAugmentedDataset,
    reverse_augmented_protein_split,
    reverse_augmented_random_split,
)


def graph(value: float) -> GraphBatch:
    return GraphBatch(
        x=torch.tensor([[value]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
        batch=torch.zeros(1, dtype=torch.long),
        mutation_index=torch.zeros(1, dtype=torch.long),
    )


def sample(index: int) -> MutationSample:
    return MutationSample(
        wild=graph(float(index)),
        mutant=graph(float(index + 100)),
        ddg=torch.tensor(float(index + 1)),
        sequence_wild=graph(float(index + 200)),
        sequence_mutant=graph(float(index + 300)),
        protein_id=f"protein-{index // 2}",
        mutation_name=f"mutation-{index}",
    )


def test_reverse_mutations_are_added_before_sklearn_split():
    dataset = [sample(index) for index in range(10)]
    train_dataset, validation_dataset = reverse_augmented_random_split(
        dataset,
        validation_fraction=0.1,
        seed=42,
    )
    expected_train, expected_validation = train_test_split(
        list(range(20)),
        test_size=0.1,
        random_state=42,
    )

    assert train_dataset.indices == expected_train
    assert validation_dataset.indices == expected_validation
    assert expected_validation == [0, 17]

    direct = validation_dataset[0]
    reverse = validation_dataset[1]
    assert direct.mutation_name == "mutation-0"
    assert reverse.mutation_name == "reverse:mutation-7"
    assert reverse.ddg.item() == -8.0
    assert reverse.wild is dataset[7].mutant
    assert reverse.mutant is dataset[7].wild
    assert reverse.sequence_wild is dataset[7].sequence_mutant
    assert reverse.sequence_mutant is dataset[7].sequence_wild


def test_protein_split_happens_before_reverse_augmentation():
    dataset = [sample(index) for index in range(10)]
    train_dataset, validation_dataset = reverse_augmented_protein_split(
        dataset,
        validation_fraction=0.4,
        seed=42,
    )

    assert isinstance(train_dataset, ReverseAugmentedDataset)
    assert isinstance(validation_dataset, ReverseAugmentedDataset)
    assert len(train_dataset) == 12
    assert len(validation_dataset) == 8

    train_proteins = {
        train_dataset[index].protein_id for index in range(len(train_dataset))
    }
    validation_proteins = {
        validation_dataset[index].protein_id
        for index in range(len(validation_dataset))
    }
    assert train_proteins.isdisjoint(validation_proteins)

    for augmented in (train_dataset, validation_dataset):
        half = len(augmented) // 2
        for index in range(half):
            direct = augmented[index]
            reverse = augmented[index + half]
            assert reverse.protein_id == direct.protein_id
            assert reverse.mutation_name == f"reverse:{direct.mutation_name}"
            assert reverse.ddg.item() == -direct.ddg.item()
