from __future__ import annotations

import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from .schema import MutationSample, reverse_mutation_features


def protein_level_split(
    dataset: Dataset,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    protein_to_indices: dict[str, list[int]] = {}
    for index in range(len(dataset)):
        if hasattr(dataset, "protein_id_at"):
            protein_id = dataset.protein_id_at(index)
        else:
            sample: MutationSample = dataset[index]
            protein_id = sample.protein_id
        protein_to_indices.setdefault(protein_id, []).append(index)
    proteins = list(protein_to_indices)
    random.Random(seed).shuffle(proteins)
    validation_count = max(1, round(len(proteins) * validation_fraction))
    validation_proteins = set(proteins[:validation_count])
    train_indices, validation_indices = [], []
    for protein, indices in protein_to_indices.items():
        target = (
            validation_indices if protein in validation_proteins else train_indices
        )
        target.extend(indices)
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


class ReverseAugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset) * 2

    def __getitem__(self, index: int) -> MutationSample:
        sample = self.dataset[index % len(self.dataset)]
        if index < len(self.dataset):
            return sample
        return MutationSample(
            wild=sample.mutant,
            mutant=sample.wild,
            ddg=-sample.ddg,
            sequence_wild=sample.sequence_mutant,
            sequence_mutant=sample.sequence_wild,
            mutation_features=(
                reverse_mutation_features(sample.mutation_features)
            ),
            protein_id=sample.protein_id,
            mutation_name=f"reverse:{sample.mutation_name}",
            mutation_id=sample.mutation_id,
            num_mutations=sample.num_mutations,
            epistasis_target=(
                -sample.epistasis_target
                if sample.epistasis_target is not None
                else None
            ),
        )

    def pair_length_at(self, index: int) -> int:
        base_index = index % len(self.dataset)
        if isinstance(self.dataset, Subset):
            source_index = self.dataset.indices[base_index]
            if hasattr(self.dataset.dataset, "pair_length_at"):
                return self.dataset.dataset.pair_length_at(source_index)
        if hasattr(self.dataset, "pair_length_at"):
            return self.dataset.pair_length_at(base_index)
        sample = self.dataset[base_index]
        return max(
            sample.wild_for_sequence.x.shape[0],
            sample.mutant_for_sequence.x.shape[0],
        )


def reverse_augmented_protein_split(
    dataset: Dataset,
    validation_fraction: float,
    seed: int,
) -> tuple[ReverseAugmentedDataset, ReverseAugmentedDataset]:
    """Split proteins first, then add reverse mutations within each subset."""

    train_dataset, validation_dataset = protein_level_split(
        dataset,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    return (
        ReverseAugmentedDataset(train_dataset),
        ReverseAugmentedDataset(validation_dataset),
    )


def reverse_augmented_random_split(
    dataset: Dataset,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Match CoMDDG's split: append reverse mutations, then split samples."""

    if len(dataset) < 1:
        raise ValueError("Cannot split an empty mutation dataset")
    augmented = ReverseAugmentedDataset(dataset)
    indices = list(range(len(augmented)))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
    )
    return (
        Subset(augmented, train_indices),
        Subset(augmented, validation_indices),
    )
