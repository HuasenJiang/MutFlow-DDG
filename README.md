# MutFlow-DDG

Research implementation of a pretraining-enhanced, paired-state multimodal
framework for predicting mutation-induced changes in monomeric protein stability.

The model combines:

- Shared WT/MUT sequence encoders using full-length ESM features.
- Shared structure encoders operating on mutation-centered residue graphs.
- State-level cross-modal attention for multimodal fusion.
- A mutation-conditioned latent vector field and explicit Euler transport.
- Regression, stability classification, ranking and uncertainty heads.

Positive ΔΔG denotes destabilization. Latent trajectories are representation-space
transitions, not physical molecular dynamics or free-energy trajectories.

## Release scope

This repository contains model source, preprocessing and training entry points,
single-checkpoint prediction, analysis utilities and data-independent tests.
It does **not** bundle benchmark datasets, ESM caches, FoldX, trained weights,
manuscripts or research-only figure mockups.

`configs/pretrain.yaml` explicitly selects the four pretraining objectives in
the manuscript. `configs/train.yaml` is its portable downstream recipe. Original
experiment settings are retained in `configs/legacy/` for provenance. These
recipes must not be mistaken for a newly completed reproduction of the paper's
tables. See [reproducibility notes](docs/REPRODUCIBILITY.md).

## Installation

Python 3.10 or newer is required; Python 3.11 is a practical starting point.
Install a PyTorch build compatible with your CPU/CUDA environment, then run
the following commands from this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,esm,analysis]'
```

For model training with existing ESM caches, the `esm` extra is not required.
For just the model and tests, use `python -m pip install -e '.[dev]'`.
Run all `python -m scripts...` commands from the repository root.

## Quick verification without datasets or checkpoints

```bash
python -m scripts.smoke_test
python -m pytest -q
```

The smoke test checks forward/backward computation for sequence-only,
structure-only and multimodal branches. Test inputs are synthetic fixtures,
not benchmark predictions or evidence for manuscript performance.

## Data preparation

See [data format and preparation](docs/DATA.md). The runtime needs paired WT/MUT
graph records, correctly aligned full-length ESM caches, and local structural
patches. Mutant coordinates must be supplied by an appropriate preparation
pipeline; this repository does not automatically call or redistribute FoldX.

Inspect the expected inputs before launching a long run:

```bash
python -m scripts.check_data --config configs/pretrain.yaml --stage pretrain
python -m scripts.check_data --config configs/train.yaml --stage train
```

These checks report missing paths and exit nonzero if required assets are absent.
They do not download datasets, validate dataset independence, or deserialize
every structure/cache entry.

## Structural pretraining

Prepare the graph-shard manifests described in `docs/DATA.md`, then run:

```bash
python -m scripts.pretrain_structure --config configs/pretrain.yaml
```

The portable recipe uses residue recovery, conditional flow matching, endpoint
transport consistency and contrastive alignment. Additional legacy auxiliary
heads remain implemented but have zero weights in this recipe.

## Downstream training and evaluation

```bash
python -m scripts.train --config configs/train.yaml
python -m scripts.train --config configs/train.yaml --resume
python -m scripts.train --config configs/train.yaml --evaluate-only
```

The downstream recipe keeps the legacy sample-level 90:10 split after reverse
augmentation. This is not a protein-grouped or homology-grouped protocol.
Protein grouping is an optional setting, not a completed validation claim.

The trainer retains Top-K validation-selected checkpoints and writes prediction
and metric CSV files under `outputs/checkpoints/mutflow_ddg/`. Evaluation-only
mode uses saved checkpoints, regenerates evaluation exports and may update
checkpoint metadata; keep an archival copy when needed. See
[configuration notes](docs/CONFIGURATION.md).

## Predict with one trained checkpoint

```bash
python -m scripts.predict \
  --config configs/train.yaml \
  --checkpoint outputs/checkpoints/mutflow_ddg/best.pt \
  --input-pkl data/ddg/ssym/samples/ssym_pythia_patch32_paired.pkl \
  --esm-cache-dir data/ddg/ssym/esm \
  --dataset-name ssym \
  --reverse \
  --output outputs/predictions/ssym.csv
```

Rows include mutation ID, direction, ΔΔG prediction, predicted uncertainty,
classification probabilities and ranking score. Use mutation IDs to align
exports: length-aware batching may change row order. With `--reverse`, each
input is also evaluated with WT/MUT exchanged. Already paired input sets may
therefore contain duplicate biological directions; deduplicate by mutation ID
before a benchmark comparison. This utility exports model predictions without
automatically aggregating direct/reverse pairs.

For samples without experimental labels, the current paired-pickle schema still
requires a numeric `ddg` placeholder. Pass `--unlabeled` so this placeholder is
not exported as an experimental value. Prediction does not use the label.

Only load trusted pickle files and PyTorch checkpoints: legacy serialization
may execute arbitrary code. Release artifacts should be verified by checksum.

## Interpretation from trained model outputs

```bash
python -m scripts.extract_plot_interpretability \
  --config configs/train.yaml \
  --checkpoint outputs/checkpoints/mutflow_ddg/best.pt \
  --dataset ssym \
  --out-dir outputs/interpretability
```

This utility exports checkpoint-derived modality/ablation, latent transport,
trajectory and uncertainty quantities. Its plot layout is an analysis example,
not a claim that it regenerates every panel of the final manuscript unchanged.

## Layout

```text
mutflow_ddg/        runtime model, graph adapters, objectives and trainer
scripts/           preparation, pretraining, training, prediction and analysis
configs/           portable recipes plus separately labeled legacy settings
tests/             data-independent unit tests
docs/              data contract, configuration and reproducibility notes
data/              instructions only; actual datasets are not bundled
checkpoints/       instructions only; actual checkpoints are not bundled
release_manifest.json  source checksums and portability-change record
```

## License and citation

Project-wide license selection is pending; no `LICENSE` file is included.
Third-party code, datasets, structures and model weights retain their own terms.
The authors will supply the final paper citation and chosen software license.
