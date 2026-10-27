# Data format and preparation

## Inputs

The downstream adapter consumes trusted CoMDDG-compatible pickles containing
paired WT/MUT records, mutation names/IDs and a numeric `ddg` field. Positive
values mean destabilization in the public recipe. It supports distinct
full-length sequence records and mutation-centered structural patches.

Inspect `mutflow_ddg/data/comddg.py` for the legacy dictionary keys and
`mutflow_ddg/data/schema.py` for the model-facing `GraphBatch`, `MutationSample`
and `MutationBatch` contracts. Do not relabel other feature dimensions as the
28-D/27-D representation without regenerating their features.

The structure recipe expects 28-D node features and 27-D edge features. Local
patches contain the mutation center and up to 32 nearest neighbors. Full-length
ESM2 representations are 1280-dimensional per-residue features, aligned to the
actual WT/MUT chain sequences; a matching array length alone does not establish
the correct chain or mutation identity.

Standard ESM cache layout:

```text
data/ddg/ssym/esm/
  <wild_type_record_name>/esm2.pt
  <mutant_record_name>/esm2.pt
```

The paired-flat legacy layout is optional and must be requested explicitly.

## Structural pretraining data

The downloader and graph converter are provided, but their execution downloads
external structures and must be initiated by the repository user:

```bash
python -m scripts.download_rcsb_bioa --out data/pretrain/raw/rcsb_bioa
python -m scripts.convert_cath_pdb_to_graphs --help
python -m scripts.split_structure_graph_shards --help
```

Prepare trusted graph shards and their manifest files at the configured paths:

```text
data/pretrain/rcsb_bioa/train/rcsb_bioa_train_graphs_shards.txt
data/pretrain/rcsb_bioa/validation/rcsb_bioa_validation_graphs_shards.txt
```

Each manifest lists the corresponding shard paths, one per line. Keep data
filtering, duplicate removal and train/test manifests from the actual experiment;
rerunning a live structure search does not guarantee the same historical set.

## Downstream conversion and ESM features

The coordinate-cache converter supports explicit chain/mutation validation:

```bash
python -m scripts.generate_pythia_structure_from_coordinate_cache --help
python -m scripts.generate_pythia_patch_ddg --help
python -m scripts.attach_mutation_ids --help
```

For full-length records with the expected amino-acid one-hot prefix, the
Transformers-based ESM extraction entry point is:

```bash
python -m scripts.extract_esm \
  --input-pkl data/input/full_length_pairs.pkl \
  --output-dir data/output/esm \
  --model-name facebook/esm2_t33_650M_UR50D
```

Do not extract full-length sequence features from an already cropped structural
patch. Legacy one-hot amino-acid orders differ between preparation sources:
verify the declared order against the true sequence before extraction. The
`fill_missing_esm` script preserves its legacy input contract and supports an
explicit local fair-esm checkpoint via `--model-path`.

FoldX and upstream datasets are not bundled or automatically licensed by this
release. Supply their prepared coordinate models and record the version,
chain selection, mutation mapping and preprocessing settings separately.
