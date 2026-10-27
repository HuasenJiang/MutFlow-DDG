# Configuration guide

## Portable manuscript recipes

`configs/pretrain.yaml` selects noisy-to-clean conditional flow pretraining:

- Central-residue recovery, weight 1.0.
- Conditional flow matching, weight 1.0.
- Endpoint transport, weight 0.1.
- Contrastive alignment, weight 0.2.
- Replacement, geometry and contact auxiliary terms explicitly set to zero.
- Mask probability 0.85; random replacement probability 0.15.
- Hidden/latent dimension 128; three structure layers; eight Euler steps.

`configs/train.yaml` uses the existing paired-state multimodal architecture,
ESM input dimension 1280, 28-D structural nodes and 27-D edges. It transfers
the structure encoder only, not the pretrained flow. Its objective weights are
regression 1.0, classification 0.2, ranking 0.1, flow 0.5, transport 0.1 and
uncertainty 0.1. Explicit output antisymmetrization is disabled.

Dataset membership is S2648 development plus S250, S350, S605, S1925 and SSYM
evaluation. The optional p53 case collection is a separate prediction input.
Edit the path fields to point to your legally obtained/prepared data.

## Provenance versus a proposed recipe

`configs/legacy/` retains the historical run settings, with private mount paths
converted to repository-relative paths. Numeric settings and dataset membership
are otherwise retained. These are provenance records, not the recommended
public quick-start configuration.

In particular, the legacy downstream configuration references a checkpoint from
`structure_mutation_flow_protocol`. That pretraining configuration uses the
`mutation_flow` mode and includes replacement, geometry and contact auxiliary
objectives. It is not identical to the manuscript's four-objective noisy-to-clean
recipe. The older `structure_flow_pretrain_rcsb_bioa.yaml` also leaves auxiliary
weights to nonzero dataclass defaults. The portable recipe disables them
explicitly. No existing experiment was relabeled or rerun by packaging the code.

Do not replace the configuration recorded in an archived checkpoint and then
claim the new recipe generated that checkpoint or the same reported numbers.
Inspect the actual checkpoint metadata and run logs when finalizing provenance.

## Splitting and evaluation

The existing sample-level setting augments reverse samples and then splits at
seed 42. `data.split_level: protein` provides an alternative grouped protocol;
changing this setting changes the experiment and requires retraining.

The portable configuration sets `report_directions: true` only for SSYM.
S669 remains in the legacy record but is not one of the five portable benchmark
inputs. Neither this packaging step nor a configuration name establishes test
set independence, deduplication or homology control.
