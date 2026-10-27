# Reproducibility and release boundaries

This is a source-only release. Installation, a forward/backward smoke test,
unit tests and command-line help checks can be performed without experimental
datasets. They do not reproduce the quantitative manuscript results.

Exact experiment reproduction additionally needs:

1. The actual original WT/MUT coordinates and their preparation provenance.
2. Mutation identities, original ordering and train/validation/test manifests.
3. Graph-generation settings and exact ESM feature caches/model identity.
4. The real pretrained encoder and downstream checkpoint, with checksums.
5. The configuration stored in each checkpoint and the training logs.
6. Per-mutation prediction exports used to calculate the reported statistics.

Dataset names are collection names, not a promise about post-filtering row
counts. Report actual accepted/missing records from the data adapter and keep
stable mutation IDs. Never silently substitute a partial intersection for a
full-set result.

The four-objective manuscript recipe and the additional-objective legacy run
are distinguished explicitly in `CONFIGURATION.md`. The source model and loss
implementation are preserved rather than rewritten to hide their differences.

No research data, generated baseline proxies, manuscript-editing utilities,
synthetic scientific illustration scripts, private account information or model
weights are uploaded by this distribution. Synthetic tensors in unit tests are
solely software fixtures. The analysis utility consumes a supplied trained
checkpoint; it does not insert the manuscript's desired metric values.

`release_manifest.json` records original hashes of copied source files and the
few portability edits. Those edits concern filesystem paths, optional analysis
defaults and explicit trusted-checkpoint loading, not model architecture,
training objectives or the core optimizer implementation.

The owner requested no license selection at this stage. There is no invented
copyright holder, permissive license or paper DOI. Finalize the license, data
access terms, citation and trained-asset release before claiming an openly
licensed, fully reproducible publication release.
