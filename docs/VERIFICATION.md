# Source release verification

Checks performed on 2026-10-20:

- 69 unit/integration tests passed, including single-checkpoint CSV inference
  with paired forward/reverse inputs.
- The sequence-only, structure-only and multimodal forward/backward smoke tests
  passed.
- Training, pretraining and prediction CLI help entry points loaded.
- A wheel was built without dependency installation, installed into an isolated
  temporary directory, and successfully imported and smoke-tested outside the
  source tree. Build dependencies were isolated from the existing environment.
- Missing pretraining assets were correctly detected by the input-location
  checker; their absence is expected in a source-only distribution.
- All copied `mutflow_ddg/` implementation files match their original SHA-256
  hashes. Source changes are confined to explicitly recorded script portability
  edits, packaging metadata, documentation and added release utilities/tests.
- No manuscript, original dataset, pretrained asset, ESM cache, user-account
  information or recognized credential pattern is included in the source package.

The runtime check used Python 3.11.11, PyTorch 2.9.1+cu128 (CPU test execution),
NumPy 1.26.4, PyYAML 6.0.1, SciPy 1.16.3, scikit-learn 1.2.2, pytest 7.4.0
and Biopython 1.88. Biopython was provided in an isolated temporary directory.
The available scikit-learn runtime is older than the declared dependency floor;
the original minimum dependency remains unchanged in the package metadata.

Tests exercise software contracts, not trained predictive accuracy. Full training,
ESM model downloading, FoldX structure generation, cross-device numerical
equivalence and reproduction of the paper's result tables were not run. A fresh
installation of all runtime dependencies was not performed in this existing
environment. Follow `README.md` to install the declared requirements in your own
environment before an actual experiment.
