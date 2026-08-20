# Experiment Scripts (`scripts/`)

This directory contains the executable entry points for running the various experiments in the Mind Attribution project. 

While the `data/` directory handles stimulus creation and the `lib/` directory contains the core analysis harness and math, `scripts/` is the glue code that ties them together to run actual end-to-end tests.

## Common Scripts

- `run_e1.py` through `run_e4_experiment.py`: Run the main experiments (extracting directions, probing, behavioral tests, etc.).
- `run_cache.py`: Helper for capturing and caching activations.
- `run_specificity.py`: Evaluates the specificity of extracted directions.
- ...and other miscellaneous check scripts.

*Note: All core math and logic should remain in `lib/`. The scripts here are intended purely as runners and configuration glue.*
