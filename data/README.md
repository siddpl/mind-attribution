# Data Pipeline

The `data/` directory builds, validates, and manages the affirm/deny stimulus sentences used for probing.

See [`schema.json`](schema.json) for the complete contrast pairs data structure, which requires fields like `claim_id`, `entity`, `template_id`, `affirm_text`, and `deny_text`.

> **Note**: The old schema and its associated `validate.py` and `models.py`
> scripts have been moved to the [`deprecated/`](deprecated/) directory.

Validation and data generation tools are located in the `scripts/` directory.

## 1. Environment Setup

To run the data scripts (not to use the data, the scripts to validate/generate data), create and activate a Python 3 virtual environment in the `data/` directory:

```bash
cd data
python3 -m venv datavenv
source datavenv/bin/activate
pip install -r scripts/requirements.txt
cd ..
```

## 2. Validation & Generation

- **Validate**: Run `python3 data/scripts/data_net.py` to run global "NET" validations ensuring data integrity across contrast pairs.
- **Generate**: Use the templates in `data/scripts/templates.py` to generate the exact affirm/deny cross-product contrasts for extraction.

## Sub-Documentation
- [`elicitation/README.md`](elicitation/README.md): Details on elicitation.
- [`referent_ladder/experiment_rationale_and_simplified_schema.md`](referent_ladder/experiment_rationale_and_simplified_schema.md): Rationale and schema for referent ladder.
