# Mind Attribution

A project for probing LLMs on mind attribution claims across different entities, templates, perspectives, and polarities.

## Data

The `data/` directory contains LLM probing prompts. See [`data/schema.json`](data/schema.json) for the complete contrast pairs data structure, which requires fields like `claim_id`, `entity`, `template_id`, `affirm_text`, and `deny_text`.

> **Note**: The old schema and its associated `validate.py` and `models.py` scripts have been moved to the [`data/depricated/`](data/depricated/) directory.

Validation and data generation tools are located in the `data/scripts/` directory.

### 1. Environment Setup

To run the data scripts (not to use the data, the scripts to validate/generate data), create and activate a Python 3 virtual environment in the `data/` directory:

```bash
cd data
python3 -m venv datavenv
source datavenv/bin/activate
pip install -r scripts/requirements.txt
cd ..
```

### 2. Validation & Generation

- **Validate**: Run `python3 data/scripts/data_net.py` to run global "NET" validations ensuring data integrity across contrast pairs.
- **Generate**: Use the templates in `data/scripts/templates.py` to generate the exact affirm/deny cross-product contrasts for extraction.
