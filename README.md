# Mind Attribution

A project for probing LLMs on mind attribution claims across different entities, templates, perspectives, and polarities.

## Data

The `data/` directory contains LLM probing prompts. See [`data/schema.json`](data/schema.json) for the complete data structure, which requires fields like `claim_id` (e.g. `audience_frames_default_user_c13`), `entity`, `template_id`, `person`, and `polarity`.

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
- **Validate**: Run `python3 data/scripts/validate.py` to check all `.json` files in the data directory (ensuring schema compliance, unique IDs, exactly 50/50 polarity balance, overlapping length distributions, capped denial words, and no text duplicates). You can also pass a specific file like `python3 data/scripts/validate.py data/audience_frames/default_user.json`.
- **Generate**: Use the Pydantic models in `data/scripts/models.py` (e.g., `PromptDataPoint`) to easily create schema-compliant dictionaries in Python with IDE autocompletion before exporting to JSON.
