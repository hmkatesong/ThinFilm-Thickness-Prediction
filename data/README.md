# Data

No dataset is included yet.

## Directory convention

- `raw/`: original, unmodified source files.
- `processed/`: cleaned or transformed files generated from raw data.

## TODO before using a dataset

1. Record the source, download or acquisition date, version, and usage terms.
2. Describe each input feature, its unit, and how it was measured or obtained.
3. Identify the target thickness field, its unit, and its measurement method.
4. Record sample count, missing values, duplicates, and any exclusions.
5. Document preprocessing and train/validation/test split rules.

Keep source data unchanged in `raw/`. Do not commit confidential or restricted data. Data files are ignored by Git by default; only the directory placeholders are tracked.
