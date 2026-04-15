# Plant Disease Classification with Hybrid Median Mean Preprocessing Technique

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run
```bash
python -m experiments.run_experiments
```

## Dataset
Place dataset in:
data/raw/

## Experiments
- Baseline
- CLAHE
- HSV
- Hybrid Median Mean