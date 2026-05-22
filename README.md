# openhand-model

Training code for the three models that power [openhand](../openhand). Each
model lives in its own subdirectory with its own `data/`, `model/`,
`scripts/`, and `exports/`:

| Subdir | Model | Powers |
|--------|-------|--------|
| [alphabet/](alphabet/) | Per-frame A-Z MLP (~62K params) | Per-letter detection |
| [fingerspelling/](fingerspelling/) | Squeezeformer + CTC (~3.3M params) | Streaming phrase transcription + J/Z Learn |
| [signs/](signs/) | Conv1D + Transformer over 250 ISLR classes | Learn-the-words view |

If you only want to run OpenHand, you don't need this repo. This is the
training side of the project; the runtime artifacts get copied into
`openhand/backend/models/artifacts/`.

## Setup

```powershell
git clone https://github.com/catherinepereira/openhand-model
cd openhand-model

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Fetch the MediaPipe hand landmarker (~8 MB) into shared/
python shared/download_mediapipe_model.py
```

## Running each model

Each subdir has a `download_*_data.py` and a `run_pipeline.py`. Two
commands per model after the shared setup above:

```powershell
# Alphabet
python alphabet/scripts/download_alphabet_data.py
python alphabet/scripts/run_pipeline.py

# Fingerspelling
python fingerspelling/scripts/download_fingerspelling_data.py
python fingerspelling/scripts/run_pipeline.py

# Signs
python signs/scripts/download_signs_data.py
python signs/scripts/run_pipeline.py
```

Each pipeline chains preprocess -> train -> eval -> export ONNX, plus a reference-build step for the alphabet and signs models. 
See the per-model READMEs:

- [alphabet/README.md](alphabet/README.md)
- [fingerspelling/README.md](fingerspelling/README.md)
- [signs/README.md](signs/README.md)


## Tests / parity

The CTC pipeline shares its landmark normalization with the openhand frontend. 
To regenerate the fixture that catches drift between the two:

```powershell
python shared/dump_landmark_vectors.py
# Writes ../openhand/frontend/src/lib/__tests__/landmark_fixtures.json
```
