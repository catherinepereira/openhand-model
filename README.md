# openhand-model

Training code for the three models that power [openhand](../openhand). Each
model lives in its own subdirectory with its own `data/`, `model/`,
`scripts/`, and `exports/`:

| Subdir | Model | Powers |
|--------|-------|--------|
| [alphabet/](alphabet/) | Per-frame A-Z MLP (~62K params) | Per-letter detection |
| [fingerspelling/](fingerspelling/) | CTC transformer (~5.5M params) | Streaming phrase transcription + J/Z Learn |
| [signs/](signs/) | Conv1D + Transformer over 250 ISLR classes | Learn-the-words view |

`shared/` holds cross-cutting utilities: the MediaPipe `.task` download,
the Python<->TS landmark parity fixture, and the bundled MediaPipe hand
model.

If you just want to run OpenHand, you don't need this repo at all once
the artifacts are built. This is the training side of the project.

## Setup

```powershell
git clone https://github.com/catherinepereira/openhand-model
cd openhand-model

python -m venv venv
venv\Scripts\Activate.ps1     # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# Fetch the MediaPipe hand landmarker (~8 MB) into shared/
python shared/download_mediapipe_model.py
```

One venv covers all three models. CUDA is detected automatically.

## Running each model

Each subdir has a `download_*_data.py` and a `run_pipeline.py`. After the
shared setup above, the common case is two commands per model:

```powershell
# Alphabet: ~1 GB, ~10 min training on a 4070-class GPU
python alphabet/scripts/download_alphabet_data.py
python alphabet/scripts/run_pipeline.py

# CTC fingerspelling: ~160 GB, ~hours of training
python fingerspelling/scripts/download_data.py
python fingerspelling/scripts/run_pipeline.py
# (--smoke for a 2-epoch sanity check)

# Isolated signs: ~5 GB, ~30 min training
python signs/scripts/download_signs_data.py
python signs/scripts/run_pipeline.py
```

Each pipeline does: preprocess -> train -> (eval) -> export ONNX -> build
references (where applicable). Step scripts are still runnable on their
own. See the per-model READMEs:

- [alphabet/README.md](alphabet/README.md)
- [fingerspelling/README.md](fingerspelling/README.md)
- [signs/README.md](signs/README.md)

## Tests / parity

The CTC pipeline shares its landmark normalization with the openhand
frontend. To regenerate the fixture that catches drift between the two:

```powershell
python shared/dump_landmark_vectors.py
# Writes ../openhand/frontend/src/lib/__tests__/landmark_fixtures.json
```

## License

MIT. See [LICENSE](LICENSE).
