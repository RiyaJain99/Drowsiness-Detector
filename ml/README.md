# DrowsyGuard ML

Offline pipeline that trains the classifier behind the dashboard's **ML prediction** card. Nothing here runs during live monitoring: training happens in Python, and the browser evaluates the exported model locally, so the app still has no server and uploads no video.

```
video ──► prepare ──► frames/*.parquet ──► build_features ──► features.parquet ──► train
          MediaPipe    per-frame ratios     60 s windows                            │
                                                                                     ▼
 webcam ─► MediaPipe ─► mlFeatures.ts ─► mlModel.ts ◄── public/ml/model.json + metrics.json
                        same features    tree walker
```

## Setup

Needs Python 3.12 or newer (xgboost 3.4 requires it), and `npm install && npm run setup` from the repo root, because extraction uses the same `public/mediapipe/face_landmarker.task` as the app.

```bash
cd ml
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Run every command below from `ml/`, using `.venv/bin/python`.

## Datasets

Respect each dataset's terms. Datasets, extracted data and the virtualenv are git-ignored. Sampled frames are off by default because 24 of RLDD's 60 participants did not agree to publication of their faces.

| Dataset | Role | Labels | How to get it |
|---|---|---|---|
| **UTA-RLDD** | Primary | Participant-reported, per video: 0 alert, 5 low vigilant, 10 drowsy | Public Google Drive folder linked from the [dataset page](https://sites.google.com/view/utarldd/home). 120 GB in ten zips. Cite Ghoddoosian, Galib and Athitsos, CVPRW 2019 |
| **NTHU-DDD** | Optional | Per frame, 0/1. Drowsiness is acted | Request from the [NTHU CV Lab](https://cv.cs.nthu.edu.tw/) |
| **YawDD** | Optional | Behaviour per clip. Treating yawning as drowsy is a weak label | Request from the authors (Abtahi et al., MMSys 2014) |

Only RLDD has been run end to end. The NTHU-DDD and YawDD adapters are tested against the documented layouts, not against the real files, so check their output on first use.

## 1. Extract

Stream RLDD with no need for 120 GB of disk. Videos are pulled out of the remote zips one at a time, CRC-checked and deleted once measured:

```bash
python -m drowsyguard_ml.prepare --dataset rldd --stream --subjects-per-fold 8
python -m drowsyguard_ml.prepare --dataset rldd --stream     # all 60 subjects
```

Or use folders you already have:

```bash
python -m drowsyguard_ml.prepare --dataset rldd  --root /path/to/RLDD    # FoldN_partM/<subject>/<label>.<ext>
python -m drowsyguard_ml.prepare --dataset nthu  --root /path/to/NTHU    # <subject>/<scenario>/<behaviour>.avi + *_drowsiness.txt
python -m drowsyguard_ml.prepare --dataset yawdd --root /path/to/YawDD   # 12-FemaleNoGlasses-Yawning.avi
```

Runs resume: finished videos are skipped. Add `--include-low-vigilant` to also process RLDD's label 5. With `--subjects-per-fold`, subjects are taken smallest-download-first, round-robin across folds, which keeps folds balanced but favours lower-bitrate recordings.

## 2. Build windows

```bash
python -m drowsyguard_ml.build_features
```

Each row is one window of up to 60 s, every 10 s, with 36 features:

| Group | Features |
|---|---|
| Eye opening (EAR) | absolute and baseline-relative mean, spread, percentiles, left/right asymmetry |
| PERCLOS | 60 s, last 15 s, and a stricter deep-closure variant |
| Blink frequency | rate, duration, amplitude, reopening speed, interval regularity |
| Eye closure duration | long closures per minute, longest, mean |
| Mouth (MAR) | level, spread, peak, yawns per minute, time yawning |
| Head pose and nodding | pitch level, spread and lowest point; yaw and roll spread; speed; nods per minute; time head-down |
| Temporal behaviour | EAR slope, and first-half to second-half change in PERCLOS, blink rate and pitch |
| Quality | fraction of the window with a visible face |

Each video gets a 4 s baseline from its opening seconds, as a live session does, and its first and last 5 s are dropped.

## 3. Train

```bash
python -m drowsyguard_ml.train
```

Splits are by subject and use RLDD's official folds: folds 1–3 train, fold 4 validation, fold 5 test. Subjects from other datasets get a stable hashed fold. Training fails rather than continuing if a subject appears in two partitions.

1. Random Forest and XGBoost candidates are fitted on train and compared on validation.
2. The best ROC-AUC wins; its decision threshold is the one with the best validation F1.
3. That configuration is refitted on train and validation together.
4. Test subjects are scored once. Nothing is tuned afterwards.
5. The configuration is rerun leaving out each official fold, as a cross-subject stability check.

The app's own PERCLOS rule is scored on the same windows so the comparison is visible.

Outputs:

| File | Contents |
|---|---|
| `artifacts/model.joblib` | Fitted model, feature names, threshold, feature config |
| `artifacts/preprocessing.json` | Feature config, landmarker, subject lists per partition |
| `artifacts/feature_metadata.json` | Feature order and dtype |
| `artifacts/metrics.json` | Every metric from the run |
| `public/ml/model.json` | Trees as flat arrays for the browser |
| `public/ml/metrics.json` | What the dashboard displays |
| `public/ml/model-check.json` | Real test rows with Python's outputs, replayed by vitest |

Export fails unless the exported trees reproduce `predict_proba`. Until a run succeeds these files do not exist, and the dashboard says no model is installed.

## Inference outside the browser

```bash
python -m drowsyguard_ml.inference path/to/video.mp4
```

```python
from drowsyguard_ml.inference import DrowsinessClassifier
DrowsinessClassifier().predict(feature_vector)
```

## Keeping Python and TypeScript in step

`src/lib/mlFeatures.ts` is a port of `features.py`. After changing either:

```bash
python -m drowsyguard_ml.parity --real data/frames/rldd/<subject>/<video>.parquet
cd .. && npm test
```

## How the app uses the model

Once calibrated, the app thins frames to 15 Hz, keeps a 60 s window and evaluates the model once a second, starting after 30 s. Five consecutive drowsy verdicts raise the existing **drowsy** warning. The model cannot raise **critical** or clear a rule-based alert, so eyes-shut and head-drop detection behave exactly as before.

"Confidence" is the classifier's probability for the class it chose. It is not calibrated, so read it as a score.
