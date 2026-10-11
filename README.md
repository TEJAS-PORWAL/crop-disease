# 🌱 Crop Disease Detection System

An end-to-end AI/ML project that detects crop diseases from leaf images using
deep learning (transfer learning with MobileNetV2). It includes a training
pipeline, a single-image prediction CLI, treatment recommendations, and a
polished Streamlit web app.

## Features

- **Transfer learning** on MobileNetV2 (ImageNet weights)
- **Two-phase training**: train the classifier head, then fine-tune top layers
- **Data augmentation** for better generalization
- **Evaluation**: accuracy/loss curves, confusion matrix, classification report
- **Prediction CLI** for scoring a single leaf image
- **Treatment guidance** for each detected disease
- **Streamlit web app** with drag-and-drop inference and a confidence indicator
- **Kaggle downloader** that fetches and splits the real PlantVillage dataset
- **Synthetic data generator** so you can run the whole pipeline with no download

---

## 🚀 Running the project after cloning from GitHub

A fresh clone contains **code only**. The dataset, the trained model, and the
virtual environment are intentionally **not** committed (see `.gitignore`), so
you need to create them locally. Pick one of the two paths below.

### Prerequisites

- **Python 3.10–3.12** installed and on your PATH
- **git**
- ~3 GB free disk space if you use the real dataset

### Step 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd "Crop disease"
```

### Step 2 — Create a virtual environment and install dependencies

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux (bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> The commands below use the Windows form `.\.venv\Scripts\python.exe`. On
> macOS/Linux, activate the venv (`source .venv/bin/activate`) and just use
> `python` / `streamlit` instead.

### Step 3 — Get data and train a model

You need a dataset before the app will run (there is no model in the repo).
Choose **Path A** (fast, no account) or **Path B** (real results).

#### Path A — Quick demo with synthetic data (no download, ~2 min)

```powershell
.\.venv\Scripts\python.exe -m src.generate_synthetic_data
.\.venv\Scripts\python.exe -m src.train
```

This proves the pipeline works end to end. The images are colored patches, not
real leaves, so predictions are **not meaningful** — it's only for testing.

#### Path B — Real results with the PlantVillage dataset (recommended)

1. Create a free [Kaggle](https://www.kaggle.com) account.
2. Get an API token: Kaggle → Account → Settings → **Create New Token**. This
   downloads `kaggle.json`.
3. Place `kaggle.json` at:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`

   (Or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables instead.)
4. Download and split the data, then train:

```powershell
# Download just the tomato classes (faster) and split 80/20 into train/val
.\.venv\Scripts\python.exe -m src.download_data --crop Tomato

# Or all 38 PlantVillage classes (large, ~2 GB):
# .\.venv\Scripts\python.exe -m src.download_data

.\.venv\Scripts\python.exe -m src.train
```

Useful `download_data` flags:

| Flag | Purpose |
|------|---------|
| `--crop Tomato` | Only classes for one crop |
| `--classes A B C` | Only the named class folders |
| `--max-per-class 300` | Cap images per class (quick runs) |
| `--no-clean` | Add to existing train/val instead of clearing |
| `--skip-download` | Reuse an already-downloaded `data/raw` |

### Step 4 — Use the trained model

```powershell
# Evaluate (writes models/confusion_matrix.png + a classification report)
.\.venv\Scripts\python.exe -m src.evaluate

# Predict a single image (prints class, confidence, and treatment)
.\.venv\Scripts\python.exe -m src.predict --image "path\to\leaf.jpg"

# Launch the web app, then open http://localhost:8501
.\.venv\Scripts\streamlit run app.py
```

---

## Project structure

```
Crop disease/
├── requirements.txt
├── README.md
├── config.py                       # Paths + hyperparameters
├── app.py                          # Streamlit web app
├── src/
│   ├── data_loader.py              # Dataset loading + augmentation
│   ├── model.py                    # MobileNetV2 transfer-learning model
│   ├── train.py                    # Two-phase training + fine-tuning
│   ├── evaluate.py                 # Confusion matrix + report
│   ├── predict.py                  # Single-image prediction CLI
│   ├── treatments.py               # Per-disease treatment guidance
│   ├── download_data.py            # Kaggle download + train/val split
│   └── generate_synthetic_data.py  # Toy dataset generator
├── data/                           # (gitignored) train/ and val/ subfolders
└── models/                         # (gitignored) saved model, labels, plots
```

## Dataset layout

The loaders expect an [ImageFolder](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)-style
structure, one subfolder per class:

```
data/
├── train/
│   ├── Tomato___healthy/
│   ├── Tomato___Early_blight/
│   └── ...
└── val/
    ├── Tomato___healthy/
    ├── Tomato___Early_blight/
    └── ...
```

`src/download_data.py` produces exactly this layout from the PlantVillage
dataset automatically.

## Troubleshooting

- **"No trained model found" in the app** — you haven't completed Step 3. Run a
  training path first so `models/crop_disease_model.keras` exists.
- **"Kaggle API credentials not found"** — the `kaggle.json` token isn't in
  place; see Path B, step 3.
- **`pip install` can't find TensorFlow** — use Python 3.10–3.12. TensorFlow
  does not publish wheels for every newer Python version on all platforms.
- **App HTML shows as raw text** — hard refresh the browser (Ctrl+Shift+R).

## Notes

- Models are saved in the native Keras `.keras` format with a `class_names.json`
  file so prediction and the app stay in sync with training.
- Treatment guidance is general and for educational use. Confirm serious cases
  with a local agricultural extension service.
- Keep your Kaggle `kaggle.json` out of the repo; store it in `~/.kaggle`.
