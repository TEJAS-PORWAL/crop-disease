# 🏗️ Crop Disease Detection — App Architecture

An end-to-end deep-learning system that classifies crop leaf diseases from an
image and recommends treatment. It is built around **MobileNetV2 transfer
learning** and ships three entrypoints: a **training/evaluation pipeline**, a
**command-line predictor**, and a **Streamlit web app**.

---

## 1. High-level architecture

The project is organized into clean layers, with `config.py` as the single
source of truth that every layer reads from.

```
                        ┌───────────────────────────┐
                        │         config.py         │
                        │  paths · image size ·     │
                        │  hyperparameters · seeds  │
                        └─────────────┬─────────────┘
                                      │ (imported everywhere)
      ┌───────────────────────────────┼───────────────────────────────┐
      │                               │                               │
┌─────▼──────┐   ┌──────────────┐   ┌─▼───────────┐   ┌─────────────┐   ┌──▼───────────┐
│  DATA      │   │   MODEL      │   │  TRAINING   │   │ INFERENCE   │   │   DOMAIN     │
│  LAYER     │   │   LAYER      │   │  LAYER      │   │  LAYER      │   │   KNOWLEDGE  │
├────────────┤   ├──────────────┤   ├─────────────┤   ├─────────────┤   ├──────────────┤
│download_   │   │ model.py     │   │ train.py    │   │ predict.py  │   │ treatments.py│
│  data.py   │   │ (MobileNetV2 │   │ evaluate.py │   │ (CLI + API) │   │ (advice map) │
│generate_   │   │  + head)     │   │             │   │             │   │              │
│  synthetic │   │              │   │             │   │             │   │              │
│data_loader │   │              │   │             │   │             │   │              │
└────────────┘   └──────────────┘   └─────────────┘   └─────────────┘   └──────────────┘
      │                                     │                 │                 │
      └── produces data/train, data/val     └── writes        └── reads         │
                                             models/*.keras    models/           │
                                             class_names.json  + app.py ─────────┘
                                                                     (Streamlit UI)
```

**Design principles baked in**
- **Single config** — all paths and hyperparameters live in `config.py`, resolved relative to the file so the project runs from any working directory.
- **Preprocessing lives inside the model graph** — augmentation and MobileNetV2 scaling are layers in the model, so training and inference feed the exact same raw `[0, 255]` input and can never drift.
- **Class order is persisted** — `class_names.json` is saved at training time and reloaded for every prediction, keeping label indices aligned forever.
- **Runnable without a download** — a synthetic data generator lets the whole pipeline run immediately; swap in real data for real results.

---

## 2. What each file does

### Root
| File | Responsibility |
|------|----------------|
| `config.py` | Central configuration. Defines all paths (`DATA_DIR`, `MODELS_DIR`, `MODEL_PATH`, `CLASS_NAMES_PATH`, plot paths), image size (`160×160×3`), batch size, seed, and the two-phase training hyperparameters (head epochs/LR, fine-tune epochs/LR, `FINE_TUNE_AT`, dropout, early-stopping patience). Provides `ensure_dirs()`. |
| `app.py` | **Streamlit web app** — the interactive UI. Caches the model + class names, lets the user upload a leaf photo, preprocesses it, runs prediction, and renders a color-coded result banner (healthy / disease / uncertain), a treatment card, and the top-5 probability list. Flags predictions below 50% confidence as uncertain. |
| `requirements.txt` | Python dependencies: TensorFlow, NumPy, Pillow, Matplotlib, scikit-learn, Streamlit, pandas, seaborn, kaggle, tqdm. |
| `README.md` | Project overview, dataset layout, and quick-start commands. |

### `src/` — the core package
| File | Responsibility |
|------|----------------|
| `__init__.py` | Marks `src` as a package so modules run via `python -m src.<name>`. |
| `config` *(imported from root)* | — |
| `data_loader.py` | **Data layer.** Builds `tf.data` pipelines from the ImageFolder layout via `image_dataset_from_directory` (`load_datasets`), defines the augmentation block (`build_augmentation`: random flip/rotation/zoom/contrast), and persists/loads the class-name ordering (`save_class_names` / `load_class_names`). Validates that the data directories exist and are non-empty. |
| `model.py` | **Model layer.** `build_model()` assembles the network: input → augmentation → MobileNetV2 preprocessing → frozen MobileNetV2 base (ImageNet) → global average pooling → dropout → softmax `Dense`. `compile_for_head()` and `compile_for_fine_tune()` configure the two training phases; `get_base_model()` locates the nested base submodel robustly across Keras versions. |
| `train.py` | **Training layer.** Two-phase transfer learning: Phase 1 trains the classifier head with the base frozen; Phase 2 unfreezes the top layers (`FINE_TUNE_AT`) and fine-tunes at a low LR (BatchNorm kept frozen). Uses `EarlyStopping`, saves the `.keras` model and `class_names.json`, and writes the accuracy/loss history plot. |
| `evaluate.py` | **Evaluation layer.** Loads the trained model and validation set, runs predictions, prints a scikit-learn classification report, and saves a confusion-matrix heatmap and overall accuracy. |
| `predict.py` | **Inference layer / CLI.** Reusable helpers — `load_trained_model()` (cached), `preprocess_image()`, `predict_array()`, `predict_image()` (top-k) — plus a CLI (`--image`, `--top-k`) that prints the diagnosis and the matching treatment. |
| `download_data.py` | **Data acquisition.** Downloads the PlantVillage dataset from Kaggle, locates the `color` variant inside the archive, and copies classes into `data/train` + `data/val` with a per-class split. Supports `--crop`, `--classes`, `--max-per-class`, `--val-split`, `--skip-download`, `--no-clean`. |
| `generate_synthetic_data.py` | **Toy data generator.** Creates deterministic synthetic images (per-class color + spots) so the full pipeline can run end-to-end without any download. Not real leaves — for plumbing validation only. |
| `treatments.py` | **Domain knowledge.** Maps each PlantVillage class name to a `Treatment` (summary, cause, step-by-step advice) with a safe default. `get_treatment()` and `is_healthy()` are consumed by both the CLI and the web app. |

### Generated artifacts (`models/`, `data/`)
| Path | Produced by | Purpose |
|------|-------------|---------|
| `data/train/`, `data/val/` | `download_data.py` or `generate_synthetic_data.py` | Class-folder image sets for training/validation. |
| `data/raw/` | `download_data.py` | Extracted Kaggle archive. |
| `models/crop_disease_model.keras` | `train.py` | The trained model. |
| `models/class_names.json` | `train.py` | Class order, reloaded at inference to keep labels aligned. |
| `models/training_history.png` | `train.py` | Accuracy/loss curves. |
| `models/confusion_matrix.png` | `evaluate.py` | Validation confusion matrix. |

---

## 3. End-to-end flow

### A. Build-time pipeline (data → trained model)
```
download_data.py  ─┐
                   ├─►  data/train, data/val  ──►  data_loader.load_datasets()
generate_synthetic ┘                                      │
                                                          ▼
                                            model.build_model()  (MobileNetV2 + head)
                                                          │
                                        ┌─────────────────┴─────────────────┐
                                        ▼                                   ▼
                              Phase 1: train head                 Phase 2: fine-tune top
                              (base frozen)                        (low LR, BN frozen)
                                        └─────────────────┬─────────────────┘
                                                          ▼
                                   train.py saves  →  crop_disease_model.keras
                                                        class_names.json
                                                        training_history.png
                                                          │
                                                          ▼
                                   evaluate.py  →  confusion_matrix.png + report
```

### B. Run-time inference (user image → diagnosis + treatment)
```
            ┌──────────────────────────┐         ┌──────────────────────────┐
            │   CLI:  src/predict.py   │         │   Web:  app.py (Streamlit)│
            └────────────┬─────────────┘         └────────────┬─────────────┘
                         │                                     │
                 user supplies image                   user uploads leaf photo
                         │                                     │
                         ▼                                     ▼
             preprocess → resize 160×160              preprocess → resize 160×160
                         │                                     │
                         └──────────────┬──────────────────────┘
                                        ▼
                      load crop_disease_model.keras  +  class_names.json
                                        ▼
                              model.predict() → probabilities
                                        ▼
                              argmax → top label + confidence
                                        ▼
                          treatments.get_treatment(label)
                                        ▼
            ┌───────────────────────────┴───────────────────────────┐
            ▼                                                         ▼
   CLI prints label, top-k,                        App renders banner (healthy/
   and treatment steps                             disease/uncertain) + treatment
                                                   card + top-5 bars
```

---

## 4. Typical usage

```bash
# 1. Install
pip install -r requirements.txt

# 2a. Get real data...
python -m src.download_data --crop Tomato
# 2b. ...or generate synthetic data to test the pipeline
python -m src.generate_synthetic_data

# 3. Train (two-phase transfer learning)
python -m src.train

# 4. Evaluate
python -m src.evaluate

# 5. Predict a single image (CLI)
python -m src.predict --image path/to/leaf.jpg

# 6. Launch the web app
streamlit run app.py
```
