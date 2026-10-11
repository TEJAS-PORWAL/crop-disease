# 🚀 Setup Guide — Clone & Run with the Full Dataset

This is a complete, start-to-finish walkthrough for running the Crop Disease
Detection System on a fresh machine, including downloading the real
PlantVillage dataset and training a working model.

> **Important:** A cloned repo contains **code only**. The dataset, the trained
> model, and the Python virtual environment are **not** stored in Git (they are
> listed in `.gitignore`). You will create all three locally by following the
> steps below.

---

## 0. Prerequisites

Install these first and confirm they work:

| Tool | Version | Check command |
|------|---------|---------------|
| Python | 3.10 – 3.12 | `python --version` |
| Git | any recent | `git --version` |
| Disk space | ~3 GB free | — |
| Kaggle account | free | https://www.kaggle.com (needed for the dataset) |

> If `python` opens the Microsoft Store on Windows, install Python from
> https://www.python.org and re-open your terminal.

---

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Crop disease"
```

Replace `<your-repo-url>` with your GitHub URL, e.g.
`https://github.com/<username>/crop-disease.git`.

---

## 2. Create a virtual environment & install dependencies

A virtual environment keeps this project's packages isolated from the rest of
your system.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> This installs TensorFlow and friends and may take several minutes.
>
> **Command style used below:** the Windows form `.\.venv\Scripts\python.exe`.
> On macOS/Linux, activate the venv (`source .venv/bin/activate`) once, then use
> plain `python` and `streamlit`.

---

## 3. Set up your Kaggle API token (for the dataset)

The dataset is downloaded from Kaggle, which needs an API token.

1. Log in to Kaggle → click your avatar → **Settings**.
2. Scroll to the **API** section → click **Create New Token**.
3. Your browser downloads a file named `kaggle.json`.
4. Move `kaggle.json` to the correct folder:

   **Windows:**
   ```powershell
   New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
   Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
   ```

   **macOS / Linux:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

> **Alternative (no file):** set environment variables instead:
> ```powershell
> $env:KAGGLE_USERNAME="your_username"
> $env:KAGGLE_KEY="your_key"
> ```
> Keep `kaggle.json` out of the project folder — it's a secret.

---

## 4. Download the dataset & create train/val split

This downloads PlantVillage and automatically splits it into
`data/train/` and `data/val/` (80/20 per class).

**Option A — Tomato crops only (recommended first run, faster):**

```powershell
.\.venv\Scripts\python.exe -m src.download_data --crop Tomato
```

**Option B — The full dataset (all 38 classes, ~2 GB download):**

```powershell
.\.venv\Scripts\python.exe -m src.download_data
```

**Optional — Potato too (added to the existing tomato data):**

```powershell
.\.venv\Scripts\python.exe -m src.download_data --crop Potato --no-clean
```

Handy flags:

| Flag | What it does |
|------|--------------|
| `--crop Tomato` | Only one crop's classes |
| `--classes A B C` | Only the named class folders |
| `--max-per-class 300` | Cap images per class (quick tests) |
| `--no-clean` | Add to existing data instead of clearing it |
| `--skip-download` | Reuse an already-downloaded `data/raw` |

When it finishes you'll see how many train/val images landed in each class.

---

## 5. Train the model

> **Run Step 4 first.** Training does **not** download anything from Kaggle — it
> only reads the images already in `data/train/` and `data/val/` that Step 4
> created. If those folders are empty, training stops with a "dataset directory
> is missing or empty" error.

```powershell
.\.venv\Scripts\python.exe -m src.train
```

What happens:
- Loads `data/train` and `data/val`
- Phase 1: trains the classifier head (MobileNetV2 frozen)
- Phase 2: fine-tunes the top layers
- Saves the model to `models/crop_disease_model.keras`
- Saves class labels to `models/class_names.json`
- Saves training curves to `models/training_history.png`

Training on CPU takes a while; a GPU is much faster but not required.

---

## 6. Evaluate (optional but recommended)

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

Prints a per-class precision/recall/F1 report and writes a confusion matrix to
`models/confusion_matrix.png`.

---

## 7. Predict a single image from the command line

```powershell
.\.venv\Scripts\python.exe -m src.predict --image "data\val\Tomato___healthy\<some-file>.JPG"
```

Prints the predicted class, confidence, top matches, and treatment advice.

---

## 8. Launch the web app 🎉

```powershell
.\.venv\Scripts\streamlit run app.py
```

Then open the URL it prints (usually **http://localhost:8501**) in your browser.
Upload a leaf photo to get a diagnosis, confidence score, and treatment steps.

To stop the app, press `Ctrl+C` in the terminal.

---

## ⚡ TL;DR — all commands (Windows, real dataset)

```powershell
git clone <your-repo-url>
cd "Crop disease"

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# (put kaggle.json in C:\Users\<you>\.kaggle\ first — see Step 3)
.\.venv\Scripts\python.exe -m src.download_data --crop Tomato
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\streamlit run app.py
```

## 🧪 No Kaggle account? Quick synthetic demo

Runs the full pipeline with generated placeholder images (not real leaves, so
predictions are not meaningful — pipeline test only):

```powershell
.\.venv\Scripts\python.exe -m src.generate_synthetic_data
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\streamlit run app.py
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| App says **"No trained model found"** | You skipped Step 5. Train a model first. |
| **"Kaggle API credentials not found"** | `kaggle.json` isn't in `~/.kaggle/` — see Step 3. |
| `pip` can't find **TensorFlow** | Use Python 3.10–3.12; newer versions lack wheels on some platforms. |
| `python` opens Microsoft Store (Windows) | Install from python.org, reopen the terminal. |
| App HTML shows as **raw text** | Hard refresh the browser: `Ctrl+Shift+R`. |
| Download is **very slow** | The full dataset is ~2 GB; use `--crop Tomato` or `--max-per-class`. |

---

## 📂 What gets created locally (not in Git)

```
.venv/     # your virtual environment
data/
├── raw/   # the downloaded + extracted Kaggle archive
├── train/ # 80% split, one folder per class
└── val/   # 20% split, one folder per class
models/
├── crop_disease_model.keras
├── class_names.json
├── training_history.png
└── confusion_matrix.png
```

These are regenerated by the steps above, which is why they aren't committed.
