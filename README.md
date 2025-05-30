# IDG‑DREAM Drug‑Kinase Binding Prediction (Class Project)

**Predicting small‑molecule ⇆ kinase binding affinities with ESM‑2 protein embeddings and XGBoost** — built as a *course project* for the University of Pittsburgh **Scalable Machine Learning** class (Spring 2025). We retro‑ran the public IDG‑DREAM dataset for practice; no results were submitted to the official challenge.

---

| Split                  | Spearman ρ | ROC AUC † |
| ---------------------- | ---------- | --------- |
| Test 1                 | **0.327**  | —         |
| Test 2                 | **0.426**  | —         |
| Independent kinase set | **0.624**  | 0.759     |

† ROC‑AUC is reported only for the independent kinase evaluation where binary actives/inactives are provided.

---

## Project highlights

* **Protein features** – Mean‑pooled embeddings from the 3‑billion‑parameter **ESM‑2** model (`facebook/esm2_t36_3B`).
* **Ligand features** – 2048‑bit ECFP4 fingerprints (`rdkit`).
* **Model** – Gradient‑boosted decision trees (**XGBoost 3.0**), tuned via early stopping on a 90/10 validation split.
* **Single‑command pipeline** – `auto_embed/main.py` orchestrates *featurisation → training → evaluation → prediction*.
* **Reproducible GPU dev‑container** – Open in VS Code → *Reopen in Container* to get CUDA 12 + Python 3.12 + all deps.

---

## Directory layout

```text
.
├── auto_embed/          # ⬅ final pipeline (focus of this repo)
│   ├── esm_embed.py     # extract & cache ESM‑2 embeddings
│   ├── featurize.py     # ECFP + embedding → X,y .npy files
│   ├── train_xgboost.py # train + checkpoint model
│   ├── predict.py       # inference on CSV splits
│   ├── eval.py          # Spearman / ROC‑AUC metrics
│   ├── submit.py        # formats competition file (kept for completeness)
│   └── main.py          # orchestrates everything end‑to‑end
├── auto_XGB/            # earlier ligand‑only baseline
├── first_try/           # prototype notebooks & scripts
├── data/
│   ├── raw/             # ← place competition CSVs & `kinase_seqs.txt` here
│   └── processed/       # cached embeddings & features
├── models/              # trained XGBoost checkpoints
├── plots/               # figures for class report
├── requirements.txt     # Python deps
└── .devcontainer/       # VS Code + CUDA environment
```

---

## Quick‑start (GPU)

```bash
# 1 · Clone
$ git clone https://github.com/<your‑org>/idg‑dream‑kinase.git
$ cd idg‑dream‑kinase

# 2 · Env (conda/venv or dev‑container)
$ pip install -r requirements.txt

# 3 · Data (not distributed)
$ mkdir -p data/raw && mv <files> data/raw/
  #  train.csv · test1.csv · test2.csv · kinase_seqs.txt

# 4 · ESM‑2 embeddings (~3 h on 1×3090)
$ python auto_embed/esm_embed.py \
      --seq_path data/raw/kinase_seqs.txt \
      --out_path data/processed/esm2_embeddings.npy

# 5 · End‑to‑end run
$ python auto_embed/main.py
  # → models/, preds_test?.txt, results/*.json

# 6 · Local evaluation (needs ground‑truth label file)
$ python auto_embed/eval.py data/raw/test2_labels.txt preds_test2.txt
```

### Minimal API

```python
from auto_embed.predict import load_model, predict_on_dataset

yhat = predict_on_dataset(
    csv_path="my_pairs.csv",          # smiles, uniprot columns
    model_path="models/xgboost_model.pkl",
    embeddings_path="data/processed/esm2_embeddings.npy"
)
yhat.to_csv("predictions.txt", sep=" ", index=False, header=False)
```

---

## Hyper‑parameters

| parameter                | value                      |
| ------------------------ | -------------------------- |
| `max_depth`              | 8                          |
| `n_estimators`           | 1000 (early stopping = 50) |
| `learning_rate`          | 0.03                       |
| `subsample`              | 0.9                        |
| `colsample_bytree`       | 0.5                        |
| `min_child_weight`       | 1                          |
| `gamma`                  | 0.0                        |
| `reg_alpha / reg_lambda` | 0 / 1                      |

---

## Reproducing the class scores

1. Follow *Quick‑start*.
2. Use `auto_embed/eval.py` with the hidden‑label files provided by the teaching staff or your own split.
3. Your metrics should match the table above within ±0.01 ρ (GPU nondeterminism).

> ✦ **Note** `submit.py` is kept for completeness as it was submitted to a local class webpage. As you likely dont have access to this, please ignore.

---

## Acknowledgements

* Facebook AI for **ESM‑2**.
* The **IDG‑DREAM** organisers for the open dataset.
* Dr. David Koes & teaching team for the Scalable ML course inspiration.

---

## License

Released under the **MIT License**. PRs welcome!
