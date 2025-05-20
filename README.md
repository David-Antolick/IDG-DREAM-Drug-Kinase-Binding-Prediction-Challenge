# Drug-Kinase Binding Prediction
_Modular XGBoost pipelines with ECFP6 + protein features (3-mer or ESM2)_

Predict the pChEMBL affinity of a small-molecule inhibitor for a kinase using
only **SMILES** and **protein sequence** information.  
This codebase was built for a retrospective run of the
**IDG-DREAM Drug-Kinase Binding Prediction Challenge** (COBB 2060, University of
Pittsburgh). The final, top-scoring solution lives in
`auto_embed/` and fuses ligand fingerprints with **ESM2-3B** protein
embeddings. :contentReference[oaicite:0]{index=0}

---

## Project Overview

| Stage | What happens |
|-------|--------------|
| **Data ingest** | Bulk ChEMBL activities (`pChEMBL`, SMILES, target IDs) and kinase sequences (UniProt FASTA). |
| **Ligand encoding** | RDKit 2048-bit **ECFP6** fingerprints. |
| **Protein encoding** | <br>• Baseline: 3-mer bag-of-words (`auto_XGB/`).<br>• **Final**: mean-pooled **ESM2-3B** embeddings (`auto_embed/`). |
| **Model** | GPU **XGBoost** regressor (`objective=reg:squarederror`). |
| **Hyper-opt** | ~500 Bayesian trials (Optuna) on a held-out kinase split. |
| **Packaging** | Trained model + scalers serialised with `cloudpickle` for 1-file deployment. |

---

## Repository Layout

```

├── .devcontainer/        # VS Code CUDA dev-container
├── auto\_embed/           # ⭐ Final ESM2 pipeline
│   ├── esm\_embed.py
│   ├── featurize.py
│   ├── train\_xgboost.py
│   ├── predict.py
│   └── main.py
├── auto\_XGB/             # Baseline k-mer pipeline
├── first\_try/            # Early notebooks & experiments
├── plots/                # Saved figures (git-ignored)
├── data/                 # Large files (git-ignored)
├── models/               # Checkpoints (git-ignored)
├── requirements.txt
└── README.md             # You are here

````:contentReference[oaicite:1]{index=1}

---

## Quick-Start

> Tested on Ubuntu 22.04, Python 3.12, CUDA 12.x.

```bash
# 1. Clone & create an env
git clone https://github.com/David-Antolick/IDG-DREAM-Drug-Kinase-Binding-Prediction-Challenge.git
cd IDG-DREAM-Drug-Kinase-Binding-Prediction-Challenge
rebuild and reopen in container

# 2. Download public ChEMBL + kinase FASTA
 

# 3. Generate ESM2 embeddings
python auto_embed/esm_embed.py \
    --fasta  data/raw/kinase_seqs.fasta \
    --out    data/processed/esm2_embeddings.npy

# 4. Featurise and train
python auto_embed/featurize.py  --embeddings data/processed/esm2_embeddings.npy
python auto_embed/train_xgboost.py --output models/esm2_xgb.pkl

# 5. Inference on the DREAM test set
python auto_embed/predict.py \
    --model models/esm2_xgb.pkl \
    --input data/test/test1.txt \
    --output predictions/test1_pred.txt
````

All steps finish inside the 20-minute limit mandated by the course
grader (12 CPU, 24 GB GPU RAM).

---

## Results

| Dataset |           Spearman ↑ | RMSE ↓ | AUC (indep.) |
| ------- | -------------------: | -----: | ------------ |
| Test 1  |            **0.327** |   0.87 | –            |
| Test 2  |            **0.426** |   0.78 | –            |
| Indep.  | **0.624** (p < 1e-4) |      – | **0.759**    |

These scores more than **double** the challenge’s supplied mean-baseline on both
official rounds. ([GitHub][1])

---

## How This Repo Demonstrates Impact

* **End-to-end ownership** – data engineering → feature design → hyper-opt →
  packaging.
* **Modern protein LMs** – shows practical benefit of ESM2 embeddings when
  structural data are unavailable.
* **Reproducibility** – dev-container and pinned requirements keep grader and
  recruiter environments consistent.

---

## License

Apache-2.0 (see `LICENSE`).


[1]: https://github.com/David-Antolick/IDG-DREAM-Drug-Kinase-Binding-Prediction-Challenge "GitHub - David-Antolick/IDG-DREAM-Drug-Kinase-Binding-Prediction-Challenge"
