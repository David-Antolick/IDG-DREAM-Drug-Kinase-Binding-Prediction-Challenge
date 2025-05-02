# 🧬 Drug-Kinase Binding Prediction with XGBoost + ESM2 Embeddings

This repository contains a modular pipeline for predicting kinase inhibitor binding affinities, combining cheminformatics (SMILES) with either k-mer sequence analysis or pretrained ESM2 embeddings. It was developed for a retrospective version of the IDG-DREAM Drug-Kinase Binding Challenge as part of COBB 2060 at the University of Pittsburgh.

---

## 🚀 Overview

- Ligands: Featurized via ECFP fingerprints (RDKit)
- Proteins: Featurized via either:
  - 3-mer frequency vectors (`auto_XGB/`)
  - ESM2-3B embeddings (`auto_embed/`)
- Model: XGBoost regressor
- Output: pChEMBL binding affinity prediction

---

## 📁 Project Structure

├── .devcontainer/         # VS Code dev container config  
├── auto_embed/            # Final modularized pipeline using ESM2 protein embeddings  
│   ├── esm_embed.py  
│   ├── featurize.py  
│   ├── main.py  
│   ├── predict.py  
│   ├── train_xgboost.py  
│   └── ...  
├── auto_XGB/              # Equivalent modular pipeline using k-mer protein features  
│   └── [same structure as auto_embed]  
├── first_try/             # Early experiments and non-modular code  
├── data/                  # Local dataset folder (not included in repo due to size)  
├── models/                # Stores trained model checkpoints (excluded from repo)  
├── plots/                 # Stores performance visualizations (excluded from repo)  
├── .gitignore  
├── README.md              # This file  
├── requirements.txt       # All dependencies  
├── checklist.txt          # Manual testing and project progress checklist


## 📊 Sample Results

| Metric           | Score                  |
|------------------|------------------------|
| Test1 Spearman   | 0.327                  |
| Test2 Spearman   | 0.426                  |
| Indep Spearman   | 0.624 (p = 0.000)      |
| Indep ROC AUC    | 0.759 (CI: 0.614–0.889) |

---

## 📚 References

- IDG-DREAM Challenge: https://www.synapse.org/Synapse:syn15667962/wiki/
- Tang et al. (2021), Nature Communications: https://www.nature.com/articles/s41467-021-23165-1
- Lin et al. (2023), Science (ESM2): https://www.science.org/doi/10.1126/science.ade2574
- ESM2 on Hugging Face: https://huggingface.co/facebook/esm2_t36_3B_UR50D

---

## 🙋 Author

**David Antolick**  
This project was submitted as part of the graduate course COBB 2060 (Spring 2025).  
Feedback and questions are welcome!

---

## 💡 Notes

- The `data/`, `models/`, and `plots/` directories are excluded from version control due to size.
