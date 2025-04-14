# IDG-DREAM Drug-Kinase Binding Prediction Challenge

This repository contains code and data processing pipelines for the IDG-DREAM Drug-Kinase Binding Prediction Challenge.

The goal is to predict binding affinity values for kinase inhibitors given a SMILES string and UniProt ID. This project involves data extraction from ChEMBL, compound and protein featurization, model development, and evaluation on DREAM challenge test sets.

## Project Structure
- `data/` — raw and processed datasets
- `models/` — saved models (.pkl, checkpoints)
- `src/` — source code (training, preprocessing, prediction)
- `notebooks/` — exploratory analysis notebooks

## Getting Started
1. Clone the repository
2. Set up Docker or local environment
3. Download required data files (see checklist)
4. Run training or evaluation scripts

## Requirements
- Python 3.12+
- PyTorch, RDKit, pandas, numpy, etc. (see `requirements.txt`)
