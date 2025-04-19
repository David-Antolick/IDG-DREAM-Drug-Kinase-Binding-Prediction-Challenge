# predict.py

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from collections import Counter
from tqdm import tqdm
import cloudpickle

def load_uniprot_sequences(path):
    seqs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                uid, seq = line.strip().split()
                seqs[uid] = seq.upper()
    return seqs

def load_llm_embeddings(path):
    return np.load(path, allow_pickle=True).item()

def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    return np.array(fp.ToList(), dtype=np.float32)

def get_kmers(seq, k):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]

def seq_to_kmer_vector(seq, k, vocab):
    counts = Counter(get_kmers(seq, k))
    return np.array([counts.get(kmer, 0) for kmer in vocab], dtype=np.float32)

def predict_on_dataset(model, test_path, out_path, sequences, vocab=None,
                       embedding_path=None, use_llm=False, k=3, ecfp_bits=2048):
    print(f"Loading test set from {test_path}...")
    df = pd.read_csv(test_path, sep=r"\s+")
    X = []
    valid_rows = []

    if use_llm:
        if embedding_path is None:
            raise ValueError("Must provide embedding_path when use_llm=True")
        embeddings = load_llm_embeddings(embedding_path)
    else:
        embeddings = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        uid = row["UniProt"] if "UniProt" in row else row["uniprot"]
        if uid not in sequences:
            continue

        ecfp = smiles_to_ecfp(row["SMILES"] if "SMILES" in row else row["smiles"], n_bits=ecfp_bits)

        if use_llm:
            if uid not in embeddings:
                continue
            prot_vec = embeddings[uid]
        else:
            prot_vec = seq_to_kmer_vector(sequences[uid], k, vocab)

        X.append(np.concatenate([ecfp, prot_vec]))
        valid_rows.append(row)

    X = np.array(X, dtype=np.float32)
    dtest = xgb.DMatrix(X)
    preds = model.predict(dtest)

    out_df = pd.DataFrame(valid_rows)
    out_df["pChEMBL"] = preds
    out_df.to_csv(out_path, sep=" ", index=False)
    print(f"Predictions saved to {out_path}")
