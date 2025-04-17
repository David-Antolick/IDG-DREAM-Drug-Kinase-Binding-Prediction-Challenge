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

def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    return np.array(fp.ToList(), dtype=np.float32)

def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def seq_to_kmer_vector(seq, k, vocab):
    counts = Counter(get_kmers(seq, k))
    return np.array([counts.get(kmer, 0) for kmer in vocab], dtype=np.float32)

def build_vocab(seqs, k):
    kmers = set()
    for seq in seqs.values():
        kmers.update(get_kmers(seq, k))
    return sorted(kmers)

if __name__ == "__main__":
    test_path = "data/raw/test1.txt"
    seqs_path = "data/raw/kinase_seqs.txt"
    model_path = "models/xgboost_model.pkl"
    out_path = "data/predictions/test1_pred.txt"

    k = 3
    ecfp_bits = 2048

    print("Loading test set...")
    df = pd.read_csv(test_path, sep=" ")

    print("Loading model and sequences...")
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)

    seqs = load_uniprot_sequences(seqs_path)
    with open("models/kmer_vocab.txt") as f:
        vocab = [line.strip() for line in f]


    x = []
    valid_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        smile, uniprot = row["SMILES"], row["UniProt"]
        if uniprot not in seqs:
            continue
        ecfp = smiles_to_ecfp(smile, n_bits=ecfp_bits)
        prot = seq_to_kmer_vector(seqs[uniprot], k, vocab)
        x.append(np.concatenate([ecfp, prot]))
        valid_rows.append(i)

    print(f"Predicting on {len(x)} valid rows...")
    dtest = xgb.DMatrix(np.array(x, dtype=np.float32))
    preds = model.predict(dtest)

    df = df.iloc[valid_rows].copy()
    df["pChEMBL"] = preds
    df = df[["SMILES", "UniProt", "pChEMBL"]]
    df.to_csv(out_path, index=False, sep=" ")
    print(f"Saved predictions to {out_path}")
