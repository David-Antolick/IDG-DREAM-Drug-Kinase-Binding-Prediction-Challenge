import os
import pandas as pd
import numpy as np
from rdkit import Chem
from collections import Counter
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm




def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(generator.GetFingerprint(mol), dtype=np.float32)


def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def seq_to_kmer_vector(seq, k, vocab):
    kmers = get_kmers(seq, k)
    counts = Counter(kmers)
    return np.array([counts.get(kmer, 0) for kmer in vocab], dtype=np.float32)


def load_uniprot_sequences(path):
    seqs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                uid, seq = line.strip().split()
                seqs[uid] = seq.upper()
    return seqs



if __name__ == "__main__":
    data_path = "data/processed/train_dataset.csv"
    seqs_path = "data/raw/kinase_seqs.txt"
    x_out_path = "data/processed/X_train.npy"
    y_out_path = "data/processed/y_train.npy"

    ecfp_bits = 2048
    k = 3

    df = pd.read_csv(data_path)
    sequences = load_uniprot_sequences(seqs_path)

    valid_seqs = [sequences[uid] for uid in df["uniprot"] if uid in sequences]
    vocab = sorted(set(kmer for seq in valid_seqs for kmer in get_kmers(seq, k)))

    with open("models/kmer_vocab.txt", "w") as f:
        for kmer in vocab:
            f.write(kmer + "\n")


    x = []
    y = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):

        if row["uniprot"] not in sequences:
            continue
        ecfp = smiles_to_ecfp(row["smiles"], n_bits=ecfp_bits)
        protein_vec = seq_to_kmer_vector(sequences[row["uniprot"]], k, vocab)
        x.append(np.concatenate([ecfp, protein_vec]))
        y.append(row["pchembl"])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    os.makedirs(os.path.dirname(x_out_path), exist_ok=True)
    np.save(x_out_path, x)
    np.save(y_out_path, y)

    print(f"Saved X: {x.shape}, y: {y.shape}")
