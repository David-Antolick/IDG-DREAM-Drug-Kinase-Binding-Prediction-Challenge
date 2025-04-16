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


def build_kmer_vocab(sequences, k):
    return sorted(set(kmer for seq in sequences.values() for kmer in get_kmers(seq, k)))


def save_vocab(vocab, path):
    with open(path, "w") as f:
        for kmer in vocab:
            f.write(kmer + "\n")


def load_vocab(path):
    with open(path) as f:
        return [line.strip() for line in f]


def featurize_batch(df, sequences, vocab, ecfp_bits=2048, k=3):
    x, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        uid = row["uniprot"]
        if uid not in sequences:
            continue
        ecfp = smiles_to_ecfp(row["smiles"], n_bits=ecfp_bits)
        prot_vec = seq_to_kmer_vector(sequences[uid], k, vocab)
        x.append(np.concatenate([ecfp, prot_vec]))
        y.append(row["pchembl"])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def featurize_training_set(data_path, seqs_path, x_out_path, y_out_path, vocab_path=None, k=3, ecfp_bits=2048):
    df = pd.read_csv(data_path)
    sequences = load_uniprot_sequences(seqs_path)

    if vocab_path and os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path)
    else:
        valid_seqs = {uid: sequences[uid] for uid in df["uniprot"] if uid in sequences}
        vocab = build_kmer_vocab(valid_seqs, k)
        if vocab_path:
            save_vocab(vocab, vocab_path)

    x, y = featurize_batch(df, sequences, vocab, ecfp_bits=ecfp_bits, k=k)

    os.makedirs(os.path.dirname(x_out_path), exist_ok=True)
    np.save(x_out_path, x)
    np.save(y_out_path, y)

    print(f"Saved X: {x.shape}, y: {y.shape}")
    return vocab  # So it can be reused elsewhere
