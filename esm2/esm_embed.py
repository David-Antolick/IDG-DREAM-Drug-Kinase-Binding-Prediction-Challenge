
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer

def load_uniprot_sequences(path):
    seqs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                uid, seq = line.strip().split()
                seqs[uid] = seq.upper()
    return seqs

def extract_mean_embeddings(sequences, model, tokenizer, device):
    model.eval()
    model.to(device)

    uid_to_embedding = {}
    for uid, seq in tqdm(sequences.items(), desc="Embedding proteins"):
        toks = tokenizer(" ".join(seq), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**toks)
        reps = out.last_hidden_state.squeeze(0)
        mean_rep = reps.mean(dim=0).cpu().numpy()
        uid_to_embedding[uid] = mean_rep
    return uid_to_embedding

def save_embeddings(uid_to_embedding, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, uid_to_embedding)

def embed_and_save(seq_path, out_path, model_path="/large_models/models/facebook-esm2_t36_3B_UR50D"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path)

    seqs = load_uniprot_sequences(seq_path)
    uid_to_embedding = extract_mean_embeddings(seqs, model, tokenizer, device)
    save_embeddings(uid_to_embedding, out_path)

if __name__ == "__main__":
    embed_and_save(
        seq_path="data/raw/kinase_seqs.txt",
        out_path="data/processed/esm2_embeddings.npy"
    )
