import pandas as pd
import os

raw_dir = "data/raw"
processed_dir = "data/processed"

act_path = os.path.join(raw_dir, "chembl_activities.txt.gz")
map_path = os.path.join(raw_dir, "uniprot_chembl_mapping.txt")

out_path = os.path.join(processed_dir, "train_dataset.csv")


def load_activities(path):
    print("Loading activity data (with SMILES already included)...")
    df = pd.read_csv(path, sep=r"\s+", engine="python", compression="gzip")
    required_cols = {"smiles", "target", "pchembl"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}. Required: {required_cols}")
    return df[["smiles", "target", "pchembl"]]


def load_mapping(path):
    print("Loading ChEMBL → UniProt mapping...")
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2 or parts[0].startswith("#"):
                continue  # skip invalid or comment lines
            uniprot, chembl_target = parts[0], parts[1]
            mapping[chembl_target] = uniprot
    return mapping



def main():
    os.makedirs(processed_dir, exist_ok=True)

    df = load_activities(act_path)
    mapping = load_mapping(map_path)

    # Map ChEMBL target to UniProt
    df["uniprot"] = df["target"].map(mapping)
    df = df.dropna(subset=["uniprot"])
    df = df[["smiles", "uniprot", "pchembl"]].drop_duplicates()

    df.to_csv(out_path, index=False)
    print(f"Saved clean dataset to {out_path} with {len(df)} rows.")


if __name__ == "__main__":
    main()