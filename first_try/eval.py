import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

true_labels_path = "data/raw/test2_labels.txt"
predictions_path = "data/predictions/test2_pred.txt"

# Load both files, skipping the header row and assigning custom column names
true_df = pd.read_csv(true_labels_path, sep=r'\s+', header=0, names=["smiles", "uniprot", "true"])
pred_df = pd.read_csv(predictions_path, sep=r'\s+', header=0, names=["smiles", "uniprot", "pred"])


# Merge predictions with true labels
merged = pd.merge(true_df, pred_df, on=["smiles", "uniprot"])

# Spearman correlation
spearman = spearmanr(merged["true"], merged["pred"]).correlation
print(f"Spearman: {spearman:.4f}")

# ROC AUC for classification (active = pKd >= 6.0)
binary_true = (merged["true"] >= 6.0).astype(int)
roc_auc = roc_auc_score(binary_true, merged["pred"])
print(f"ROC AUC: {roc_auc:.4f}")
