import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import sys
import warnings

def evaluate_predictions(true_labels_path, predictions_path, verbose=False):
    """Evaluate predictions using Spearman and ROC AUC.

    Parameters:
        true_df (pd.DataFrame): DataFrame with columns ['smiles', 'uniprot', 'true']
        pred_df (pd.DataFrame): DataFrame with columns ['smiles', 'uniprot', 'pred']
        verbose (bool): If True, print evaluation metrics

    Returns:
        dict: { 'spearman': float, 'roc_auc': float }
    """
    true_df = pd.read_csv(true_labels_path, sep=r"\s+", header=0, names=["smiles", "uniprot", "true"])
    pred_df = pd.read_csv(predictions_path, sep=r"\s+", header=0, names=["smiles", "uniprot", "pred"])

    merged = pd.merge(true_df, pred_df, on=["smiles", "uniprot"])

    spearman = spearmanr(merged["true"], merged["pred"]).correlation
    binary_true = (merged["true"] >= 6.0).astype(int)
    auc = roc_auc_score(binary_true, merged["pred"])

    return spearman, auc

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if len(sys.argv) != 3:
        print("Usage: python eval.py true_labels.txt pred_labels.txt")
        sys.exit(1)

    true_path = sys.argv[1]
    pred_path = sys.argv[2]

    true_df = pd.read_csv(true_path, sep=r"\s+", header=0, names=["smiles", "uniprot", "true"])
    pred_df = pd.read_csv(pred_path, sep=r"\s+", header=0, names=["smiles", "uniprot", "pred"])

    evaluate_predictions(true_df, pred_df, verbose=True)
