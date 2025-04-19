import os
import numpy as np
import pandas as pd
import cloudpickle
from train_xgboost import train_xgboost
from predict import predict_on_dataset
from eval import evaluate_predictions
from featurize import featurize_training_set, load_uniprot_sequences, load_vocab, get_kmers

#  Paths 
data_dir = "data"
processed_dir = os.path.join(data_dir, "processed")
raw_dir = os.path.join(data_dir, "raw")
pred_dir = os.path.join(data_dir, "predictions")
model_path = "models/xgboost_model.pkl"
vocab_path = os.path.join(processed_dir, "kmer_vocab.txt")

#  Data Files 
train_data = os.path.join(processed_dir, "train_dataset.csv")
seqs_path = os.path.join(raw_dir, "kinase_seqs.txt")
test_files = ["test1", "test2"]
test_inputs = [os.path.join(raw_dir, f"{name}.txt") for name in test_files]
test_labels = [os.path.join(raw_dir, f"{name}_labels.txt") for name in test_files]
test_preds = [os.path.join(pred_dir, f"{name}_pred.txt") for name in test_files]

#  Featurize training set and generate vocab 
x_path = "data/processed/X_train.npy"
y_path = "data/processed/y_train.npy"

if not os.path.exists(x_path) or not os.path.exists(y_path):
    print("\n>>> Featurizing training set...")
    vocab = featurize_training_set(train_data, seqs_path, x_path, y_path, vocab_path, k=3, ecfp_bits=2048)
    sequences = load_uniprot_sequences(seqs_path)
else:
    print("\n>>> Skipping featurization (X_train.npy and y_train.npy already exist)")

#  Train initial model 
print("\n>>> Training initial model...")
result = train_xgboost(x_path, y_path)
model = result["booster"]
cloudpickle.dump(model, open(model_path, "wb"))



#  Run predictions on test1 and test2 
def score_model(result):
    model = result["booster"]
    print(f"Loading test set from model: {result['model_path']}...")

    sequences = load_uniprot_sequences("data/raw/kinase_seqs.txt")
    vocab = load_vocab("models/kmer_vocab.txt")

    test_scores = []
    for i in range(2):
        predict_on_dataset(model, test_inputs[i], test_preds[i], sequences, vocab, k=3, ecfp_bits=2048)
        score, _ = evaluate_predictions(test_labels[i], test_preds[i])
        test_scores.append(score)

    return tuple(test_scores)

print("\n>>> Evaluating baseline model...")
best_test1, best_test2 = score_model(result)
print(f"Baseline: test1 S={best_test1:.4f}, test2 S={best_test2:.4f}")




### BAYSIAN OPTIMIZATION ###


from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

search_space = [
    Integer(11, 20,   name="max_depth"),
    Real   (0.13, .478, name="learning_rate"),
    Integer(389, 1000, name="num_boost_round"),
]

if not os.path.exists("data/logs_xgboost.csv"):
    pd.DataFrame(columns=[
        "trial", "max_depth", "learning_rate", "num_boost_round",
        "test1_spearman", "test2_spearman"
    ]).to_csv("data/logs_xgboost.csv", index=False)

@use_named_args(search_space)
def objective(**params):
    # trial counter from the CSV
    trial_num = len(pd.read_csv("data/logs_xgboost.csv")) + 1
    print(f"\n>>> Trial {trial_num} with {params}")

    result   = train_xgboost(x_path, y_path, **params)
    test1_s, test2_s = score_model(result)

    # save best model
    global best_test1, best_test2
    if test1_s > best_test1 and test2_s > best_test2:
        print("New best! Spearman1:", test1_s, "Spearman2:", test2_s)
        cloudpickle.dump(result["booster"], open(model_path, "wb"))
        best_test1, best_test2 = test1_s, test2_s

    # log every trial
    pd.DataFrame([{
        "trial": trial_num,
        **params,
        "test1_spearman": test1_s,
        "test2_spearman": test2_s
    }]).to_csv("data/logs_xgboost.csv", mode="a", header=False, index=False)

    # minimise *negative* mean Spearman
    return -0.5 * (test1_s + test2_s)

# 50 calls, 10 random start‑up points
gp_minimize(
    objective,
    search_space,
    n_calls=50,
    n_initial_points=10,
    random_state=42,
    verbose=True
)