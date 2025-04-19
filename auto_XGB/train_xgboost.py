import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from scipy.stats import spearmanr
import cloudpickle
from tqdm import tqdm


def train_xgboost(
    x_path="data/processed/X_train.npy",
    y_path="data/processed/y_train.npy",
    model_path="models/xgboost_model.pkl",
    test_size=0.1,
    max_depth=6,
    learning_rate=0.1,
    num_boost_round=300,
    seed=42,
    verbose=True
):
    x = np.load(x_path)
    y = np.load(y_path)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": max_depth,
        "eta": learning_rate
    }

    class TQDMCallback(xgb.callback.TrainingCallback):
        def __init__(self, total):
            self.pbar = tqdm(total=total, desc="Training", unit="iter")

        def after_iteration(self, model, epoch, evals_log):
            self.pbar.update(1)
            return False

        def after_training(self, model):
            self.pbar.close()
            return model  # <--- This is the required fix


    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        callbacks=[TQDMCallback(num_boost_round)] if verbose else None,
        verbose_eval=False
    )

    y_pred = booster.predict(dvalid)
    rmse = root_mean_squared_error(y_val, y_pred)
    spearman = spearmanr(y_val, y_pred).correlation

    if verbose:
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Spearman: {spearman:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        cloudpickle.dump(booster, f)

    return {
        "booster": booster,
        "rmse": rmse,
        "spearman": spearman,
        "model_path": model_path
    }
