import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from scipy.stats import spearmanr
from tqdm import tqdm
import cloudpickle

x_path = "data/processed/X_train.npy"
y_path = "data/processed/y_train.npy"
model_path = "models/xgboost_model.pkl"

test_size = 0.1
max_depth = 15
learning_rate = 0.45164012945943416
num_boost_round = 397

x = np.load(x_path)
y = np.load(y_path)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=test_size, random_state=42
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


progress = tqdm(total=num_boost_round, desc="Training", unit="iter")

class TQDMCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        progress.update(1)
        return False

evals = [(dtrain, "train"), (dvalid, "valid")]

booster = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    callbacks=[TQDMCallback()],
    verbose_eval=False
)

progress.close()

y_pred = booster.predict(dvalid)
rmse = root_mean_squared_error(y_val, y_pred)
spearman = spearmanr(y_val, y_pred).correlation

print(f"Validation RMSE: {rmse:.4f}")
print(f"Spearman: {spearman:.4f}")

os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, "wb") as f:
    cloudpickle.dump(booster, f)

print(f"Model saved to {model_path}")
