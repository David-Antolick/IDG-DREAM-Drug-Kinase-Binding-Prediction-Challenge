import os
import numpy as np
import cloudpickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from scipy.stats import spearmanr
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm

def train_catboost(
    x_path="data/processed/X_train.npy",
    y_path="data/processed/y_train.npy",
    model_path="models/catboost_model.pkl",
    test_size=0.1,
    max_depth=15,
    learning_rate=0.45164012945943416,
    num_boost_round=397,
    seed=42,
    verbose=True
):
    x = np.load(x_path)
    y = np.load(y_path)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    model = CatBoostRegressor(
        depth=max_depth,
        learning_rate=learning_rate,
        iterations=num_boost_round,
        random_seed=seed,
        loss_function='RMSE',
        task_type="GPU",
        verbose=False
    )

    model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)


    y_pred = model.predict(x_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    spearman = spearmanr(y_val, y_pred).correlation

    if verbose:
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Spearman: {spearman:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        cloudpickle.dump(model, f)

    return {
        "booster": model,
        "rmse": rmse,
        "spearman": spearman,
        "model_path": model_path
    }
