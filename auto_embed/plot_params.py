
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("data/logs_xgboost_embed.csv")

# Convert columns to numeric
df["test1_spearman"] = pd.to_numeric(df["test1_spearman"], errors="coerce")
df["test2_spearman"] = pd.to_numeric(df["test2_spearman"], errors="coerce")
df["combined_spearman"] = df["test1_spearman"] + df["test2_spearman"]

df["max_depth"] = pd.to_numeric(df["max_depth"], errors="coerce")
df["learning_rate"] = pd.to_numeric(df["learning_rate"], errors="coerce")
df["num_boost_round"] = pd.to_numeric(df["num_boost_round"], errors="coerce")

df = df.dropna(subset=["combined_spearman", "max_depth", "learning_rate", "num_boost_round"])

# Output folder
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Plotting function
def plot_binned_param(df, param, bins=10):
    df = df.copy()
    df["binned"] = pd.cut(df[param], bins=bins)
    grouped = df.groupby("binned")["combined_spearman"].mean().reset_index()

    plt.figure()
    plt.bar(grouped["binned"].astype(str), grouped["combined_spearman"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Combined Spearman (test1 + test2)")
    plt.xlabel(param)
    plt.title(f"Binned {param} vs Combined Spearman")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{param}_vs_spearman.png"))
    plt.close()

# Generate plots
for param in ["max_depth", "learning_rate", "num_boost_round"]:
    if param == 'max_depth':
        plot_binned_param(df, param, bins=13)
    else:
        plot_binned_param(df, param, bins=20)
