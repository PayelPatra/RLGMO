# rl/plot_med_similarity_score.py
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = ROOT / "rl" / "rl_test_with_eval.csv"   # change if you want rl_test_results.csv instead

if not CSV_FILE.exists():
    raise FileNotFoundError(f"Missing file: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

# Accept different column names (VS Code vs Colab)
score_col = None
for c in ["similarity_score", "cosine_similarity"]:
    if c in df.columns:
        score_col = c
        break

if score_col is None:
    raise ValueError(
        f"No similarity column found. Expected one of: "
        f"['similarity_score', 'cosine_similarity'] but got: {list(df.columns)}"
    )

# Convert to numeric safely (some values might be None/NaN/strings)
df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

plt.figure(figsize=(10, 5))
df[score_col].plot(kind="line", marker="o", linestyle="-")

plt.title(f"{score_col} per Patient")
plt.xlabel("Row Index")
plt.ylabel(score_col)
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
