from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RL_DIR = ROOT / "rl"
INPUT_FILE = RL_DIR / "rl_test_with_eval.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Result file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

if "selected_medication" not in df.columns:
    raise ValueError("Column 'selected_medication' not found in CSV")

df = df.dropna(subset=["selected_medication"])
med_counts = df["selected_medication"].value_counts()

if med_counts.empty:
    raise ValueError("No medication recommendations found to plot.")

plt.figure(figsize=(10, 6))
colors = sns.color_palette("Dark2", n_colors=len(med_counts))
med_counts.plot(kind="barh", color=colors)

plt.xlabel("Number of Recommendations")
plt.ylabel("Medication")
plt.title("Top Recommended Medications by RL Model")
plt.gca().invert_yaxis()
plt.gca().spines[["top", "right"]].set_visible(False)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
