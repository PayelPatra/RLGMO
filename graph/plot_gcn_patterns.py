from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_PNG = ROOT / "graph" / "gcn_rlgmo_pattern_graph.png"

pattern_df = pd.read_csv(DATA_DIR / "gcn_based_rlgmo_patterns.csv", dtype=str)
train_df = pd.read_csv(DATA_DIR / "extracted_annotations_cleaned.csv", dtype=str)

# Your IDs are likely RLGMO_ID (XYZ###)
if "RLGMO_ID" not in train_df.columns:
    raise KeyError(f"train_df missing RLGMO_ID. Columns: {list(train_df.columns)}")
train_df["RLGMO_ID"] = train_df["RLGMO_ID"].astype(str)

G = nx.DiGraph()
subset = pattern_df.head(15)

def parse_symptoms(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
        return [t.strip().lower() for t in s.split(",") if t.strip()]
    return []

for _, row in subset.iterrows():
    rid = str(row.get("RLGMO_ID", "")).strip()
    disease = (row.get("Top_Disease", "") or "unknown_disease").strip()
    med = (row.get("Top_Medication", "") or "unknown_med").strip()

    meta = train_df[train_df["RLGMO_ID"] == rid]
    if not meta.empty:
        symptoms = parse_symptoms(meta.iloc[0].get("Symptoms", ""))
    else:
        symptoms = []

    symptoms = [s.strip("[](){}'\" ").lower() for s in symptoms if isinstance(s, str) and s.strip()]

    r_label = rid if rid else "unknown_rlgmo"
    G.add_node(r_label, type="rlgmo")
    G.add_node(disease, type="disease")
    G.add_node(med, type="medication")
    G.add_edge(r_label, disease)

    if symptoms:
        top_symptom = symptoms[0]
        G.add_node(top_symptom, type="symptom")
        G.add_edge(disease, top_symptom)
        G.add_edge(top_symptom, med)
    else:
        G.add_edge(disease, med)

pos = nx.spring_layout(G, seed=42, k=1.2, iterations=100)

colors = []
for n, data in G.nodes(data=True):
    t = data.get("type", "")
    if t == "rlgmo":
        colors.append("skyblue")
    elif t == "disease":
        colors.append("lightgreen")
    elif t == "medication":
        colors.append("lightcoral")
    elif t == "symptom":
        colors.append("gold")
    else:
        colors.append("gray")

plt.figure(figsize=(14, 10))
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6, width=1.2, connectionstyle="arc3,rad=0.07")
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2200, edgecolors="black", linewidths=0.6)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

plt.title("GCN-Based RLGMO_ID → Disease/Symptom → Medication Patterns", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.show()

print(f"Saved: {OUT_PNG}")
