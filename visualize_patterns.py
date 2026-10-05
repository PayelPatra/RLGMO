from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMBED_DIR = ROOT / "embeddings"

PATTERN_FILE = EMBED_DIR / "gcn_based_rlgmo_patterns.csv"
TRAIN_FILE = DATA_DIR / "extracted_annotations_cleaned.csv"
OUT_PNG = EMBED_DIR / "gcn_rlgmo_pattern_graph.png"


def to_list_safe(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        return [t.strip() for t in s.split(",") if t.strip()]


def main():
    if not PATTERN_FILE.exists():
        raise FileNotFoundError(f"Missing: {PATTERN_FILE}. Run embeddings/pattern_similarity.py first.")
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_FILE}")

    pattern_df = pd.read_csv(PATTERN_FILE, dtype=str)
    train_df = pd.read_csv(TRAIN_FILE, dtype=str)
    train_df["RLGMO_ID"] = train_df["RLGMO_ID"].astype(str)

    if "Symptoms" in train_df.columns:
        train_df["Symptoms"] = train_df["Symptoms"].apply(to_list_safe)

    subset = pattern_df.head(15)

    G = nx.DiGraph()

    pid_to_symptoms = {}
    if "Symptoms" in train_df.columns:
        pid_to_symptoms = dict(zip(train_df["RLGMO_ID"], train_df["Symptoms"]))

    for _, row in subset.iterrows():
        rid = str(row.get("RLGMO_ID", ""))
        disease = str(row.get("Top_Disease", "")).strip().lower()
        med = str(row.get("Top_Medication", "")).strip().lower()

        if not rid:
            continue

        G.add_node(rid, kind="patient")
        if disease:
            G.add_node(disease, kind="disease")
            G.add_edge(rid, disease)

        for s in pid_to_symptoms.get(rid, [])[:3]:
            sym = str(s).strip().lower()
            if sym:
                G.add_node(sym, kind="symptom")
                if disease:
                    G.add_edge(disease, sym)
                else:
                    G.add_edge(rid, sym)

        if med:
            G.add_node(med, kind="medication")
            if disease:
                G.add_edge(disease, med)
            else:
                G.add_edge(rid, med)

    pos = nx.spring_layout(G, seed=42, k=0.8)

    colors = []
    for n in G.nodes():
        kind = G.nodes[n].get("kind")
        if kind == "patient":
            colors.append("lightblue")
        elif kind == "disease":
            colors.append("lightgreen")
        elif kind == "symptom":
            colors.append("khaki")
        elif kind == "medication":
            colors.append("salmon")
        else:
            colors.append("gray")

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6, width=1.2, connectionstyle="arc3,rad=0.07")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2200, edgecolors="black", linewidths=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    plt.title("GCN-Based RLGMO_ID → Disease → Symptom → Medication Patterns", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()

    print(f" Saved graph: {OUT_PNG}")


if __name__ == "__main__":
    main()
