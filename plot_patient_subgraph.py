from pathlib import Path
import random
import networkx as nx
import matplotlib.pyplot as plt
import pickle

ROOT = Path(__file__).resolve().parents[1]
GRAPH_FILE = ROOT / "graph" / "knowledge_graph.pkl"
OUT_PNG = ROOT / "graph" / "patient_subgraph.png"


def layered_pos(patient, symptoms, diseases, meds, stages):
    """Simple layered layout: symptom -> disease -> medication, patient in center."""
    pos = {}
    pos[patient] = (0.0, 0.0)

    # spread nodes vertically in each layer
    def place(layer_nodes, x):
        if not layer_nodes:
            return
        step = 1.0 / max(1, len(layer_nodes))
        start = -0.5
        for i, n in enumerate(layer_nodes):
            pos[n] = (x, start + i * step)

    place(symptoms, -1.8)
    place(diseases, -0.6)
    place(stages, 0.6)
    place(meds, 1.8)
    return pos


def main():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Missing: {GRAPH_FILE}")

    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)

    patient_nodes = [
        n for n, d in G.nodes(data=True)
        if str(n).startswith("XYZ") or d.get("type") == "patient"
    ]
    if not patient_nodes:
        raise ValueError("No patient nodes found in the graph.")

    patient = random.choice(patient_nodes)

    # --- 1-hop neighbors only (much cleaner) ---
    neighbors = list(G.neighbors(patient))

    # categorize by node type
    symptoms, diseases, meds, stages, other = [], [], [], [], []
    for n in neighbors:
        t = G.nodes[n].get("type", "")
        if t == "symptom":
            symptoms.append(n)
        elif t == "disease":
            diseases.append(n)
        elif t == "medication":
            meds.append(n)
        elif t == "stage":
            stages.append(n)
        else:
            other.append(n)

    # limit counts to avoid clutter (tune these)
    MAX_SYM = 8
    MAX_DIS = 4
    MAX_MED = 6

    symptoms = symptoms[:MAX_SYM]
    diseases = diseases[:MAX_DIS]
    meds = meds[:MAX_MED]

    # build a small subgraph: patient + selected neighbors only
    keep_nodes = [patient] + symptoms + diseases + meds + stages
    SG = G.subgraph(keep_nodes).copy()

    # layered layout (readable)
    pos = layered_pos(patient, symptoms, diseases, meds, stages)

    # colors
    colors = []
    for n in SG.nodes():
        t = SG.nodes[n].get("type", "")
        if n == patient:
            colors.append("deepskyblue")
        elif t == "symptom":
            colors.append("khaki")
        elif t == "disease":
            colors.append("lightgreen")
        elif t == "medication":
            colors.append("salmon")
        elif t == "stage":
            colors.append("plum")
        else:
            colors.append("lightgray")

    plt.figure(figsize=(14, 8))
    nx.draw_networkx_edges(SG, pos, alpha=0.5, width=1.3)
    nx.draw_networkx_nodes(SG, pos, node_color=colors, node_size=1600, edgecolors="black", linewidths=0.6)
    nx.draw_networkx_labels(SG, pos, font_size=8)

    plt.title(f"Patient 1-hop Subgraph (filtered) for {patient}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()

    print(f"Saved subgraph plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
