from pathlib import Path
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
EMBED_DIR = ROOT / "embeddings"
OUT_PNG = EMBED_DIR / "patient_embeddings_2d.png"


def load_embeddings_and_map():
    emb_candidates = [EMBED_DIR / "gnn_embeddings.pt", EMBED_DIR / "patient_gnn_embeddings.pt"]
    map_candidates = [EMBED_DIR / "node_mapping.pkl", EMBED_DIR / "patient_node_map.pkl"]

    emb_path = next((p for p in emb_candidates if p.exists()), None)
    map_path = next((p for p in map_candidates if p.exists()), None)

    if emb_path is None:
        raise FileNotFoundError(f"Missing embeddings file. Tried: {emb_candidates}")
    if map_path is None:
        raise FileNotFoundError(f"Missing node map file. Tried: {map_candidates}")

    gnn_embeddings = torch.load(emb_path, map_location="cpu")
    with open(map_path, "rb") as f:
        node_mapping = pickle.load(f)

    return gnn_embeddings, node_mapping


def main():
    gnn_embeddings, node_mapping = load_embeddings_and_map()
    rev_node_mapping = {v: k for k, v in node_mapping.items()}

    gnn = gnn_embeddings.detach().cpu().float()

    patient_vectors, patient_ids = [], []
    for idx, node_name in rev_node_mapping.items():
        s = str(node_name)
        if s.startswith("XYZ"):
            patient_ids.append(s)
            patient_vectors.append(gnn[idx].numpy())

    if not patient_vectors:
        raise ValueError("No patient nodes found to visualize (expected IDs like XYZ001...)")

    X = np.vstack(patient_vectors)
    n = X.shape[0]

    if n < 3:
        reduced = PCA(n_components=2).fit_transform(X)
        title = "2D Projection of Patient Embeddings (PCA)"
    else:
        perplexity = max(2, min(30, n - 1))
        reduced = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
        ).fit_transform(X)
        title = "2D Projection of Patient Embeddings (t-SNE)"

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], edgecolors="k")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()

    print(f" Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
