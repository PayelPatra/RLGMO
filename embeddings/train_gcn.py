# embeddings/train_gcn.py
from __future__ import annotations

from pathlib import Path
import pickle
import ast
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GRAPH_FILE = ROOT / "graph" / "knowledge_graph.pkl"
EMBED_DIR = ROOT / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

NODE_MAP_FILE = EMBED_DIR / "node_mapping.pkl"
OUT_EMB = EMBED_DIR / "gnn_embeddings.pt"


def to_list_safe(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return [t.strip() for t in re.split(r"[;,]\s*", s) if t.strip()]


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def main():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Missing: {GRAPH_FILE}")

    # Load graph
    with open(GRAPH_FILE, "rb") as f:
        G: nx.Graph = pickle.load(f)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    # Build patient feature vectors like Colab: [age, gender, stage]
    train_df = pd.read_csv(DATA_DIR / "extracted_annotations_cleaned.csv", dtype=str)
    test_df = pd.read_csv(DATA_DIR / "test_cleaned.csv", dtype=str)
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    df_all.columns = df_all.columns.str.strip()

    if "RLGMO_ID" not in df_all.columns:
        raise KeyError("RLGMO_ID missing in cleaned files")

    df_all["Age"] = pd.to_numeric(df_all.get("Age", 0), errors="coerce").fillna(0).astype(float)
    df_all["Gender"] = pd.to_numeric(df_all.get("Gender", -1), errors="coerce").fillna(-1).astype(float)
    df_all["Cancer stage"] = pd.to_numeric(df_all.get("Cancer stage", -1), errors="coerce").fillna(-1).astype(float)

    features = torch.zeros((len(G.nodes()), 3), dtype=torch.float32)

    # Fill features only for patient nodes that exist in df_all
    meta = df_all.set_index("RLGMO_ID")[["Age", "Gender", "Cancer stage"]].to_dict("index")

    for node, idx in node_mapping.items():
        if str(node) in meta:
            a = float(meta[str(node)]["Age"])
            g = float(meta[str(node)]["Gender"])
            s = float(meta[str(node)]["Cancer stage"])
            features[idx] = torch.tensor([a, g, s], dtype=torch.float32)

    # Normalize age and stage like Colab
    if features[:, 0].max() > 0:
        features[:, 0] = features[:, 0] / features[:, 0].max()
    if features[:, 2].max() > 0:
        features[:, 2] = features[:, 2] / features[:, 2].max()

    # edges
    edges = []
    for u, v in G.edges():
        edges.append([node_mapping[u], node_mapping[v]])
        edges.append([node_mapping[v], node_mapping[u]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=features, edge_index=edge_index)

    # Colab settings
    model = GCNEncoder(in_channels=3, hidden_channels=16, out_channels=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(2000):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    with open(NODE_MAP_FILE, "wb") as f:
        pickle.dump(node_mapping, f)
    torch.save(embeddings, OUT_EMB)

    print(f"Saved node mapping: {NODE_MAP_FILE}")
    print(f"Saved embeddings: {OUT_EMB} shape={tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
