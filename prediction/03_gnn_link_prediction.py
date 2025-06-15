import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Φόρτωση δεδομένων από το .pkl 
with open("../data/link_dataset.pkl", "rb") as f:
    data_dict = pickle.load(f)

G_train = data_dict["G_train"]
X_train = data_dict["X_train"]
y_train = data_dict["y_train"]
X_test = data_dict["X_test"]
y_test = data_dict["y_test"]

# Mapping κόμβων και μετατροπή σε PyTorch Geometric
node_mapping = {node: i for i, node in enumerate(G_train.nodes())}
G_nx = nx.relabel_nodes(G_train, node_mapping)
pyg_graph = from_networkx(G_nx)

num_nodes = pyg_graph.num_nodes
feature_dim = 64  # Τυχαία χαρακτηριστικά με 64 διαστάσεις ανά κόμβο
# === Αντί για τυχαία χαρακτηριστικά, χρησιμοποιούμε το degree κάθε κόμβου ως feature ===
degree = np.array([G_nx.degree(n) for n in G_nx.nodes()])
degree = (degree - degree.mean()) / degree.std()  # Κανονικοποίηση (standardization)
pyg_graph.x = torch.tensor(degree, dtype=torch.float32).unsqueeze(1)  # Μετατροπή σε [N, 1] tenso
feature_dim = 1  # Αφού έχουμε μόνο 1 χαρακτηριστικό ανά κόμβο


# GCN Μοντέλο
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, edge_pairs):
        z_u = z[edge_pairs[0]]
        z_v = z[edge_pairs[1]]
        return (z_u * z_v).sum(dim=1)  # dot product

# Συνάρτηση μετατροπής edge list σε tensor με ασφαλές mapping
def edge_list_to_tensor(edge_list):
    mapped = []
    for u, v in edge_list:
        if u in node_mapping and v in node_mapping:
            mapped.append((node_mapping[u], node_mapping[v]))
    if not mapped:
        raise ValueError("⚠️ Δεν βρέθηκαν έγκυρα edges στο edge_list_to_tensor!")
    edge_index = torch.tensor(mapped, dtype=torch.long).t().contiguous()
    return edge_index

# Εκτυπώσεις ελέγχου
print("Ακμές στον G_train:", G_train.number_of_edges())
print("Ακμές στον G_nx:", G_nx.number_of_edges())

train_edge_index = torch.tensor(list(G_nx.edges()), dtype=torch.long).t().contiguous()
print("Μέγεθος train_edge_index:", train_edge_index.size())

# Αρχικοποίηση μοντέλου και optimizer 
model = GCNLinkPredictor(in_channels=feature_dim, hidden_channels=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss()

# Εκπαίδευση GNN 
model.train()
for epoch in range(1, 301):
    optimizer.zero_grad()
    z = model.encode(pyg_graph.x, train_edge_index)
    edge_index_train = edge_list_to_tensor(X_train)
    labels = torch.tensor(y_train, dtype=torch.float)
    preds = model.decode(z, edge_index_train)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Αξιολόγηση GNN
model.eval()
with torch.no_grad():
    z = model.encode(pyg_graph.x, train_edge_index)
    edge_index_test = edge_list_to_tensor(X_test)
    logits = model.decode(z, edge_index_test)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_pred = (probs > 0.5).astype(int)

    print("\n===== ΑΠΟΤΕΛΕΣΜΑΤΑ =====")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, probs):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

# --- 5. Confusion Matrix & ROC Curve  ---
# --- Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Link", "Link"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - GNN Link Prediction")
plt.grid(False)
plt.tight_layout()
plt.savefig("../diagrams/confusion_matrix_gnn.png")
plt.show()

# ROC Curve
RocCurveDisplay.from_predictions(y_test, probs)
plt.title("ROC Curve - GNN Link Prediction")
plt.grid(True)
plt.tight_layout()
plt.savefig("../diagrams/roc_curve_gnn.png")
plt.show()

# --- Διάγραμμα κατανομής πιθανοτήτων πρόβλεψης ανά Label ---
df = pd.DataFrame({
    "Probability": probs,
    "True Label": y_test
})

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Probability", hue="True Label", bins=30, stat="density", common_norm=False, palette="pastel")
plt.title("Κατανομή πιθανοτήτων πρόβλεψης ανά Label (GNN)")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.savefig("../diagrams/probability_distribution_gnn.png")
plt.show()