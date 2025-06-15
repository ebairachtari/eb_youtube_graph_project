import networkx as nx
import random
import pickle
import os
from sklearn.model_selection import train_test_split

# --- Δημιουργώ dataset με θετικά και αρνητικά παραδείγματα για link prediction ---

# Διαδρομή αρχείου γράφου (.gexf)
# Φορτώνω τον υπογράφο των 5000 κόμβων από το αρχείο
G = nx.read_gexf("../graphs/top_5000_degree.gexf")

# Βεβαιώνομαι ότι είναι directed γράφος
G = nx.DiGraph(G)

# Θετικά παραδείγματα (positive samples) 
# Παίρνω όλες τις ακμές του γράφου
positive_edges = list(G.edges())
positive_labels = [1] * len(positive_edges)

# 4. Αρνητικά παραδείγματα (negative samples)
# Παράγω τυχαία ζεύγη κόμβων χωρίς σύνδεση (ούτε προς τα μπρος, ούτε προς τα πίσω)
nodes = list(G.nodes())
negative_edges = set()
while len(negative_edges) < len(positive_edges):
    u, v = random.sample(nodes, 2)
    if not G.has_edge(u, v) and not G.has_edge(v, u):
        negative_edges.add((u, v))
negative_edges = list(negative_edges)
negative_labels = [0] * len(negative_edges)

# Συνένωση θετικών και αρνητικών παραδειγμάτων
all_edges = positive_edges + negative_edges
all_labels = positive_labels + negative_labels

# Διαχωρισμός σε training και test set
X_train, X_test, y_train, y_test = train_test_split(
    all_edges, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# Αφαίρεση test-positive ακμών από τον γράφο για να αποφύγουμε leakage
test_positive_edges = [edge for edge, label in zip(X_test, y_test) if label == 1]
G_train = G.copy()
G_train.remove_edges_from(test_positive_edges)

# Αποθήκευση όλων σε αρχείο .pkl
output = {
    "G_train": G_train,
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test
}
output_path = os.path.join("..", "data", "link_dataset.pkl")
with open(output_path, "wb") as f:
    pickle.dump(output, f)

print(f"Το dataset αποθηκεύτηκε στο: {output_path}")
print(f"Training examples: {len(X_train)}")
print(f"Test examples:     {len(X_test)}")
print(f"Test positive edges που αφαιρέθηκαν από τον γράφο: {len(test_positive_edges)}")
