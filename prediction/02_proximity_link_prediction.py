import networkx as nx
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Υπολογισμός proximity-based χαρακτηριστικών και αξιολόγηση ---

# Φόρτωση δεδομένων από το .pkl
data_path = os.path.join("..", "data", "link_dataset.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)

G_train = data["G_train"]
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Μετατροπή σε undirected γράφο για τις proximity μετρικές ---
G_undirected = G_train.to_undirected()

# Ορισμός μετρικών proximity ---
def common_neighbors(u, v):
    return len(list(nx.common_neighbors(G_undirected, u, v)))

def jaccard_coefficient(u, v):
    try:
        union = len(set(G_undirected[u]) | set(G_undirected[v]))
        intersection = len(set(G_undirected[u]) & set(G_undirected[v]))
        return intersection / union if union != 0 else 0
    except:
        return 0

def adamic_adar_index(u, v):
    cn = set(nx.common_neighbors(G_undirected, u, v))
    return sum(1 / np.log(len(G_undirected[n])) for n in cn if len(G_undirected[n]) > 1)

# Υπολογισμός χαρακτηριστικών για κάθε ζεύγος ---
def compute_features(pairs):
    features = []
    for u, v in tqdm(pairs, desc="Υπολογισμός χαρακτηριστικών"):
        cn = common_neighbors(u, v)
        jc = jaccard_coefficient(u, v)
        aa = adamic_adar_index(u, v)
        features.append([cn, jc, aa])
    return np.array(features)

X_train_feat = compute_features(X_train)
X_test_feat = compute_features(X_test)

# --- RANKING & Precision ---

# Χρησιμοποιοώ το Jaccard ως score για ταξινόμηση
test_scores = X_test_feat[:, 1]  # Jaccard index
sorted_indices = np.argsort(test_scores)[::-1]
k = 500  # μπορείς να το αλλάξεις σε 100, 1000 κ.λπ.
top_k_indices = sorted_indices[:k]
top_k_labels = np.array(y_test)[top_k_indices]
precision_at_k = np.sum(top_k_labels) / k

# --- Classification ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Τελική εκτύπωση 
print("\n====== ΑΠΟΤΕΛΕΣΜΑΤΑ ======")
print(f"Precision@{k}: {round(precision_at_k, 4)}")
print(f"Accuracy (classification): {round(accuracy, 4)}")
print(f"ROC AUC: {round(roc_auc, 4)}")

# --- Ενδεικτικός πίνακας με proximity metrics για 10 ζεύγη ---

# Επιλογή 10 τυχαίων δειγμάτων από το test set
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), size=10, replace=False)
sample_pairs = [X_test[i] for i in sample_indices]
sample_labels = [y_test[i] for i in sample_indices]
sample_features = X_test_feat[sample_indices]

# Δημιουργία πίνακα
df_sample = pd.DataFrame(sample_pairs, columns=["Node U", "Node V"])
df_sample["Common Neighbors"] = sample_features[:, 0]
df_sample["Jaccard Coefficient"] = sample_features[:, 1]
df_sample["Adamic-Adar Index"] = sample_features[:, 2]
df_sample["Label"] = sample_labels

print("\n====== ΠΙΝΑΚΑΣ 10 ΔΕΙΓΜΑΤΩΝ ======")
print(df_sample.to_string(index=False))

# --- Διάγραμμα κατανομής Jaccard Coefficient ανά Label ---
df_plot = pd.DataFrame({
    "Jaccard Coefficient": X_test_feat[:, 1],
    "Label": y_test
})

plt.figure(figsize=(8, 5))
sns.boxplot(x="Label", y="Jaccard Coefficient", data=df_plot)
plt.title("Κατανομή Jaccard Coefficient ανά Label (0 = όχι σύνδεση, 1 = σύνδεση)")
plt.xlabel("Label")
plt.ylabel("Jaccard Coefficient")
plt.grid(True)
plt.tight_layout()
plt.savefig("../diagrams/jaccard_boxplot.png")
plt.show()

# --- Confusion Matrix & ROC Curve για Logistic Regression ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Link", "Link"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.grid(False)
plt.tight_layout()
plt.savefig("../diagrams/confusion_matrix.png")
plt.show()

# ROC Curve
roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve - Logistic Regression")
plt.grid(True)
plt.tight_layout()
plt.savefig("../diagrams/roc_curve.png")  
plt.show()
