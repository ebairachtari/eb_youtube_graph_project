import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Φορτώνω τον υπογράφο των 5000 κόμβων από το αρχείο
G = nx.read_gexf("../graphs/top_5000_degree.gexf")

# --- Υπολογίζω τα in και out degree για κάθε κόμβο ---
in_degrees = [G.in_degree(n) for n in G.nodes()]
out_degrees = [G.out_degree(n) for n in G.nodes()]

# In-degree histogram 
plt.figure(figsize=(10, 5))
plt.hist(in_degrees, bins=30, color='skyblue')
plt.title("Κατανομή Εισερχόμενων Ακμών (In-degree)")
plt.xlabel("Αριθμός εισερχόμενων ακμών")
plt.ylabel("Αριθμός κόμβων")
plt.grid(True)
plt.tight_layout()
plt.savefig("../diagrams/in_degree_hist.png")
print("Το in-degree histogram αποθηκεύτηκε ως 'in_degree_hist.png'")
plt.show()

# Out-degree histogram 
plt.figure(figsize=(10, 5))
plt.hist(out_degrees, bins=30, color='salmon')
plt.title("Κατανομή Εξερχόμενων Ακμών (Out-degree)")
plt.xlabel("Αριθμός εξερχόμενων ακμών")
plt.ylabel("Αριθμός κόμβων")
plt.grid(True)
plt.tight_layout()
plt.savefig("../diagrams/out_degree_hist.png")
print("Το out-degree histogram αποθηκεύτηκε ως 'out_degree_hist.png'")
plt.show()

# --- Εμφανίζω τα μηδενικά degrees ---
zero_in = sum(1 for d in in_degrees if d == 0)
zero_out = sum(1 for d in out_degrees if d == 0)
print("Κόμβοι με in-degree = 0:", zero_in)
print("Κόμβοι με out-degree = 0:", zero_out)

# --- Log-log διάγραμμα συνολικού degree ---
total_degrees = [G.degree(n) for n in G.nodes()]
degree_counts = Counter(total_degrees)

# Ταξινόμηση (x: degree, y: πλήθος κόμβων με αυτό το degree)
x = sorted(degree_counts.keys())
y = [degree_counts[d] for d in x]

# log-log διάγραμμα 
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='darkblue')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Degree (log scale)")
plt.ylabel("Αριθμός κόμβων (log scale)")
plt.title("Log-Log Degree Distribution")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("../diagrams/loglog_degree_distribution.png")
print("Το log-log διάγραμμα αποθηκεύτηκε ως 'loglog_degree_distribution.png'")
plt.show()