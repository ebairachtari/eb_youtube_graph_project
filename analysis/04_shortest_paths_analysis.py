import networkx as nx
import matplotlib.pyplot as plt

# Φορτώνω τον υπογράφο των 5000 κόμβων από το αρχείο
G = nx.read_gexf("../graphs/top_5000_degree.gexf")

# Υπολογισμός μήκους συντομότερων μονοπατιών για κάθε κόμβο
path_lengths = []

for source in G.nodes():
    lengths = nx.single_source_shortest_path_length(G, source) # Υπολογισμός μήκους μονοπατιών από τον κόμβο source
    path_lengths.extend(lengths.values()) # Αποθηκεύω τα μήκη των μονοπατιών

# Αφαίρεση μηδενικών μήκους (αν υπάρχουν)
path_lengths = [length for length in path_lengths if length > 0]

# Δημιουργία histogram για την κατανομή των μήκους των συντομότερων μονοπατιών
plt.figure(figsize=(10, 5))
plt.hist(path_lengths, bins=range(1, max(path_lengths)+2), color='lightseagreen', edgecolor='black')
plt.title("Κατανομή Μήκους Συντομότερων Μονοπατιών")
plt.xlabel("Μήκος μονοπατιού")
plt.ylabel("Αριθμός διαδρομών")
plt.grid(True)
plt.tight_layout()
plt.savefig("../diagrams/shortest_paths_distribution.png")
print("Η κατανομή αποστάσεων αποθηκεύτηκε ως 'shortest_paths_distribution.png'")
plt.show()

# Εκτύπωση στατιστικών
print(f"\nΜέγιστο μήκος μονοπατιού: {max(path_lengths)}")
print(f"Μέσο μήκος μονοπατιού: {sum(path_lengths)/len(path_lengths):.2f}")
print(f"Συνολικά paths υπολογίστηκαν: {len(path_lengths)}")