import networkx as nx
import matplotlib.pyplot as plt

# Φορτώνω τον υπογράφο των 100 κόμβων από το αρχείο
G = nx.read_gexf("../graphs/top_100_degree.gexf")

# Υπολογισμός Degree Centrality
degree_centrality = nx.degree_centrality(G)

# Υπολογισμός Closeness Centrality
closeness_centrality = nx.closeness_centrality(G)

# Υπολογισμός Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G)

# Εμπλουτισμός κόμβων με τις τιμές centrality
for node in G.nodes():
    G.nodes[node]['degree_centrality'] = degree_centrality[node]
    G.nodes[node]['closeness_centrality'] = closeness_centrality[node]
    G.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]

    # Επιπλέον, χρησιμοποιώ το degree_centrality για να χρωματίσω
    # όσο πιο κοντά στο 1, τόσο πιο έντονο χρώμα
    G.nodes[node]['viz'] = {
        'color': {
            'r': int(degree_centrality[node] * 255),
            'g': 0,
            'b': int((1 - degree_centrality[node]) * 255),
            'a': 1.0
        }
    }

# Αποθήκευση εμπλουτισμένου γράφου με 100 κόμβους
nx.write_gexf(G, "../graphs/top_100_with_centrality.gexf")
print("Ο γράφος εμπλουτίστηκε με centrality scores και χρώμα, και αποθηκεύτηκε ως 'top_100_with_centrality.gexf'")

# Ταξινόμηση και εκτύπωση Top-5 για κάθε centrality

print("Top 5 Degree Centrality:")
for node, score in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{node}: {score:.4f}")

print("\nTop 5 Closeness Centrality:")
for node, score in sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{node}: {score:.4f}")

print("\nTop 5 Betweenness Centrality:")
for node, score in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{node}: {score:.4f}")