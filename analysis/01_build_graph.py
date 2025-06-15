import os
import networkx as nx
import matplotlib.pyplot as plt

# --- Δημιουργία directed γράφου από τα δεδομένα των YouTube videos ---

# Ορίζω το path των αρχείων δεδομένων
data_folder = os.path.join("..", "data") 

# Φτιάχνω τον directed γράφο μου
G = nx.DiGraph()  # Directed γράφος, δηλαδή με κατευθυνόμενες ακμές

# Διαβάζω τα αρχεία ένα-ένα
for filename in os.listdir(data_folder):
    if filename.endswith('.txt') and filename != 'log.txt':  # Αγνοώ το log
        file_path = os.path.join(data_folder, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')  # Κάθε πεδίο χωρίζεται με tab
                
                # Αν η γραμμή δεν έχει τουλάχιστον 10 πεδία, την αγνοώ
                if len(parts) < 10:
                    continue

                video_id = parts[0]
                related_ids = parts[9:]  # Τα υπόλοιπα είναι τα related videos

                # Προσθέτω τον κόμβο (αν δεν υπάρχει ήδη)
                G.add_node(video_id)

                # Προσθέτω τις directed ακμές από το video προς κάθε related
                for related_id in related_ids:
                    G.add_edge(video_id, related_id)

# Στατιστικά γράφου
print("Αριθμός κόμβων (videos):", G.number_of_nodes())
print("Αριθμός ακμών (συνδέσεις related):", G.number_of_edges())

# Δείχνω 3 τυχαίους κόμβους με τις ακμές τους
print("\nΔείγμα κόμβων και related videos:")
for i, node in enumerate(G.nodes()):
    if i == 3:
        break
    print(f"Video ID: {node}")
    print("Related:", list(G.successors(node)))
    print()


# --- Υπογράφος με τους top 100 κόμβους με το μεγαλύτερο degree ---

# Υπολογίζω το συνολικό degree (in + out) για κάθε κόμβο
top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:100]
top_node_ids = [node for node, degree in top_nodes]

# Φτιάχνω τον υπογράφο
top_subgraph = G.subgraph(top_node_ids)

# Αποθηκεύω τον υπογράφο για Gephi
nx.write_gexf(top_subgraph, "../graphs/top_100_degree.gexf")
print("Ο υπογράφος με τους top 100 κόμβους αποθηκεύτηκε ως 'top_100_degree.gexf'")

# --- Υπογράφος με τους top 5000 κόμβους με το μεγαλύτερο degree ---

# Υπολογίζω το συνολικό degree (in + out) για κάθε κόμβο
top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5000]
top_node_ids = [node for node, degree in top_nodes]

# Φτιάχνω τον υπογράφο
top_subgraph = G.subgraph(top_node_ids)

# Αποθηκεύω τον υπογράφο για Gephi
nx.write_gexf(top_subgraph, "../graphs/top_5000_degree.gexf")
print("Ο υπογράφος με τους top 5000 κόμβους αποθηκεύτηκε ως 'top_5000_degree.gexf'")