# YouTube Graph Analysis & Link Prediction

## Περιγραφή
Αυτό το project περιλαμβάνει την ανάλυση γράφου του YouTube και την πρόβλεψη ακμών με δύο διαφορετικές τεχνικές: proximity-based χαρακτηριστικά και Graph Neural Networks (GNN).
Τα δεδομένα που χρησιμοποίησα βρίσκονται στο αρχείο 0222.zip από το [Youtube Dataset](https://netsg.cs.sfu.ca/youtubedata) .

## Δομή Φακέλων

- `analysis/`: Ανάλυση του γράφου (degree, centrality, shortest paths)
- `prediction/`: Δημιουργία dataset πρόβλεψης και εφαρμογή δύο μοντέλων:
   - Link Prediction μέσω Proximity Metrics
   - Graph Neural Networks (GNNs)

## Εκτέλεση

1. **Βήμα 1:** Δημιουργία γράφου  
   ```bash
   python analysis/01_build_graph.py
   ```

2. **Βήμα 2:** Εξερεύνηση χαρακτηριστικών του γράφου  
   - Degree: `02_degree_analysis.py`
   - Centrality: `03_centrality_analysis.py`
   - Shortest Paths: `04_shortest_paths_analysis.py`

3. **Βήμα 3:** Δημιουργία dataset πρόβλεψης ακμών  
   ```bash
   python prediction/01_prepare_link_dataset.py
   ```

4. **Βήμα 4:** Πειραματισμός με τεχνικές πρόβλεψης  
   - Link Prediction μέσω Proximity Metrics:  
     ```bash
     python prediction/02_proximity_link_prediction.py
     ```
   - Με Graph Neural Network (GNN):  
     ```bash
     python prediction/03_gnn_link_prediction.py
     ```

## Απαιτούμενες Βιβλιοθήκες

Εγκατάσταση με pip:

```bash
pip install -r requirements.txt
```

## Output

- Αποτελέσματα (confusion matrix, ROC curves, κλπ.) αποθηκεύονται στον φάκελο `diagrams/`.
- Το dataset link prediction αποθηκεύεται ως pickle (`link_dataset.pkl`) στον φάκελο `data/`.
- Οι παραγόμενοι γράφοι (σε μορφή .gexf) αποθηκεύονται στον φάκελο `graphs/`.

>*Αναπτύχθηκε αποκλειστικά για εκπαιδευτικούς σκοπούς.*
