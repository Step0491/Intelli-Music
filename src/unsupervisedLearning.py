import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import joblib
from joblib import Parallel, delayed
import numpy as np


def calcola_inertia(k, dataSet):
    """Calcola l'inertia per un dato numero di cluster k."""
    try:
        kmeans = KMeans(n_clusters=k, n_init=10,
                        init='k-means++', random_state=42)
        kmeans.fit(dataSet)
        return kmeans.inertia_
    except Exception as e:
        print(f"Errore durante il calcolo dell'inertia per k={k}: {e}")
        return None


def regolaGomito(dataSet, maxK=15):
    """Calcola il numero ottimale di cluster K usando il metodo del gomito."""
    inertia = Parallel(n_jobs=-1)(delayed(calcola_inertia)
                                  (k, dataSet) for k in range(4, maxK + 1))

    inertia = [i for i in inertia if i is not None]  # Filtra valori None

    kl = KneeLocator(range(4, len(inertia) + 4), inertia,
                     curve="convex", direction="decreasing")

    plt.figure(figsize=(8, 5))
    plt.plot(range(4, len(inertia) + 4), inertia,
             'bo-', markersize=5, label='Inertia')
    plt.scatter(kl.elbow, inertia[kl.elbow - 4], c='red',
                marker='o', s=100, label=f'Elbow at k={kl.elbow}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Metodo del Gomito')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join("charts", "regolaGomito.png")
    os.makedirs("charts", exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    return kl.elbow


def plot_silhouette_scores(silhouette_scores):
    """Visualizza il grafico del Silhouette Score per i diversi valori di K testati."""
    plt.figure(figsize=(8, 5))

    k_values = [score[0] for score in silhouette_scores]
    scores = [score[1] for score in silhouette_scores]

    best_k, best_score = max(silhouette_scores, key=lambda x: x[1])

    plt.plot(k_values, scores, 'bo-', markersize=5, label="Silhouette Score")
    plt.scatter(best_k, best_score, c='red', marker='o',
                s=100, label=f'Best k={best_k}')
    plt.xlabel("Numero di Cluster (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Analisi del Silhouette Score")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join("charts", "silhouette_scores.png")
    os.makedirs("charts", exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    plt.close()


def calcolaCluster(dataSet, maxK=15):
    """Esegue il clustering K-Means con il numero ottimale di cluster e verifica silhouette score."""
    k = regolaGomito(dataSet, maxK)

    # Proviamo silhouette score per confermare il numero ottimale di cluster
    silhouette_scores = []
    possible_K = list(range(max(4, k - 2), min(maxK, k + 3))
                      )  # Proviamo ±2 intorno a K
    for test_k in possible_K:
        km = KMeans(n_clusters=test_k, n_init=10,
                    init='k-means++', random_state=42)
        labels = km.fit_predict(dataSet)
        score = silhouette_score(dataSet, labels)
        silhouette_scores.append((test_k, score))

    # Mostriamo il grafico dei silhouette score
    plot_silhouette_scores(silhouette_scores)

    # Trova il valore di k con il miglior silhouette score
    best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
    print(f"Silhouette Score migliore: {best_score:.3f} con K={best_k}")

    # Ora eseguiamo il clustering con il miglior K
    km = KMeans(n_clusters=best_k, n_init=10,
                init='k-means++', random_state=42)
    km.fit(dataSet)

    joblib.dump(km, f'kmeans_model_{best_k}.joblib')

    return km.labels_, km.cluster_centers_
