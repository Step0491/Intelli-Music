import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import joblib
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, K2Score, BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from joblib import Parallel, delayed


def bNetCreation(dataset):
    """
    Creazione della rete bayesiana con ricerca della struttura ottimale
    """
    required_columns = [
        "track_popularity", "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "clusterIndex"
    ]

    # Verifica che tutte le colonne necessarie siano presenti
    missing_cols = [
        col for col in required_columns if col not in dataset.columns]
    if missing_cols:
        raise ValueError(
            f"Le seguenti colonne sono mancanti nel dataset: {missing_cols}")

    # Seleziona una frazione del dataset (0.05%)
    dataset = dataset.sample(frac=0.0006, random_state=42)

    # Discretizzazione dei dati numerici
    discretizer = KBinsDiscretizer(
        n_bins=4, encode='ordinal', strategy='uniform')
    numeric_columns = dataset.select_dtypes(
        include=['float64', 'int64']).columns
    dataset[numeric_columns] = discretizer.fit_transform(
        dataset[numeric_columns])
    dataset[numeric_columns] = dataset[numeric_columns].astype(
        int)  # Conversione in interi

    # Controllo eventuali NaN
    if dataset.isna().sum().sum() > 0:
        raise ValueError("Ci sono valori NaN dopo la discretizzazione!")

    # Ricerca della struttura ottimale con Hill Climbing e K2 Score
    hc_k2 = HillClimbSearch(dataset)
    k2_model = hc_k2.estimate(scoring_method=K2Score(dataset))

    # Creazione della rete bayesiana
    model = BayesianNetwork(k2_model.edges())

    # Fitting del modello con BayesianEstimator
    equivalent_sample_size = 10
    if equivalent_sample_size <= 0:
        raise ValueError(
            "equivalent_sample_size deve essere maggiore di zero!")
    model.fit(dataset, estimator=BayesianEstimator, prior_type='BDeu',
              equivalent_sample_size=equivalent_sample_size)

    # Salvataggio della rete bayesiana
    joblib.dump(model, 'bayesian_network.joblib', compress=True)
    plotBayesianNetwork(model)
    print("Rete Bayesiana creata e salvata con successo!")
    return model


def loadBayesianNetwork():
    """
    Carica la rete bayesiana salvata
    """
    return joblib.load('bayesian_network.joblib')


def inferenzaBayesiana(model, evidence):
    """
    Esegue l'inferenza bayesiana dati i valori di evidenza
    """
    infer = VariableElimination(model)
    return infer.map_query(variables=["track_popularity"], evidence=evidence)


def plotBayesianNetwork(model):
    """
    Visualizza e salva la struttura della rete bayesiana.
    """
    plt.figure(figsize=(14, 10))  # Increase figure size for better visibility
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    # Using a shell layout for better node ordering (you can customize the shells)
    pos = nx.shell_layout(G)

    # Adding padding and adjusting layout to prevent overlapping
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9,
                        bottom=0.1)  # Adding padding to the plot

    # Draw the graph with more customized styling
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue',
            font_size=12, font_weight='bold', edge_color='gray', width=2, alpha=0.7)

    plt.title("Struttura della Rete Bayesiana", fontsize=16)

    # Salvataggio immagine
    save_path = os.path.join("charts", "bayesian_network.png")
    os.makedirs("charts", exist_ok=True)
    plt.savefig(save_path, format="png", dpi=300)

    # Mostra il grafico
    plt.show()
    plt.close()


def main():
    bNetCreation(pd.read_csv("cleanDataset.csv"))


if __name__ == "__main__":
    main()
