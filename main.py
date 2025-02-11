import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import classification_report
from src.bayesianNetwork import bNetCreation
from src.songsProlog import query_execute, writeSongsInfo, writeClusterInfo, writeRules
from src.unsupervisedLearning import calcolaCluster
from src.balanceSongs import visualizeAspectRatioChart, overSampling
from src.supervisedLearning import train_model_kfold


def load_data(file_path):
    return pd.read_csv(file_path)


def save_clean_dataset(df, clean_columns, output_path):
    clean_df = df[clean_columns].copy()
    clean_df.to_csv(output_path, index=False)
    print(f"Dataset pulito salvato in: {output_path}")
    return clean_df


def process_clustering(clean_df, clean_dataset_path):
    # Per il clustering utilizziamo solo le colonne numeriche, per evitare errori di conversione
    numeric_for_clustering = clean_df.select_dtypes(
        include=['float64', 'int64'])
    if numeric_for_clustering.empty:
        print("Nessuna colonna numerica disponibile per il clustering.")
        return clean_df
    print("Eseguo il Clustering (KMeans):")
    etichette_cluster, _ = calcolaCluster(numeric_for_clustering)
    # Aggiungi la colonna clusterIndex al DataFrame pulito
    clean_df['clusterIndex'] = etichette_cluster
    clean_df.to_csv(clean_dataset_path, index=False)
    visualizeAspectRatioChart(
        clean_df, "clusterIndex", "Distribuzione Cluster", "aspectRatioChartClusteringKM")
    print("Clustering completato!")
    # Genera fatti relativi ai cluster in Prolog
    writeClusterInfo(clean_df, "kb.pl")
    return clean_df


def process_supervised_learning(clean_df):
    """Gestisce il flusso dell'apprendimento supervisionato con oversampling"""
    if "clusterIndex" in clean_df.columns:
        print("Eseguo l'Apprendimento Supervisionato")
        model_performance = train_model_kfold(clean_df, 'clusterIndex')

        oversampled_data = overSampling(
            clean_df, 'clusterIndex', threshold_ratio=0.7)
        visualizeAspectRatioChart(oversampled_data, 'clusterIndex',
                                  "POST Oversampling", "aspectRatioChartOverSampled")

        print("Eseguo l'Apprendimento Supervisionato dopo OverSampling")
        model_performance = train_model_kfold(oversampled_data, 'clusterIndex')

        print("Apprendimento supervisionato completato!")
    else:
        print("Errore: Il clustering deve essere eseguito prima dell'apprendimento supervisionato.")


def process_bayesian_reasoning(clean_df):
    # Ragionamento probabilistico
    print("Eseguo il ragionamento probabilistico (Bayesiano)")
    # Per il ragionamento discretizziamo solo le colonne numeriche
    numeric_columns = clean_df.select_dtypes(
        include=['float64', 'int64']).columns
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')
    clean_df[numeric_columns] = discretizer.fit_transform(
        clean_df[numeric_columns])
    print("Ragionamento probabilistico completato!")


def generate_knowledge_base(clean_df, df):
    # Genera la KB utilizzando il dataset pulito
    # Genera i fatti relativi alle canzoni
    writeSongsInfo(clean_df, "kb.pl")
    # Aggiunge le regole al file Prolog
    writeRules("kb.pl")
    print("Knowledge Base (kb.pl) generata con successo.")
    query_execute("kb.pl", clean_df, df)


def main():

    dataset_path = "datasetSongs.csv"
    clean_dataset_path = "cleanDataset.csv"

    # I required_columns corrispondono al dataset originale
    required_columns = [
        "track_id", "track_name", "track_artist", "lyrics", "track_popularity",
        "track_album_id", "track_album_name", "track_album_release_date", "playlist_name",
        "playlist_id", "playlist_genre", "playlist_subgenre", "danceability", "energy",
        "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "duration_ms", "language"
    ]

    # Per la KB e le operazioni successive usiamo il dataset pulito.
    # Inseriamo anche "track_id" e "track_name" in modo da avere 18 colonne (corrispondenti al predicato song/18)
    clean_columns = [
        "track_popularity", "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
    ]

    if os.path.exists(dataset_path):
        df = load_data(dataset_path)
        # Controlla che le colonne richieste siano presenti
        if set(required_columns).issubset(df.columns):
            print("Anteprima del dataset integrato:")
            print(df.head())
        else:
            missing_columns = list(set(required_columns) - set(df.columns))
            if missing_columns:
                print(f"Colonne mancanti: {missing_columns}")
                return
            else:
                print("Anteprima del dataset integrato:")
                print(df.head())
                return
    else:
        print(f"Il file '{dataset_path}' non esiste.")
        return

    visualizeAspectRatioChart(
        df, "playlist_genre", "Distribuzione Generi Playlist", "aspectRatioChartGenre")

    clean_df = save_clean_dataset(df, clean_columns, os.path.join(
        os.path.dirname(__file__), "cleanDataset.csv"))

    # Normalizzazione: applichiamo il MinMaxScaler solo alle colonne numeriche
    scaler = MinMaxScaler()
    numeric_columns = clean_df.select_dtypes(
        include=['float64', 'int64']).columns
    if not numeric_columns.empty:
        clean_df[numeric_columns] = scaler.fit_transform(
            clean_df[numeric_columns])
    else:
        print("Nessuna colonna numerica trovata per la normalizzazione.")

    # Processo di Clustering (utilizza solo colonne numeriche per il clustering)
    clean_df = process_clustering(clean_df, clean_dataset_path)

    # Genera la Knowledge Base (kb.pl) usando il dataset pulito
    generate_knowledge_base(clean_df, df)

    # Processo di Apprendimento supervisionato
    process_supervised_learning(clean_df)

    # Genera la rete Bayesiana
    bNetCreation(clean_df)


if __name__ == "__main__":
    main()
