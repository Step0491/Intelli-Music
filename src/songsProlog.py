import os
import random
import pandas as pd
from pyswip import Prolog
from concurrent.futures import ThreadPoolExecutor
import threading

lock = threading.Lock()


def write_fact_to_file(fact, facts_set):
    with lock:
        if fact not in facts_set:
            facts_set.add(fact)


def save_facts_to_file(file_path, facts_set):
    with open(file_path, 'a', encoding='utf-8') as file:
        for fact in sorted(facts_set):
            file.write(fact + "\n")


def process_song(song, facts_set):
    try:
        cluster = song.get('clusterIndex', -1)
        fact = (f"song({song['track_popularity']}, {song['danceability']}, {song['energy']}, {song['key']},"
                f" {song['loudness']}, {song['mode']}, {song['speechiness']}, {song['acousticness']},"
                f" {song['instrumentalness']}, {song['liveness']}, {song['valence']}, {song['tempo']},"
                f" {song['duration_ms']}).")

        if fact.count('(') == fact.count(')'):
            write_fact_to_file(fact, facts_set)
        else:
            print(f"⚠️ ERRORE: Parentesi sbilanciate -> {fact}")
    except Exception as e:
        print(f"❌ ERRORE nel generare il fatto per la canzone: {e}")


def process_cluster(song, facts_set):
    try:
        cluster = song.get('clusterIndex', -1)
        fact = f"clustered_song({song['track_popularity']}, {song['danceability']}, {song['energy']}, {song['key']}," \
            f" {song['loudness']}, {song['mode']}, {song['speechiness']}, {song['acousticness']}," \
            f" {song['instrumentalness']}, {song['liveness']}, {song['valence']}, {song['tempo']}," \
            f" {song['duration_ms']}, {cluster})."
        write_fact_to_file(fact, facts_set)
    except Exception as e:
        print(f"❌ ERRORE nel generare il fatto per il cluster: {e}")


def writeSongsInfo(dataSet, file_path):
    facts_set = set()
    with ThreadPoolExecutor() as executor:
        for _, song in dataSet.iterrows():
            executor.submit(process_song, song, facts_set)
    save_facts_to_file(file_path, facts_set)


def writeClusterInfo(dataSet, file_path):
    facts_set = set()
    with ThreadPoolExecutor() as executor:
        for _, song in dataSet.iterrows():
            executor.submit(process_cluster, song, facts_set)
    save_facts_to_file(file_path, facts_set)


def writeRules(file_path):
    rule = """relazione_song(Popularity, Cluster, Duration, Energy) :-
    song(Popularity, Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration, Cluster),
    clustered_song(Popularity, Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration, Cluster)."""

    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(rule + "\n")


def esegui_query_durata(prolog, df):
    min_duration = df['duration_ms'].min()
    max_duration = df['duration_ms'].max()

    while True:
        durata = input(
            "▶ Inserisci la durata della canzone (1. breve -b, 2. media -m, 3. lunga -l): ").lower()
        if durata in ['1', 'breve', 'b']:
            query_durata = "song(P, _, _, _, _, _, _, _, _, _, _, _, Duration), (Duration < 0.3)."
            break
        elif durata in ['2', 'media', 'm']:
            query_durata = "song(P, _, _, _, _, _, _, _, _, _, _, _, Duration), (Duration >= 0.3, Duration < 0.6)."
            break
        elif durata in ['3', 'lunga', 'l']:
            query_durata = "song(P, _, _, _, _, _, _, _, _, _, _, _, Duration), (Duration >= 0.6)."
            break
        else:
            print("❌ Errore: inserisci '1. breve', '2. media' o '3. lunga'.")

    try:
        results = list(prolog.query(query_durata))
        if results:
            print(f"\n✅ Canzoni trovate per durata '{durata}':")
            # Scegli casualmente fino a 4 canzoni tra i risultati
            sampled_results = random.sample(results, min(4, len(results)))

            for sol in sampled_results:
                duration_norm = sol['Duration']
                duration_val = duration_norm * \
                    (max_duration - min_duration) + min_duration
                matching_rows = df[df['duration_ms'].between(
                    duration_val - 2000, duration_val + 2000)]

                if not matching_rows.empty:
                    track_name = matching_rows.sample(
                        n=1).iloc[0]['track_name']
                    print(
                        f"   🎵 {track_name} (duration: {duration_val:.0f} ms)")
            print("...")
        else:
            print(f"\n⚠️ Nessuna canzone trovata per durata '{durata}'.")
    except Exception as e:
        print("❌ Errore durante la query Prolog:", e)


def esegui_query_popolarita(prolog, df):
    min_popularity = df['track_popularity'].min()
    max_popularity = df['track_popularity'].max()

    while True:
        popolarita = input(
            "▶ Inserisci la popolarità della canzone (1. bassa -b, 2. media -m, 3. alta -a): ").lower()
        if popolarita in ['1', 'bassa', 'b']:
            query_pop = "song(Popularity, _, _, _, _, _, _, _, _, _, _, _, _), (Popularity < 0.3)."
            break
        elif popolarita in ['2', 'media', 'm']:
            query_pop = "song(Popularity, _, _, _, _, _, _, _, _, _, _, _, _), (Popularity >= 0.3, Popularity < 0.6)."
            break
        elif popolarita in ['3', 'alta', 'a']:
            query_pop = "song(Popularity, _, _, _, _, _, _, _, _, _, _, _, _), (Popularity >= 0.6)."
            break
        else:
            print("❌ Errore: inserisci '1. bassa', '2. media' o '3. alta'.")

    try:
        results = list(prolog.query(query_pop))
        if results:
            print(f"\n✅ Canzoni trovate per popolarità '{popolarita}':")
            # Scegli casualmente fino a 4 canzoni tra i risultati
            sampled_results = random.sample(results, min(4, len(results)))

            for sol in sampled_results:
                popularity_norm = sol['Popularity']
                popularity_val = popularity_norm * \
                    (max_popularity - min_popularity) + min_popularity
                matching_rows = df[df['track_popularity'].between(
                    popularity_val - 0.05, popularity_val + 0.05)]

                if not matching_rows.empty:
                    track_name = matching_rows.sample(
                        n=1).iloc[0]['track_name']
                    print(
                        f"   🎵 {track_name} (popolarità: {popularity_val:.2f})")
            print("...")
        else:
            print(
                f"\n⚠️ Nessuna canzone trovata per popolarità '{popolarita}'.")
    except Exception as e:
        print("❌ Errore durante la query Prolog:", e)


def query_execute(path_file, clean_df, df):
    print("--- Test della KB con query di esempio ---")
    df = pd.read_csv("datasetSongs.csv")
    prolog = Prolog()
    prolog.consult(path_file)

    while True:
        print("\nScegli una query:")
        print("1. Cerca canzoni per durata")
        print("2. Cerca canzoni per popolarità")
        print("3. Esci")
        scelta = input("▶ Inserisci il numero della query: ")

        if scelta == '1':
            esegui_query_durata(prolog, df)
        elif scelta == '2':
            esegui_query_popolarita(prolog, df)
        elif scelta == '3':
            print("Test terminato...")
            break
        else:
            print("❌ Scelta non valida. Riprova.")
