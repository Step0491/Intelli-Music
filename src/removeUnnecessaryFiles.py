import os
import shutil


def remove_unnecessary_files():
    """
    Rimuove i file non necessari dalla directory corrente e da src\\__pycache__.
    """
    # Lista dei file da rimuovere nella directory corrente
    files_to_remove = [
        'cleanDataset.csv',
        'bayesian_network.joblib',
        'dataset_resampled.joblib',
        'DecisionTree_model.pkl',
        'kb.pl',
        'kmeans_model_7.joblib',
        'LogisticRegression_model.pkl',
        'RandomForest_model.pkl'
    ]

    # Rimuovi i file nella directory corrente
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                print(
                    f"\033[92mFile '{file_name}' rimosso con successo!\033[0m")
            except Exception as e:
                print(
                    f"\033[91mErrore durante la rimozione del file '{file_name}': {e}\033[0m")
        else:
            print(f"\033[93mFile '{file_name}' non trovato!\033[0m")

    # Rimuovi i file .pyc nella cartella __pycache__
    pycache_dir = 'src\\__pycache__'
    if os.path.exists(pycache_dir):
        try:
            # Rimuovi la directory __pycache__ e tutto il suo contenuto
            shutil.rmtree(pycache_dir)
            print(
                f"\033[92mCartella '{pycache_dir}' e il suo contenuto rimosso con successo!\033[0m")
        except Exception as e:
            print(
                f"\033[91mErrore durante la rimozione della cartella '{pycache_dir}': {e}\033[0m")
    else:
        print(f"\033[93mLa cartella '{pycache_dir}' non esiste!\033[0m")


def main():
    # Chiamata alla funzione per rimuovere i file
    remove_unnecessary_files()


if __name__ == "__main__":
    main()
