import os
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.cm as cm


import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE


def overSampling(dataSet, differentialColumn, threshold_ratio=0.7):
    """
    Applica SMOTE per bilanciare solo i cluster più piccoli, mantenendo quelli più grandi invariati.

    :param dataSet: DataFrame da riequilibrare
    :param differentialColumn: Nome della colonna con le classi/cluster
    :param threshold_ratio: Percentuale rispetto al cluster più grande sotto la quale applicare SMOTE
    :return: DataFrame bilanciato
    """

    if differentialColumn not in dataSet.columns:
        raise ValueError(
            f"Errore: La colonna '{differentialColumn}' non è presente nel dataset!")

    # Rimuove righe con valori NaN nella colonna di riferimento
    dataSet = dataSet.dropna(subset=[differentialColumn])
    X = dataSet.drop(columns=[differentialColumn])
    y = dataSet[differentialColumn]

    # Conta la distribuzione iniziale delle classi
    class_counts = y.value_counts()
    max_count = class_counts.max()  # Numero di campioni del cluster più grande
    # Soglia sotto cui fare oversampling
    threshold = int(max_count * threshold_ratio)

    # Seleziona solo le classi con meno campioni della soglia
    classes_to_resample = class_counts[class_counts < threshold].index.tolist()

    if not classes_to_resample:
        print("Nessun cluster ha bisogno di oversampling.")
        return dataSet  # Restituisce il dataset originale se non serve SMOTE

    # Applica SMOTE solo alle classi sotto la soglia
    smote = SMOTE(sampling_strategy={
                  cls: threshold for cls in classes_to_resample}, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Ricrea il DataFrame bilanciato
    dataSet_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    dataSet_resampled[differentialColumn] = y_resampled

    print("OverSampling Effettuato solo sui cluster più piccoli.")

    # Salva il dataset bilanciato
    joblib.dump(dataSet_resampled, 'dataset_resampled.joblib')

    return dataSet_resampled


def visualizeAspectRatioChart(dataSet, differentialColumn, title, filename):
    """
    Genera un grafico a torta per visualizzare la distribuzione della colonna specificata.
    """
    if differentialColumn not in dataSet.columns:
        raise ValueError(f"Errore: La colonna '{
                         differentialColumn}' non è presente nel dataset!")

    # Ordina in ordine decrescente
    counts = dataSet[differentialColumn].value_counts(
    ).sort_values(ascending=False)
    labels = counts.index.tolist()

    # Escapa il carattere '$' nelle etichette
    labels = [label.replace('$', '\\$') if isinstance(
        label, str) else label for label in labels]

    # Usa una palette di colori automatica
    available_colors = plt.cm.get_cmap(
        'tab20').colors  # Usa tab10 per colori chiari

    colors = cm.get_cmap('tab20', len(labels))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors(range(len(labels))),
        autopct='%1.2f%%', startangle=90
    )
    ax.legend(labels, loc='lower left', fontsize='small')
    plt.title(title)

    save_path = os.path.join("charts", f"{filename}.png")
    os.makedirs("charts", exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    plt.close()


def salvaDataSet(dataSet, filename):
    """
    Serializza il dataset e lo salva su file.
    """
    joblib.dump(dataSet, filename)


def caricaDataSet(filename):
    """
    Carica un dataset precedentemente serializzato.
    """
    return joblib.load(filename)
