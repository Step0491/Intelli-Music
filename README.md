<p align="center">
  <h1 align="center">IntelliMusic System</h1>
</p>

Questa applicazione gestisce un processo complesso di analisi e clustering di playlist musicali.
Il programma è progettato per caricare un dataset di canzoni, preprocessarlo, applicare algoritmi di clustering, normalizzazione e discretizzazione; successivamente si procede con l'applicazione di algoritmi di apprendimento supervisionato/non supervisionato, generazione di una KnowledgeBase con seguente testing attraverso query, oversampling e conseguente ri-applicazione dei vari algoritmi per confronti ed infine si genera una rete bayesian Il tutto illustrando i vari processi attraverso grafici di vario tipo.


## Programming Language: Python

## Local Setup

Per eseguire il progetto:
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the `main.py`

## Data Source
Il dataset impiegato nel progetto è stato installato dalla piattaforma Kaggle, è pubblico e disponibile all'utilizzo per chiunque

## Punti Chiave del Progetto

- **1 Configurazione iniziale**
Importa molte librerie, tra cui pandas, sklearn, streamlit, e moduli personalizzati (SongsProlog, unsupervisedLearning, ecc.).

- **2 Pulizia e Preprocessing del Dataset**
Mantiene solo le colonne rilevanti (clean_columns).
Rimuove colonne non utili (come ID, nome traccia, artista, ecc.).
Visualizza statistiche sui dati con visualizeAspectRatioChart().

- **3 Normalizzazione e Clustering**
Applica MinMaxScaler per normalizzare le feature numeriche.
Esegue il clustering sui dati con K-Means.

- **4 Apprendimento Supervisionato e Oversampling**
Allena un modello con K-Fold Cross Validation (trainModelKFold()).
Esegue oversampling per bilanciare i dati (overSampling()).
Visualizza il nuovo dataset riequilibrato.

- **5 Creazione della Rete Bayesiana**
Discretizza le feature numeriche (KBinsDiscretizer()).
Costruisce una rete bayesiana (bNetCreation() e loadBayesianNetwork()).



