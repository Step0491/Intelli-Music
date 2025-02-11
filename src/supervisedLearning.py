import json
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, learning_curve, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from joblib import Parallel, delayed, dump


def save_model(model, filename):
    """Salva il modello su disco con controllo se il file esiste già."""
    try:
        if os.path.exists(filename):
            os.remove(filename)
        joblib.dump(model, filename)
        print(f"Modello salvato come {filename}")
    except Exception as e:
        print(f"Errore durante il salvataggio del modello: {e}")


def load_model(filename):
    """Carica un modello salvato."""
    if os.path.exists(filename):
        try:
            model = joblib.load(filename)
            print(f"Modello {filename} caricato con successo.")
            return model
        except Exception as e:
            print(f"Errore durante il caricamento del modello: {e}")
            return None
    else:
        print(f"Il file {filename} non esiste.")
        return None


def plot_learning_curves(model, X_train, y_train, model_name):
    """Genera, salva e visualizza la curva di apprendimento."""
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=10, scoring='accuracy')

        train_errors = 1 - train_scores
        test_errors = 1 - test_scores

        mean_train_errors = np.mean(train_errors, axis=1)
        mean_test_errors = np.mean(test_errors, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, mean_train_errors,
                 label='Errore di training', color='green')
        plt.plot(train_sizes, mean_test_errors,
                 label='Errore di testing', color='red')
        plt.title(f'Curva di apprendimento per {model_name}')
        plt.xlabel('Dimensione del training set')
        plt.ylabel('Errore')
        plt.legend()
        save_path = os.path.join("charts", f"learning_curve_{model_name}.png")
        os.makedirs("charts", exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Errore nella generazione della curva di apprendimento: {e}")


def return_best_hyperparameters(dataset, target_column):
    """Restituisce i migliori iperparametri per vari modelli."""
    X = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression()
    }

    param_grids = {
        'DecisionTree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        },
        'RandomForest': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        },
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        }
    }

    best_params = {}

    def search_best_params(name, model, param_grid):
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return name, grid_search.best_params_

    results = Parallel(n_jobs=-1)(delayed(search_best_params)(name, model, param_grids[name])
                                  for name, model in models.items())

    for name, params in results:
        best_params[name] = params

    return best_params


def train_model_kfold(dataset, target_column):
    """Addestra i modelli con validazione incrociata, salva i modelli migliori e valuta su test set."""
    best_params = return_best_hyperparameters(dataset, target_column)

    print("Migliori iperparametri trovati:\n",
          json.dumps(best_params, indent=4))

    X = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        'DecisionTree': DecisionTreeClassifier(**best_params['DecisionTree']),
        'RandomForest': RandomForestClassifier(**best_params['RandomForest']),
        'LogisticRegression': LogisticRegression(**best_params['LogisticRegression'])
    }

    # Train models and store scores
    for name, model in models.items():
        model.fit(X_train, y_train)
        dump(model, f"{name}_model.pkl")

    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

    def cross_val_score_parallel(model):
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

    model_scores = Parallel(n_jobs=-1)(delayed(cross_val_score_parallel)(model)
                                       for model in models.values())

    for (name, model), scores in zip(models.items(), model_scores):
        print(
            f"Modello {name}: Accuratezza media: {np.mean(scores):.4f}, Deviazione standard: {np.std(scores):.4f}")

        model.fit(X_train, y_train)
        dump(model, f"{name}_model.pkl")

        y_pred = model.predict(X_test)
        print(f"Performance su test set per {name}:")
        print(classification_report(y_test, y_pred))

        plot_learning_curves(model, X_train, y_train, name)
    plot_model_performance(models, X_test, y_test)
    return models


def plot_model_performance(models, X_test, y_test):
    """
    Plots the performance of each model without displaying hyperparameters.
    """
    metrics = []
    model_names = list(models.keys())

    # Gather metrics for each model
    for name in model_names:
        model = models[name]
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(
            y_test, y_pred, average='weighted', zero_division=1)

        metrics.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        })

    # Create a DataFrame for easy plotting
    metrics_df = pd.DataFrame(metrics)

    # Plot metrics for each model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Accuracy Plot
    axes[0].bar(metrics_df['Model'], metrics_df['Accuracy'], color='skyblue')
    axes[0].set_title('Accuracy of Models')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')

    # Precision Plot
    axes[1].bar(metrics_df['Model'], metrics_df['Precision'], color='orange')
    axes[1].set_title('Precision of Models')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Precision')

    # Recall Plot
    axes[2].bar(metrics_df['Model'], metrics_df['Recall'], color='green')
    axes[2].set_title('Recall of Models')
    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('Recall')

    plt.tight_layout()

    save_path = os.path.join("charts", "model_performance.png")
    os.makedirs("charts", exist_ok=True)
    plt.savefig(save_path, format="png", dpi=300)
    # Show and save the plot
    plt.show()
    plt.close()

