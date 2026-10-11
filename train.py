import argparse
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Definición del experimento en MLFlow
MLFLOW_EXPERIMENT_NAME = "UCI_Heart_Disease_MLflow."


def set_seed(seed: int):
    """Asegura la reproducibilidad de los resultados fijando la semilla aleatoria."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    """Carga el dataset de Heart Disease desde UCI Repository."""
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    return pd.concat([X, y], axis=1)


def preprocess_data(df: pd.DataFrame, seed: int):
    """Limpia el dataset, realiza la limpieza de nulos, binariza la variable objetivo
    y escala las características.
    """
    df_clean = df.copy()

    # Limpieza de nulos con mediana (ca) y moda (thal)
    df_clean["ca"] = df_clean["ca"].fillna(df_clean["ca"].median())
    df_clean["thal"] = df_clean["thal"].fillna(df_clean["thal"].mode()[0])

    # Binarización de la variable objetivo (0: Sano, 1: Enfermo)
    df_clean["target"] = (df_clean["num"] > 0).astype(int)
    df_clean = df_clean.drop(columns=["num"])

    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    # Divisón Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train, C: float, max_iter: int, solver: str, seed: int):
    """Entrena un modelo de Regresión Logística."""
    model = LogisticRegression(
        C=C, max_iter=max_iter, solver=solver, random_state=seed
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Calcula las métricas de evaluación del modelo."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    return metrics, y_pred


def plot_and_save_confusion_matrix(y_test, y_pred, output_path="confusion_matrix.png"):
    """Genera y guarda la matriz de confusión como un artefacto visual."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sano", "Enfermo"],
        yticklabels=["Sano", "Enfermo"],
    )
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento y Tracking con MLFlow para Heart Disease"
    )

    # Argumentos parametrizados con valores por defecto
    parser.add_argument(
        "--C", type=float, default=1.0, help="Inverso de la fuerza de regularización"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Número máximo de iteraciones de optimización",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear", "saga"],
        help="Algoritmo de optimización",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla de aleatoriedad"
    )

    args = parser.parse_args()

    # Fijación de la semilla
    set_seed(args.seed)

    # Configuración de MLFlow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        # Carga y procesamiento
        df_raw = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(df_raw, args.seed)

        # Registro de hiperparámetros y semilla
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("seed", args.seed)

        # Entrenamiento
        model = train_model(
            X_train, y_train, args.C, args.max_iter, args.solver, args.seed
        )

        # Evaluación
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        # Registro de métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Generación y registro de artefacto (Matriz de Confusión)
        cm_path = plot_and_save_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path)
        if os.path.exists(cm_path):
            os.remove(cm_path)

        # Registro del modelo entrenado
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Corrida completada con éxito. F1-Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()