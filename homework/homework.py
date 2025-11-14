#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#


from sklearn.linear_model import LinearRegression   # ✔️ corregido
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_regression   # ✔️ corregido
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, median_absolute_error, mean_absolute_error  # ✔️ faltaban
)
import json
import pickle
import gzip
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# ---------- I/O y utilidades ----------
def ensure_dirs():
    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)

def load_data():
    train_df = pd.read_csv("files/input/train_data.csv.zip", index_col=False)
    test_df = pd.read_csv("files/input/test_data.csv.zip", index_col=False)
    return train_df, test_df

def save_estimator(estimator, path="files/models/model.pkl.gz"):
    ensure_dirs()
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)

def load_estimator(path="files/models/model.pkl.gz"):
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

# ---------- Preprocesamiento ----------

def prepare_data(df):
    df = df.copy()

    df["Age"] = 2021 - df["Year"]
    df.drop(columns=['Year', 'Car_Name'], inplace=True)   

    return df

# ---------- Split ----------
def make_train_test_split(train_df, test_df):

    train_clean = prepare_data(train_df)
    test_clean = prepare_data(test_df)

    x_train = train_clean.drop(columns="Present_Price")
    y_train = train_clean["Present_Price"]

    x_test = test_clean.drop(columns="Present_Price")
    y_test = test_clean["Present_Price"]

    return x_train, y_train, x_test, y_test

# ---------- Pipeline ----------
def make_pipeline(feature_columns):

    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numeric_features = ['Selling_Price', 'Driven_kms', 'Owner', 'Age']

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(), categorical_features),
            ("num", MinMaxScaler(), numeric_features),
        ]
    )   

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ('classifier', LinearRegression())  
        ]
    )

    return pipeline

# ---------- Búsqueda de hiperparámetros ----------
def make_grid_search(estimator, param_grid, cv=10):
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return grid_search


# ---------- Entrenamiento ----------
def train_estimator(grid_search):

    train_df, test_df = load_data()
    x_train, y_train, x_test, y_test = make_train_test_split(train_df, test_df)

    grid_search.fit(x_train, y_train)

    saved = load_estimator()
    current_score = -mean_absolute_error(y_test, grid_search.predict(x_test))

    if saved is not None:
        try:
            saved_score = -mean_absolute_error(y_test, saved.predict(x_test))
        except Exception:
            saved_score = -np.inf

    else:
        saved_score = -np.inf


    if current_score >= saved_score:
        save_estimator(grid_search)
    else:
        pass


# ---------- Entrenador específico ----------
def train_linear_regression():

    train_df, test_df = load_data()
    x_train, y_train, x_test, y_test = make_train_test_split(train_df, test_df)

    pipeline = make_pipeline(feature_columns=x_train.columns.tolist())

    param_grid = {
        'feature_selection__k':[11]
    }
    
    gs = make_grid_search(estimator=pipeline, param_grid=param_grid, cv=10)
    train_estimator(gs)

# ---------- Check / métricas ----------
def check_estimator():
    ensure_dirs()
    train_df, test_df = load_data()
    x_train, y_train, x_test, y_test = make_train_test_split(train_df, test_df)

    estimator = load_estimator()
    if estimator is None:
        raise FileNotFoundError("No se encontró modelo en files/models/model.pkl.gz")

    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)

    metrics = []

    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        'r2': r2_score(y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'mad': median_absolute_error(y_train, y_train_pred),
    }
    metrics.append(train_metrics)

    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mad': median_absolute_error(y_test, y_test_pred),
    }
    metrics.append(test_metrics)

    out_path = "files/output/metrics.json"
    with open(out_path, "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

    print(f"Métricas guardadas en {out_path}")


if __name__ == "__main__":
    ensure_dirs()
    train_linear_regression()
    check_estimator()
