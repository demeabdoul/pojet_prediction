import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Prédiction de la Puissance PV avec Machine Learning")

# Upload du fichier Excel
file = st.file_uploader("Importer le fichier Excel", type=["xlsx"])

if file is not None:

    df = pd.read_excel(file)

    st.subheader("Aperçu des données")
    st.write(df.head())

    st.write("Dimension des données :", df.shape)

    # Nettoyage
    df = df.dropna()

    colonnes = [
        "Rayonnement global plan PV (W/m2)",
        "Température cellule PV (C°)",
        "Température generateur PV (C°)",
        "Température Batterie (C°)",
        "Température Ambiante (C°)",
        "Consommation sortie Ond.(W)",
        "Puissance PV"
    ]

    scaler = MinMaxScaler()
    df[colonnes] = scaler.fit_transform(df[colonnes])

    st.subheader("Matrice de corrélation")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

    # Variables
    X = df[[
        "Rayonnement global plan PV (W/m2)",
        "Température cellule PV (C°)",
        "Température generateur PV (C°)",
        "Température Ambiante (C°)"
    ]]

    y = df["Puissance PV"]

    # Division
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Modèles
    model_lr = LinearRegression()
    model_rf = RandomForestRegressor(n_estimators=100)
    model_dt = DecisionTreeRegressor()

    # Entrainement
    model_lr.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)
    model_dt.fit(X_train, y_train)

    # Prédictions
    y_pred = model_lr.predict(X_test)
    y_pred_rf = model_rf.predict(X_test)
    y_pred_dt = model_dt.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    r2_dt = r2_score(y_test, y_pred_dt)

    metrics_df = pd.DataFrame({
        "Modèle": ["Régression Linéaire", "Random Forest", "Decision Tree"],
        "MAE": [mae, mae_rf, mae_dt],
        "RMSE": [rmse, rmse_rf, rmse_dt],
        "R²": [r2, r2_rf, r2_dt]
    })

    st.subheader("Performance des modèles")

    st.write(metrics_df)

    # Graphique comparaison
    fig2, ax2 = plt.subplots()

    ax2.scatter(y_test, y_pred, label="Régression Linéaire")
    ax2.scatter(y_test, y_pred_rf, label="Random Forest")
    ax2.scatter(y_test, y_pred_dt, label="Decision Tree")

    ax2.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )

    ax2.set_xlabel("Puissance réelle")
    ax2.set_ylabel("Puissance prédite")
    ax2.set_title("Comparaison des modèles")

    ax2.legend()

    st.pyplot(fig2)