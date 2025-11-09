# ============================================================
# MODELO KNN EN ARRIENDOS (usando Polars)
# ============================================================

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import pandas as pd

# 1. Cargar datos con Polars
df = pl.read_csv("DatosRevisadosLimpios.csv")

# 2. Renombrar columnas
df = df.rename({
    'Precio ($)': 'Precio',
    'Metros (m²)': 'MetrosCuadrados',
    'Gastos_Comunes ($)': 'GastosComunes'
})

# 3. Seleccionar columnas relevantes y eliminar nulos
df = df.select(['Precio', 'MetrosCuadrados', 'Comuna', 'Baños', 'Habitaciones', 'GastosComunes']).drop_nulls()

# 4. Crear variable binaria de "Arriendo alto"
umbral = df['Precio'].median()
df = df.with_columns([
    (df['Precio'] > umbral).cast(pl.Int8).alias('ArriendoAlto')
])

print(f"Mediana del precio: {umbral:.2f}")
print(df['ArriendoAlto'].value_counts())

# 5. Convertir a pandas (para sklearn)
df_pd = df.to_pandas()

# 6. Separar X e y
X = df_pd[['MetrosCuadrados', 'Comuna', 'Baños', 'Habitaciones', 'GastosComunes']]
y = df_pd['ArriendoAlto']

# 7. División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Preprocesamiento (escalar + codificar)
num_cols = ['MetrosCuadrados', 'Baños', 'Habitaciones', 'GastosComunes']
cat_cols = ['Comuna']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# ============================================================
# MODELO: K-NEAREST NEIGHBORS (KNN)
# ============================================================

knn_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# ============================================================
# MÉTRICAS
# ============================================================

knn_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_knn),
    'Precision': precision_score(y_test, y_pred_knn),
    'Recall': recall_score(y_test, y_pred_knn)
}

print("\n📊 Métricas del Modelo KNN:")
for k, v in knn_metrics.items():
    print(f"{k}: {v:.3f}")

# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(knn_model, X_test, y_test, ax=ax, cmap='Greens')
ax.set_title("Matriz de Confusión - KNN")
plt.show()

# ============================================================
# GRÁFICO DE MÉTRICAS
# ============================================================

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=list(knn_metrics.keys()), y=list(knn_metrics.values()), color="green", ax=ax)
ax.set_ylim(0, 1)
ax.set_title("Desempeño del Modelo KNN")
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)
plt.show()

# ============================================================
# PREDICCIÓN MANUAL DE NUEVOS DATOS
# ============================================================

print("\n--- 🔍 Predicción manual con modelo KNN ---")

# Pedir datos al usuario
try:
    metros = float(input("Metros cuadrados: "))
    banos = int(input("Número de baños: "))
    habs = int(input("Número de habitaciones: "))
    gastos = float(input("Gastos comunes ($): "))
    comuna = input("Nombre exacto de la comuna (según dataset): ").strip()

    # Crear DataFrame con los valores ingresados
    nuevo_dato = pd.DataFrame([{
        'MetrosCuadrados': metros,
        'Comuna': comuna,
        'Baños': banos,
        'Habitaciones': habs,
        'GastosComunes': gastos
    }])

    # Predicción
    prediccion = knn_model.predict(nuevo_dato)[0]
    print("\n🔎 Resultado del modelo:")
    if prediccion == 1:
        print("➡️ El modelo predice que el arriendo sería ALTO 💰")
    else:
        print("➡️ El modelo predice que el arriendo sería BAJO 🏡")

except Exception as e:
    print("\n⚠️ Error al ingresar o procesar los datos:", e)
