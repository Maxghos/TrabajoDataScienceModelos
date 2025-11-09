# -*- coding: utf-8 -*-
# ==============================================================
# SEMANA 6: ARBOLES DE DECISION Y REGULARIZACION
# Código Educativo basado en "Clase 6.html"
# Aplicado a datos de arriendos de Santiago
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import sys
import io

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# Configuración de visualización
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

print("="*80)
print("SEMANA 6: ARBOLES DE DECISION Y REGULARIZACION")
print("="*80)
print("\nEste codigo demuestra los conceptos ensenados en la Clase 6:")
print("1. Arboles de Decision (modelos interpretables)")
print("2. Sobreajuste (Overfitting) y como detectarlo")
print("3. Regularizacion (tecnicas para evitar overfitting)")
print("4. Random Forest (modelos de ensamble)")
print("5. Comparacion entre modelos simples y complejos")
print("="*80)

# ==============================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================

print("\nPASO 1: Cargando y preparando los datos...")
df = pd.read_csv("DatosRevisadosLimpios.csv")

# Renombrar columnas
df = df.rename(columns={
    'Precio ($)': 'Precio',
    'Metros (m²)': 'MetrosCuadrados',
    'Gastos_Comunes ($)': 'GastosComunes'
})

# Seleccionar columnas relevantes
df = df[['Precio', 'MetrosCuadrados', 'Comuna', 'Baños', 'Habitaciones', 'GastosComunes']].dropna()

print(f"   [OK] Datos cargados: {len(df)} registros")
print(f"   [OK] Variables predictoras: MetrosCuadrados, Comuna, Banos, Habitaciones, GastosComunes")
print(f"   [OK] Variable objetivo: Precio")

# Separar variables
X = df[['MetrosCuadrados', 'Comuna', 'Baños', 'Habitaciones', 'GastosComunes']]
y = df['Precio']

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"   [OK] Datos de entrenamiento: {len(X_train)} registros")
print(f"   [OK] Datos de prueba: {len(X_test)} registros")

# Preprocesamiento
num_cols = ['MetrosCuadrados', 'Baños', 'Habitaciones', 'GastosComunes']
cat_cols = ['Comuna']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

# ==============================================================
# 2. CONCEPTO: ÁRBOL DE DECISIÓN SIN REGULARIZACIÓN
# ==============================================================

print("\n" + "="*80)
print("CONCEPTO 1: Arbol de Decision SIN Regularizacion")
print("="*80)
print("\nUn árbol sin restricciones crecerá hasta memorizar los datos.")
print("Esto es útil para entender el concepto, pero genera SOBREAJUSTE.\n")

# Árbol sin regularización (muy profundo)
tree_sin_regularizacion = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(
        max_depth=None,  # Sin límite de profundidad
        min_samples_leaf=1,  # Mínimo de muestras por hoja
        random_state=42
    ))
])

tree_sin_regularizacion.fit(X_train, y_train)
y_pred_train_sin_reg = tree_sin_regularizacion.predict(X_train)
y_pred_test_sin_reg = tree_sin_regularizacion.predict(X_test)

# Métricas en entrenamiento y prueba
mae_train_sin = mean_absolute_error(y_train, y_pred_train_sin_reg)
mae_test_sin = mean_absolute_error(y_test, y_pred_test_sin_reg)
r2_train_sin = r2_score(y_train, y_pred_train_sin_reg)
r2_test_sin = r2_score(y_test, y_pred_test_sin_reg)

print("RESULTADOS DEL ARBOL SIN REGULARIZACION:")
print(f"   Entrenamiento - MAE: ${mae_train_sin:,.0f}, R²: {r2_train_sin:.4f}")
print(f"   Prueba         - MAE: ${mae_test_sin:,.0f}, R²: {r2_test_sin:.4f}")
print(f"\n[ADVERTENCIA] DIFERENCIA ENTRE ENTRENAMIENTO Y PRUEBA:")
print(f"   La diferencia en R² es: {r2_train_sin - r2_test_sin:.4f}")
print(f"   Esto indica SOBREAJUSTE: el modelo memoriza los datos de entrenamiento")
print(f"   pero no generaliza bien a datos nuevos.")

# ==============================================================
# 3. CONCEPTO: REGULARIZACIÓN (Evitando el Sobreajuste)
# ==============================================================

print("\n" + "="*80)
print("CONCEPTO 2: Regularizacion (Evitando el Sobreajuste)")
print("="*80)
print("\nAplicaremos técnicas de PODA para limitar el crecimiento del árbol:")
print("  • max_depth: Limita la profundidad máxima")
print("  • min_samples_leaf: Exige mínimo de datos por hoja\n")

# Árbol con regularización
tree_con_regularizacion = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(
        max_depth=5,  # Limita profundidad (REGULARIZACIÓN)
        min_samples_leaf=10,  # Mínimo de muestras por hoja (REGULARIZACIÓN)
        random_state=42
    ))
])

tree_con_regularizacion.fit(X_train, y_train)
y_pred_train_con_reg = tree_con_regularizacion.predict(X_train)
y_pred_test_con_reg = tree_con_regularizacion.predict(X_test)

# Métricas
mae_train_con = mean_absolute_error(y_train, y_pred_train_con_reg)
mae_test_con = mean_absolute_error(y_test, y_pred_test_con_reg)
r2_train_con = r2_score(y_train, y_pred_train_con_reg)
r2_test_con = r2_score(y_test, y_pred_test_con_reg)

print("RESULTADOS DEL ARBOL CON REGULARIZACION:")
print(f"   Entrenamiento - MAE: ${mae_train_con:,.0f}, R²: {r2_train_con:.4f}")
print(f"   Prueba         - MAE: ${mae_test_con:,.0f}, R²: {r2_test_con:.4f}")
print(f"\n[OK] MEJORA CON REGULARIZACION:")
print(f"   Diferencia en R² (entrenamiento - prueba): {r2_train_con - r2_test_con:.4f}")
print(f"   Esta diferencia es MENOR, lo que indica mejor generalizacion.")
print(f"   El modelo es mas simple pero mas confiable.")

# ==============================================================
# 4. VISUALIZACIÓN: Comparación de Sobreajuste
# ==============================================================

print("\n" + "="*80)
print("VISUALIZACION 1: Comparacion de Sobreajuste")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Sin regularización
axes[0].scatter(y_test, y_pred_test_sin_reg, alpha=0.5, s=30)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Precio Real', fontsize=12)
axes[0].set_ylabel('Precio Predicho', fontsize=12)
axes[0].set_title(f'Arbol SIN Regularizacion\nR² = {r2_test_sin:.3f} | MAE = ${mae_test_sin:,.0f}', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Con regularización
axes[1].scatter(y_test, y_pred_test_con_reg, alpha=0.5, s=30)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Precio Real', fontsize=12)
axes[1].set_ylabel('Precio Predicho', fontsize=12)
axes[1].set_title(f'Arbol CON Regularizacion\nR² = {r2_test_con:.3f} | MAE = ${mae_test_con:,.0f}', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparacion_sobreajuste.png', dpi=300, bbox_inches='tight')
print("   [OK] Grafico guardado: comparacion_sobreajuste.png")
plt.show()

# ==============================================================
# 5. CONCEPTO: RANDOM FOREST (Modelo de Ensamble)
# ==============================================================

print("\n" + "="*80)
print("CONCEPTO 3: Random Forest (Modelo de Ensamble)")
print("="*80)
print("\nRandom Forest combina MUCHOS árboles para crear un modelo más robusto:")
print("  • Cada árbol se entrena con una muestra aleatoria de datos")
print("  • Cada árbol considera solo una muestra aleatoria de variables")
print("  • La predicción final es el promedio de todos los árboles")
print("  • Esto reduce el sobreajuste y mejora la precisión\n")

# Random Forest
forest_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,  # 100 árboles en el bosque
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1  # Usar todos los cores disponibles
    ))
])

print("   [ENTRENANDO] Entrenando Random Forest (100 arboles)...")
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)

# Métricas
mae_forest = mean_absolute_error(y_test, y_pred_forest)
rmse_forest = np.sqrt(mean_squared_error(y_test, y_pred_forest))
r2_forest = r2_score(y_test, y_pred_forest)

print("   [OK] Entrenamiento completado\n")

print("RESULTADOS DEL RANDOM FOREST:")
print(f"   Prueba - MAE: ${mae_forest:,.0f}")
print(f"   Prueba - RMSE: ${rmse_forest:,.0f}")
print(f"   Prueba - R²: {r2_forest:.4f}")

# ==============================================================
# 6. COMPARACIÓN COMPLETA DE MODELOS
# ==============================================================

print("\n" + "="*80)
print("COMPARACION COMPLETA DE MODELOS")
print("="*80)

resultados = pd.DataFrame({
    'Modelo': [
        'Arbol SIN Regularizacion',
        'Arbol CON Regularizacion',
        'Random Forest (100 arboles)'
    ],
    'MAE (Prueba)': [
        mae_test_sin,
        mae_test_con,
        mae_forest
    ],
    'R² (Prueba)': [
        r2_test_sin,
        r2_test_con,
        r2_forest
    ],
    'Diferencia R² (Train-Test)': [
        r2_train_sin - r2_test_sin,
        r2_train_con - r2_test_con,
        'N/A (ensamble más robusto)'
    ]
})

print("\n" + resultados.to_string(index=False))
print("\nINTERPRETACION:")
print("   • Arbol SIN regularizacion: Mejor R² en entrenamiento, pero peor en prueba (SOBREAJUSTE)")
print("   • Arbol CON regularizacion: Balance mejor entre entrenamiento y prueba")
print("   • Random Forest: Mejor rendimiento general y mas robusto")

# ==============================================================
# 7. VISUALIZACIÓN: Comparación de los 3 Modelos
# ==============================================================

print("\n" + "="*80)
print("VISUALIZACION 2: Comparacion de los 3 Modelos")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Modelo 1: Sin regularización
axes[0].scatter(y_test, y_pred_test_sin_reg, alpha=0.4, s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Precio Real', fontsize=11)
axes[0].set_ylabel('Precio Predicho', fontsize=11)
axes[0].set_title(f'Arbol SIN Regularizacion\nR² = {r2_test_sin:.3f}', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Modelo 2: Con regularización
axes[1].scatter(y_test, y_pred_test_con_reg, alpha=0.4, s=20)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Precio Real', fontsize=11)
axes[1].set_ylabel('Precio Predicho', fontsize=11)
axes[1].set_title(f'Arbol CON Regularizacion\nR² = {r2_test_con:.3f}', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Modelo 3: Random Forest
axes[2].scatter(y_test, y_pred_forest, alpha=0.4, s=20)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[2].set_xlabel('Precio Real', fontsize=11)
axes[2].set_ylabel('Precio Predicho', fontsize=11)
axes[2].set_title(f'Random Forest\nR² = {r2_forest:.3f}', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparacion_tres_modelos.png', dpi=300, bbox_inches='tight')
print("   [OK] Grafico guardado: comparacion_tres_modelos.png")
plt.show()

# ==============================================================
# 8. IMPORTANCIA DE VARIABLES (Random Forest)
# ==============================================================

print("\n" + "="*80)
print("IMPORTANCIA DE VARIABLES (Random Forest)")
print("="*80)
print("\nRandom Forest puede mostrar que variables son mas importantes")
print("para predecir el precio de arriendo.\n")

# Extraer importancias
rf_regressor = forest_model.named_steps['regressor']
encoder = forest_model.named_steps['preprocessor'].named_transformers_['cat']
encoded_cols = list(encoder.get_feature_names_out(cat_cols))
feature_names = num_cols + encoded_cols

importances = pd.DataFrame({
    'Variable': feature_names,
    'Importancia': rf_regressor.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("TOP 10 Variables Mas Importantes:")
print(importances.head(10).to_string(index=False))

# Visualización
plt.figure(figsize=(12, 8))
top_15 = importances.head(15)
sns.barplot(data=top_15, y='Variable', x='Importancia', palette='viridis')
plt.title('Importancia de las Variables - Random Forest\n(Top 15)', fontsize=16, fontweight='bold')
plt.xlabel('Importancia', fontsize=12)
plt.ylabel('Variable', fontsize=12)
plt.tight_layout()
plt.savefig('importancia_variables.png', dpi=300, bbox_inches='tight')
print("\n   [OK] Grafico guardado: importancia_variables.png")
plt.show()

# ==============================================================
# 9. VISUALIZACIÓN DEL ÁRBOL (Simplificado)
# ==============================================================

print("\n" + "="*80)
print("VISUALIZACION 3: Estructura del Arbol de Decision")
print("="*80)
print("\nMostrando la estructura de un arbol regularizado (profundidad 3)")
print("para entender como toma decisiones.\n")

# Crear árbol simplificado para visualización
X_simple = X.copy()
X_simple = pd.get_dummies(X_simple, columns=['Comuna'], drop_first=False, prefix='Comuna')
tree_vis = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
tree_vis.fit(X_simple, y)

plt.figure(figsize=(20, 10))
plot_tree(tree_vis, feature_names=X_simple.columns, filled=True, fontsize=9, 
          max_depth=3, rounded=True, precision=2)
plt.title('Estructura de un Arbol de Decision Regularizado\n(max_depth=3, min_samples_leaf=20)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('arbol_decision_estructura.png', dpi=300, bbox_inches='tight')
print("   [OK] Grafico guardado: arbol_decision_estructura.png")
plt.show()

# ==============================================================
# 10. GRÁFICO DE BARRAS: Comparación de Métricas
# ==============================================================

print("\n" + "="*80)
print("VISUALIZACION 4: Comparacion de Metricas")
print("="*80)

# Preparar datos para gráfico de barras
comparacion_metricas = pd.DataFrame({
    'Modelo': [
        'Arbol\n(Sin Reg.)',
        'Arbol\n(Con Reg.)',
        'Random\nForest'
    ],
    'MAE': [mae_test_sin, mae_test_con, mae_forest],
    'R²': [r2_test_sin, r2_test_con, r2_forest]
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico MAE
axes[0].bar(comparacion_metricas['Modelo'], comparacion_metricas['MAE'], 
           color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
axes[0].set_ylabel('MAE (Error Absoluto Medio)', fontsize=12)
axes[0].set_title('Comparación de MAE\n(Menor es mejor)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(comparacion_metricas['MAE']):
    axes[0].text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=10)

# Gráfico R²
axes[1].bar(comparacion_metricas['Modelo'], comparacion_metricas['R²'], 
           color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
axes[1].set_ylabel('R² (Coeficiente de Determinación)', fontsize=12)
axes[1].set_title('Comparación de R²\n(Mayor es mejor)', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(comparacion_metricas['R²']):
    axes[1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('comparacion_metricas.png', dpi=300, bbox_inches='tight')
print("   [OK] Grafico guardado: comparacion_metricas.png")
plt.show()

# ==============================================================
# 11. CONCLUSIÓN Y RESUMEN
# ==============================================================

print("\n" + "="*80)
print("CONCLUSION Y RESUMEN")
print("="*80)

print("\nCONCEPTOS DEMOSTRADOS:")
print("   1. Arbol de Decision: Modelo interpretable que divide datos con reglas simples")
print("   2. Sobreajuste: Ocurre cuando el modelo memoriza los datos de entrenamiento")
print("   3. Regularizacion: Tecnicas (max_depth, min_samples_leaf) para evitar sobreajuste")
print("   4. Random Forest: Ensamble de muchos arboles que mejora precision y robustez")
print("   5. Trade-off: Interpretabilidad vs Precision")

print("\nRESULTADOS PRINCIPALES:")
print(f"   • Arbol SIN regularizacion: R² = {r2_test_sin:.3f} (sobreajuste evidente)")
print(f"   • Arbol CON regularizacion: R² = {r2_test_con:.3f} (mejor balance)")
print(f"   • Random Forest: R² = {r2_forest:.3f} (mejor rendimiento)")

print("\nRECOMENDACIONES:")
print("   • Para interpretabilidad: Usar Arbol de Decision con regularizacion")
print("   • Para precision: Usar Random Forest")
print("   • Siempre validar con datos de prueba para detectar sobreajuste")

print("\nARCHIVOS GENERADOS:")
print("   • comparacion_sobreajuste.png")
print("   • comparacion_tres_modelos.png")
print("   • importancia_variables.png")
print("   • arbol_decision_estructura.png")
print("   • comparacion_metricas.png")

print("\n" + "="*80)
print("Fin de la demostracion de la Semana 6!")
print("="*80 + "\n")

