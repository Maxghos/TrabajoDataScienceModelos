import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import sys
import json
import pickle

# ============================================================================
# MODELO DE PREDICCIÓN DE PRECIOS DE ARRIENDO USANDO RANDOM FOREST
# ============================================================================
#
# Random Forest (Bosque Aleatorio) es un algoritmo de aprendizaje automático
# que combina múltiples árboles de decisión para hacer predicciones.
#
# ¿Cómo funciona?
# 1. Crea múltiples árboles de decisión (100 por defecto)
# 2. Cada árbol se entrena con una muestra aleatoria de los datos
# 3. Cada árbol hace una predicción independiente
# 4. El resultado final es el promedio de todas las predicciones
#
# Ventajas sobre Regresión Lineal:
# - Puede capturar relaciones NO LINEALES entre variables
# - Es más robusto ante valores atípicos (outliers)
# - Puede manejar interacciones complejas entre características
# - No asume una relación lineal entre variables (más flexible)
# - Generalmente ofrece mejor precisión en problemas reales
#
# Desventajas:
# - Más lento de entrenar que regresión lineal
# - Menos interpretable (no tiene coeficientes simples)
# - Puede sobreajustar si no se configuran bien los parámetros
#
# ============================================================================

# Configurar la codificación de salida para mostrar acentos correctamente
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# PREPARACIÓN DE DATOS
# ============================================================================

# Cargar los datos desde el archivo CSV
# Los datos deben contener información sobre propiedades en arriendo
df = pd.read_csv('DatosRevisadosLimpios.csv')

# Limpiar nombres de columnas (eliminar espacios en blanco al inicio/final)
# Esto asegura que no haya problemas al acceder a las columnas
df.columns = df.columns.str.strip()

# Seleccionar las columnas relevantes para el modelo
# Estas son las características que usaremos para predecir el precio
columnas_importantes = ['Precio ($)', 'Gastos_Comunes ($)', 'Comuna', 'Metros (m²)', 'Habitaciones', 'Baños']
df_modelo = df[columnas_importantes].copy()

# Separar la variable objetivo (lo que queremos predecir)
# En este caso, queremos predecir el precio de arriendo
y = df_modelo['Precio ($)'].values

# Preparar características numéricas
# Random Forest puede trabajar directamente con valores numéricos
# Estas variables tienen un orden y pueden ser medidas en una escala
X_numericas = df_modelo[['Metros (m²)', 'Gastos_Comunes ($)', 'Habitaciones', 'Baños']].values

# Preparar características categóricas (Comuna) usando one-hot encoding
# One-hot encoding convierte variables categóricas en variables binarias
# Por ejemplo, si hay 3 comunas, crea 3 columnas: Comuna_A, Comuna_B, Comuna_C
# Solo una de estas columnas tendrá valor 1 para cada propiedad, las demás serán 0
X_categoricas = pd.get_dummies(df_modelo['Comuna'], prefix='Comuna')

# Combinar características numéricas y categóricas en una sola matriz
# Random Forest necesita todas las características en el mismo formato (array numpy)
# El orden es importante: primero numéricas, luego categóricas
X = np.hstack([X_numericas, X_categoricas.values])

print(f"Dataset cargado: {len(df_modelo)} propiedades")
print(f"Características numéricas: {X_numericas.shape[1]}")
print(f"Características categóricas (comunas): {X_categoricas.shape[1]}")
print(f"Total de características: {X.shape[1]}")
print()

# Entrenar el modelo de Random Forest
# n_estimators: número de árboles en el bosque (más árboles = mejor rendimiento pero más tiempo)
# max_depth: profundidad máxima de los árboles (None = sin límite, puede causar sobreajuste)
# random_state: semilla para reproducibilidad
# min_samples_split: número mínimo de muestras requeridas para dividir un nodo
# min_samples_leaf: número mínimo de muestras requeridas en un nodo hoja
modelo = RandomForestRegressor(
    n_estimators=100,  # 100 árboles en el bosque
    max_depth=None,    # Sin límite de profundidad (se detiene cuando hay menos de min_samples_split)
    min_samples_split=2,  # Mínimo 2 muestras para dividir un nodo
    min_samples_leaf=1,   # Mínimo 1 muestra en una hoja
    random_state=42,      # Semilla para reproducibilidad
    n_jobs=-1            # Usar todos los núcleos del procesador para acelerar el entrenamiento
)
print("Entrenando modelo de Random Forest...")
print("  Esto puede tomar unos momentos dependiendo del tamaño del dataset...")
modelo.fit(X, y)
print("✓ Modelo entrenado correctamente")
print()

# ============================================================================
# EVALUACIÓN DEL MODELO
# ============================================================================

# Hacer predicciones con todos los datos de entrenamiento
# Nota: En un escenario real, deberíamos dividir los datos en entrenamiento y prueba
# para evaluar mejor el rendimiento. Aquí usamos todos los datos para simplificar.
y_pred = modelo.predict(X)

# Calcular métricas de rendimiento
# R² (Coeficiente de determinación): Mide qué tan bien el modelo explica la variabilidad
#   - R² = 1.0: Perfecto (el modelo explica el 100% de la variabilidad)
#   - R² = 0.0: El modelo no es mejor que simplemente predecir el promedio
#   - R² < 0.0: El modelo es peor que predecir el promedio
r2 = r2_score(y, y_pred)

# MAE (Error Absoluto Medio): Promedio de las diferencias absolutas entre predicciones y valores reales
#   - Un MAE más bajo indica mejores predicciones
#   - Se expresa en las mismas unidades que la variable objetivo (pesos chilenos)
mae = mean_absolute_error(y, y_pred)

# Precio promedio: Útil para contextualizar el MAE
precio_promedio = y.mean()

print("EVALUACIÓN DEL MODELO:")
print(f"  R²: {r2:.4f}")
print(f"  MAE: ${mae:,.2f}")
print(f"  Precio promedio: ${precio_promedio:,.2f}")
print()

# ============================================================================
# IMPORTANCIA DE CARACTERÍSTICAS (FEATURE IMPORTANCE)
# ============================================================================

# Obtener nombres de todas las características
nombres_numericas = ['Metros (m²)', 'Gastos_Comunes ($)', 'Habitaciones', 'Baños']
nombres_comunas = X_categoricas.columns.tolist()
nombres_caracteristicas = nombres_numericas + nombres_comunas

# Obtener importancia de características (feature importance) del modelo Random Forest
# 
# ¿Qué es la importancia de características?
# - Indica qué tan relevante es cada característica para hacer las predicciones
# - Se calcula basándose en cómo cada característica reduce la impureza en los árboles
# - Valores más altos = mayor importancia en la predicción del precio
# - Todas las importancias suman 1.0 (se pueden interpretar como porcentajes)
#
# Diferencias con coeficientes de regresión lineal:
# - En regresión lineal, los coeficientes pueden ser positivos o negativos
# - En Random Forest, la importancia siempre es positiva (0 a 1)
# - La importancia muestra la "contribución" de cada variable, no su "dirección"
#
importancias = modelo.feature_importances_

# Crear DataFrames separados para comunas y variables numéricas
df_importancia_comunas = pd.DataFrame({
    'Variable': [c.replace('Comuna_', '') for c in nombres_comunas],
    'Importancia': importancias[len(nombres_numericas):]
})
df_importancia_comunas = df_importancia_comunas.sort_values('Importancia', ascending=False)

df_importancia_numericas = pd.DataFrame({
    'Variable': nombres_numericas,
    'Importancia': importancias[:len(nombres_numericas)]
})
df_importancia_numericas = df_importancia_numericas.sort_values('Importancia', ascending=False)

# ============= SALIDA ORDENADA =============
print("\n")
print("RESULTADOS DEL MODELO DE RANDOM FOREST")
print("\n")

# Información sobre Random Forest
print("INFORMACIÓN DEL MODELO")
print("Algoritmo: Random Forest Regressor")
print(f"Número de árboles: {modelo.n_estimators}")
print(f"Características utilizadas: {len(nombres_caracteristicas)}")
print("\n")

# 1. Promedio de precio
print("PROMEDIO DE PRECIO")
print(f"Precio promedio de arriendo: ${precio_promedio:,.2f}")
print("\n")

# 2. R²
print("COEFICIENTE DE DETERMINACIÓN (R²)")
print(f"R²: {r2:.4f}")
print(f"Interpretación: El modelo explica el {r2*100:.2f}% de la variabilidad en los precios")
if r2 > 0.8:
    print("  → Excelente ajuste del modelo (R² > 0.8)")
elif r2 > 0.6:
    print("  → Buen ajuste del modelo (R² > 0.6)")
else:
    print("  → Ajuste moderado del modelo (R² < 0.6)")
print("\n")

# 3. MAE
print("ERROR ABSOLUTO MEDIO (MAE)")
print(f"MAE: ${mae:,.2f}")
print(f"Interpretación: En promedio, las predicciones difieren ${mae:,.2f} del precio real")
porcentaje_error = (mae / precio_promedio) * 100
print(f"  → Esto representa un error del {porcentaje_error:.2f}% respecto al precio promedio")
print("\n")

# 4. Importancia de las Comunas
print("IMPORTANCIA DE LAS COMUNAS EN LA PREDICCIÓN")
print("\nLa importancia indica qué tan relevante es cada comuna para predecir el precio.")
print("Valores más altos = mayor influencia en el precio predicho")
print("\nImportancia de cada comuna (ordenado de mayor a menor):")
print()
for _, row in df_importancia_comunas.iterrows():
    comuna = row['Variable']
    importancia = row['Importancia']
    porcentaje = (importancia / importancias.sum()) * 100
    print(f"  {comuna:<30} Importancia: {importancia:.4f} ({porcentaje:.2f}%)")
print("\n")

# 5. Importancia de las variables numéricas
print("IMPORTANCIA DE VARIABLES NUMÉRICAS")
print("\nLa importancia indica qué tan relevante es cada variable para predecir el precio.")
print("Valores más altos = mayor influencia en el precio predicho")
print("\nImportancia de cada variable numérica (ordenado de mayor a menor):")
print()
for _, row in df_importancia_numericas.iterrows():
    variable = row['Variable']
    importancia = row['Importancia']
    porcentaje = (importancia / importancias.sum()) * 100
    
    if variable == 'Metros (m²)':
        interpretacion = "Los metros cuadrados son fundamentales para determinar el precio"
    elif variable == 'Gastos_Comunes ($)':
        interpretacion = "Los gastos comunes pueden indicar la calidad y ubicación de la propiedad"
    elif variable == 'Habitaciones':
        interpretacion = "El número de habitaciones influye en el tamaño y precio de la propiedad"
    elif variable == 'Baños':
        interpretacion = "El número de baños es importante para el confort y precio"
    
    print(f"  {variable:<30} Importancia: {importancia:.4f} ({porcentaje:.2f}%)")
    print(f"    → {interpretacion}")
    print()

print("\nFIN DEL ANÁLISIS")

# ============================================================================
# VISUALIZACIÓN DEL ÁRBOL DE DECISIONES
# ============================================================================

print("\n\nGENERANDO VISUALIZACIÓN DEL ÁRBOL DE DECISIONES...")

# Seleccionar el primer árbol del Random Forest para visualización
# Random Forest tiene múltiples árboles, visualizamos uno representativo
primer_arbol = modelo.estimators_[0]

# Configurar la figura con un tamaño grande para que sea clara y legible
fig, ax = plt.subplots(figsize=(25, 15))

# Crear nombres de características más descriptivos para la visualización
nombres_visualizacion = nombres_numericas + [c.replace('Comuna_', 'Comuna: ') for c in nombres_comunas]

# Visualizar el árbol con configuración clara
# max_depth: Limitar la profundidad para que sea más legible (None = mostrar todo, pero puede ser muy grande)
# feature_names: Nombres de las características para que se muestren claramente
# filled: Rellenar con colores según el valor predicho
# rounded: Bordes redondeados para mejor apariencia
# fontsize: Tamaño de fuente para legibilidad
plot_tree(
    primer_arbol,
    max_depth=4,  # Mostrar hasta 4 niveles de profundidad (ajustable si es muy grande o pequeño)
    feature_names=nombres_visualizacion,
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    proportion=True,  # Mostrar proporciones de muestras
    impurity=False,  # No mostrar impureza para simplificar
    label='all'  # Mostrar información completa en cada nodo
)

# Agregar título descriptivo con información clara
titulo = (
    'Árbol de Decisiones - Random Forest (Primer Árbol del Bosque)\n'
    'Predicción de Precios de Arriendo\n'
    f'Total de árboles en el bosque: {modelo.n_estimators} | '
    f'Características: {len(nombres_caracteristicas)} | '
    f'Profundidad mostrada: 4 niveles\n'
    '\n'
    'Interpretación: Cada nodo muestra la condición de división, '
    'el número de muestras, el valor predicho y la proporción de muestras.'
)
plt.title(titulo, fontsize=14, fontweight='bold', pad=20)

# Ajustar el layout para que todo quepa bien
plt.tight_layout()

# Guardar la imagen en alta resolución (300 DPI para claridad)
nombre_archivo = 'arbol_decisiones.png'
plt.savefig(
    nombre_archivo,
    dpi=300,  # Alta resolución para claridad
    bbox_inches='tight',  # Ajustar bien los bordes
    facecolor='white',  # Fondo blanco
    edgecolor='none'
)

print(f"✓ Árbol de decisiones guardado en: {nombre_archivo}")
print(f"  - Tamaño de la imagen: 25x15 pulgadas (alta resolución - 300 DPI)")
print(f"  - Profundidad mostrada: 4 niveles (para mantener claridad)")
print(f"  - Información en cada nodo:")
print(f"    * Condición de división (característica y umbral)")
print(f"    * Número de muestras en el nodo")
print(f"    * Valor predicho (precio de arriendo)")
print(f"    * Proporción de muestras")
print(f"  - Nota: Este es el primer árbol de {modelo.n_estimators} árboles en el bosque")
print(f"  - Los colores indican el valor predicho (más oscuro = precio más alto)")
print()

# Cerrar la figura para liberar memoria
plt.close()

# ============= FUNCIÓN PARA HACER PREDICCIONES =============
def predecir_precio(metros, gastos_comunes, habitaciones, banos, comuna):
    """
    Predice el precio de un arriendo usando el modelo de Random Forest.
    
    Random Forest hace la predicción promediando las predicciones de múltiples árboles
    de decisión. Cada árbol analiza las características y vota sobre el precio,
    y el resultado final es el promedio de todas las predicciones.
    
    Ventajas de Random Forest sobre regresión lineal:
    - Puede capturar relaciones no lineales entre variables
    - Es más robusto ante valores atípicos (outliers)
    - Puede manejar interacciones complejas entre características
    - Generalmente ofrece mejor precisión en problemas con múltiples variables
    
    Parámetros:
    - metros: Metros cuadrados de la propiedad
    - gastos_comunes: Gastos comunes en pesos chilenos
    - habitaciones: Número de habitaciones
    - banos: Número de baños
    - comuna: Nombre de la comuna (debe estar en el dataset original)
    
    Retorna:
    - Precio predicho en pesos chilenos (promedio de todas las predicciones de los árboles)
    """
    # Preparar características numéricas en el mismo formato que se usó para entrenar
    # Es importante mantener el mismo orden: [metros, gastos_comunes, habitaciones, banos]
    caracteristicas_numericas = np.array([[metros, gastos_comunes, habitaciones, banos]])
    
    # Preparar características categóricas (comunas) usando one-hot encoding
    # Random Forest puede trabajar directamente con variables categóricas codificadas
    todas_las_comunas = X_categoricas.columns.tolist()
    comuna_features = np.zeros((1, len(todas_las_comunas)))
    
    # Buscar la columna correspondiente a la comuna y activarla (poner 1)
    # El resto de comunas quedan en 0 (one-hot encoding)
    comuna_col = f'Comuna_{comuna}'
    if comuna_col in todas_las_comunas:
        idx = todas_las_comunas.index(comuna_col)
        comuna_features[0, idx] = 1
    else:
        print(f"Advertencia: La comuna '{comuna}' no se encontró en los datos de entrenamiento.")
        print(f"Comunas disponibles: {[c.replace('Comuna_', '') for c in todas_las_comunas]}")
        return None
    
    # Combinar características numéricas y categóricas en el mismo orden que durante el entrenamiento
    # Este orden es crítico: primero numéricas, luego categóricas (comunas)
    caracteristicas = np.hstack([caracteristicas_numericas, comuna_features])
    
    # Realizar la predicción usando Random Forest
    # El modelo promedia las predicciones de todos los árboles (100 árboles por defecto)
    # Cada árbol analiza las características y predice un precio, y el resultado final es el promedio
    precio_predicho = modelo.predict(caracteristicas)[0]
    
    return precio_predicho

# ============= GUARDAR DATOS DEL MODELO =============
print("\n\nGUARDANDO DATOS DEL MODELO...")

# Guardar el modelo entrenado de Random Forest
# El archivo contiene el modelo completo con todos los árboles entrenados
with open('modelo_random_forest.pkl', 'wb') as f:
    pickle.dump(modelo, f)

# Guardar información del modelo en JSON para usar en HTML
# Random Forest no tiene intercepto como la regresión lineal, pero tiene parámetros del modelo
datos_modelo = {
    'tipo_modelo': 'RandomForestRegressor',
    'parametros': {
        'n_estimators': int(modelo.n_estimators),
        'max_depth': modelo.max_depth if modelo.max_depth else 'None',
        'min_samples_split': int(modelo.min_samples_split),
        'min_samples_leaf': int(modelo.min_samples_leaf)
    },
    'metricas': {
        'precio_promedio': float(precio_promedio),
        'r2': float(r2),
        'mae': float(mae)
    },
    'importancias_comunas': df_importancia_comunas.to_dict('records'),
    'importancias_numericas': df_importancia_numericas.to_dict('records'),
    'comunas_disponibles': [c.replace('Comuna_', '') for c in nombres_comunas],
    'nombres_caracteristicas': nombres_caracteristicas
}

with open('datos_modelo.json', 'w', encoding='utf-8') as f:
    json.dump(datos_modelo, f, ensure_ascii=False, indent=2)

# Guardar las columnas de las comunas para usar en predicciones
# Esto es necesario para mantener el mismo orden de características al hacer predicciones
with open('comunas_columnas.pkl', 'wb') as f:
    pickle.dump(X_categoricas.columns.tolist(), f)

print("✓ Modelo guardado en: modelo_random_forest.pkl")
print("✓ Datos del modelo guardados en: datos_modelo.json")
print("✓ Columnas de comunas guardadas en: comunas_columnas.pkl")
print("\nNota: Random Forest puede ofrecer mejores predicciones que regresión lineal")
print("      porque puede capturar relaciones no lineales entre las variables.")

# ============= INTERFAZ PARA PREDICCIONES =============
print("\n\n" + "="*60)
print("PREDICTOR DE PRECIOS DE ARRIENDO")
print("="*60)

def solicitar_prediccion():
    """Solicita datos al usuario y realiza una predicción"""
    print("\nIngrese los datos de la propiedad:")
    print()
    
    try:
        metros = float(input("Metros cuadrados (m²): "))
        gastos_comunes = float(input("Gastos comunes ($): "))
        habitaciones = int(input("Número de habitaciones: "))
        banos = int(input("Número de baños: "))
        
        print("\nComunas disponibles:")
        comunas_disponibles = sorted([c.replace('Comuna_', '') for c in nombres_comunas])
        for i, comuna in enumerate(comunas_disponibles, 1):
            print(f"  {i}. {comuna}")
        
        entrada_comuna = input("\nIngrese el número o nombre de la comuna: ").strip()
        
        # Verificar si ingresó un número o el nombre
        try:
            numero = int(entrada_comuna)
            if 1 <= numero <= len(comunas_disponibles):
                comuna = comunas_disponibles[numero - 1]
            else:
                print(f"Error: Ingrese un número entre 1 y {len(comunas_disponibles)}")
                return True
        except ValueError:
            # Si no es un número, asumir que es el nombre de la comuna
            comuna = entrada_comuna
        
        precio_predicho = predecir_precio(metros, gastos_comunes, habitaciones, banos, comuna)
        
        if precio_predicho is not None:
            print("\n" + "-"*60)
            print("RESULTADO DE LA PREDICCIÓN")
            print("-"*60)
            print(f"Metros cuadrados: {metros} m²")
            print(f"Gastos comunes: ${gastos_comunes:,.2f}")
            print(f"Habitaciones: {habitaciones}")
            print(f"Baños: {banos}")
            print(f"Comuna: {comuna}")
            print()
            print(f"PRECIO PREDICHO: ${precio_predicho:,.2f}")
            print("-"*60)
            
            # Guardar la predicción
            prediccion_data = {
                'metros': metros,
                'gastos_comunes': gastos_comunes,
                'habitaciones': habitaciones,
                'banos': banos,
                'comuna': comuna,
                'precio_predicho': float(precio_predicho)
            }
            
            # Cargar predicciones anteriores si existen
            try:
                with open('predicciones.json', 'r', encoding='utf-8') as f:
                    predicciones = json.load(f)
            except FileNotFoundError:
                predicciones = []
            
            predicciones.append(prediccion_data)
            
            with open('predicciones.json', 'w', encoding='utf-8') as f:
                json.dump(predicciones, f, ensure_ascii=False, indent=2)
            
            print("\n✓ Predicción guardada en: predicciones.json")
        
    except ValueError as e:
        print(f"\nError: Ingrese valores numéricos válidos. {e}")
    except KeyboardInterrupt:
        print("\n\nPredicción cancelada.")
        return False
    
    return True

# Preguntar si desea hacer predicciones
while True:
    respuesta = input("\n¿Desea predecir el precio de una propiedad? (s/n): ").strip().lower()
    
    if respuesta == 's':
        if not solicitar_prediccion():
            break
    elif respuesta == 'n':
        print("\nGracias por usar el predictor de precios.")
        break
    else:
        print("Por favor, ingrese 's' para sí o 'n' para no.")

print("\n" + "="*60)
print("Programa finalizado")
print("="*60)