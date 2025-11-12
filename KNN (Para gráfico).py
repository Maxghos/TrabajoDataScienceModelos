import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys
import json
import pickle

# Configurar la codificación de salida para mostrar acentos correctamente
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Cargar los datos
df = pd.read_csv('DatosRevisadosLimpios.csv')

# Limpiar nombres de columnas
df.columns = df.columns.str.strip()

# Seleccionar las columnas importantes
columnas_importantes = ['Precio ($)', 'Gastos_Comunes ($)', 'Comuna', 'Metros (m²)', 'Habitaciones', 'Baños']
df_modelo = df[columnas_importantes].copy()

# Separar variable objetivo
y_precio = df_modelo['Precio ($)'].values
precio_promedio = y_precio.mean()

# Crear variable de clasificación: 1 si precio >= media (alto), 0 si precio < media (bajo)
y_clasificacion = (y_precio >= precio_promedio).astype(int)

# Preparar características numéricas
X_numericas = df_modelo[['Metros (m²)', 'Gastos_Comunes ($)', 'Habitaciones', 'Baños']].values

# Preparar características categóricas (Comuna) usando one-hot encoding
X_categoricas = pd.get_dummies(df_modelo['Comuna'], prefix='Comuna')

# Combinar características numéricas y categóricas
X = np.hstack([X_numericas, X_categoricas.values])

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train_clas, y_test_clas, y_train_precio, y_test_precio = train_test_split(
    X, y_clasificacion, y_precio, test_size=0.2, random_state=42
)

# Entrenar el modelo de clasificación KNN
modelo_clasificacion = KNeighborsClassifier(n_neighbors=5)
modelo_clasificacion.fit(X_train, y_train_clas)

# Entrenar el modelo de regresión KNN para predecir el precio numérico
modelo_regresion = KNeighborsRegressor(n_neighbors=5)
modelo_regresion.fit(X_train, y_train_precio)

# Hacer predicciones con todos los datos
y_pred_clasificacion = modelo_clasificacion.predict(X)

# Calcular métricas de clasificación
accuracy = accuracy_score(y_clasificacion, y_pred_clasificacion)
precision = precision_score(y_clasificacion, y_pred_clasificacion, average='weighted')
recall = recall_score(y_clasificacion, y_pred_clasificacion, average='weighted')
f1 = f1_score(y_clasificacion, y_pred_clasificacion, average='weighted')
matriz_confusion = confusion_matrix(y_clasificacion, y_pred_clasificacion)

# Obtener nombres de todas las características
nombres_numericas = ['Metros (m²)', 'Gastos_Comunes ($)', 'Habitaciones', 'Baños']
nombres_comunas = X_categoricas.columns.tolist()
nombres_caracteristicas = nombres_numericas + nombres_comunas

# ============= SALIDA ORDENADA =============
print("\n" + "="*70)
print(" " * 15 + "RESULTADOS DEL MODELO KNN")
print(" " * 10 + "(K-Nearest Neighbors - Clasificación)")
print("="*70 + "\n")

# 1. Promedio de precio
print("─" * 70)
print(" PROMEDIO DE PRECIO")
print("─" * 70)
print(f"  Precio promedio de arriendo: ${precio_promedio:,.2f}")
print(f"  Clasificación:")
print(f"    • Precios >= ${precio_promedio:,.2f} → 'Precio Alto'")
print(f"    • Precios <  ${precio_promedio:,.2f} → 'Precio Bajo'")
print()

# 2. Métricas de Clasificación
print("─" * 70)
print(" MÉTRICAS DE CLASIFICACIÓN")
print("─" * 70)
print(f"  Accuracy (Precisión Global):  {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"    → Porcentaje de predicciones correctas")
print()
print(f"  Precision (Precisión):        {precision:.4f}  ({precision*100:.2f}%)")
print(f"    → De las predicciones positivas, cuántas son correctas")
print()
print(f"  Recall (Sensibilidad):        {recall:.4f}  ({recall*100:.2f}%)")
print(f"    → De los casos reales, cuántos identifica correctamente")
print()
print(f"  F1-Score:                     {f1:.4f}  ({f1*100:.2f}%)")
print(f"    → Media armónica entre Precision y Recall")
print()

# 3. Matriz de Confusión
print("─" * 70)
print(" MATRIZ DE CONFUSIÓN")
print("─" * 70)
print(" " * 20 + "PREDICHO")
print(" " * 15 + "Bajo" + " " * 8 + "Alto")
print(f"  REAL    Bajo    {matriz_confusion[0,0]:4d}    {matriz_confusion[0,1]:4d}")
print(f"          Alto    {matriz_confusion[1,0]:4d}    {matriz_confusion[1,1]:4d}")
print()

# 4. Información del modelo
print("─" * 70)
print(" CONFIGURACIÓN DEL MODELO")
print("─" * 70)
print(f"  Modelo:              K-Nearest Neighbors (KNN)")
print(f"  Tipo:                Clasificación")
print(f"  Número de vecinos:   5")
print(f"  Características:     {len(nombres_caracteristicas)}")
print(f"    • Variables numéricas:  {len(nombres_numericas)}")
print(f"    • Comunas (one-hot):     {len(nombres_comunas)}")
print()

print("="*70)
print(" " * 25 + "FIN DEL ANÁLISIS")
print("="*70)

# ============= FUNCIÓN PARA HACER PREDICCIONES =============
def predecir_clasificacion(metros, gastos_comunes, habitaciones, banos, comuna):
    """
    Predice la clasificación y el precio de un arriendo basándose en las características proporcionadas.
    
    Parámetros:
    - metros: Metros cuadrados
    - gastos_comunes: Gastos comunes en pesos chilenos
    - habitaciones: Número de habitaciones
    - banos: Número de baños
    - comuna: Nombre de la comuna (debe estar en el dataset original)
    
    Retorna:
    - Diccionario con 'clasificacion' (str) y 'precio_predicho' (float)
    """
    # Crear array con características numéricas
    caracteristicas_numericas = np.array([[metros, gastos_comunes, habitaciones, banos]])
    
    # Crear array para características categóricas (comunas)
    todas_las_comunas = X_categoricas.columns.tolist()
    comuna_features = np.zeros((1, len(todas_las_comunas)))
    
    # Buscar la columna correspondiente a la comuna
    comuna_col = f'Comuna_{comuna}'
    if comuna_col in todas_las_comunas:
        idx = todas_las_comunas.index(comuna_col)
        comuna_features[0, idx] = 1
    else:
        print(f"Advertencia: La comuna '{comuna}' no se encontró en los datos.")
        print(f"Comunas disponibles: {[c.replace('Comuna_', '') for c in todas_las_comunas]}")
        return None
    
    # Combinar características
    caracteristicas = np.hstack([caracteristicas_numericas, comuna_features])
    
    # Predecir clasificación (precio alto o bajo)
    clasificacion_predicha = modelo_clasificacion.predict(caracteristicas)[0]
    clasificacion_texto = "Precio Alto" if clasificacion_predicha == 1 else "Precio Bajo"
    
    # Predecir precio numérico
    precio_predicho = modelo_regresion.predict(caracteristicas)[0]
    
    return {
        'clasificacion': clasificacion_texto,
        'precio_predicho': precio_predicho
    }

# ============= GUARDAR DATOS DEL MODELO =============
print("\n\n" + "─" * 70)
print(" GUARDANDO DATOS DEL MODELO...")
print("─" * 70)

# Guardar los modelos entrenados
with open('modelo_clasificacion_knn.pkl', 'wb') as f:
    pickle.dump(modelo_clasificacion, f)

with open('modelo_regresion_knn.pkl', 'wb') as f:
    pickle.dump(modelo_regresion, f)

# Guardar información del modelo en JSON para usar en HTML
datos_modelo = {
    'metricas': {
        'precio_promedio': float(precio_promedio),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    'comunas_disponibles': [c.replace('Comuna_', '') for c in nombres_comunas],
    'nombres_caracteristicas': nombres_caracteristicas,
    'modelo': 'KNN Clasificación',
    'n_neighbors': 5
}

with open('datos_modelo.json', 'w', encoding='utf-8') as f:
    json.dump(datos_modelo, f, ensure_ascii=False, indent=2)

# Guardar las columnas de las comunas para usar en predicciones
with open('comunas_columnas.pkl', 'wb') as f:
    pickle.dump(X_categoricas.columns.tolist(), f)

print("  ✓ Modelo de clasificación guardado en: modelo_clasificacion_knn.pkl")
print("  ✓ Modelo de regresión guardado en: modelo_regresion_knn.pkl")
print("  ✓ Datos del modelo guardados en: datos_modelo.json")
print("  ✓ Columnas de comunas guardadas en: comunas_columnas.pkl")
print()

# ============= INTERFAZ PARA PREDICCIONES =============
print("\n\n" + "="*70)
print(" " * 20 + "PREDICTOR DE PRECIOS DE ARRIENDO")
print("="*70)

def solicitar_prediccion():
    """Solicita datos al usuario y realiza una predicción"""
    print("\n" + "─" * 70)
    print(" INGRESE LOS DATOS DE LA PROPIEDAD")
    print("─" * 70 + "\n")
    
    try:
        metros = float(input("  Metros cuadrados (m²): "))
        gastos_comunes = float(input("  Gastos comunes ($): "))
        habitaciones = int(input("  Número de habitaciones: "))
        banos = int(input("  Número de baños: "))
        
        print("\n  Comunas disponibles:")
        comunas_disponibles = sorted([c.replace('Comuna_', '') for c in nombres_comunas])
        for i, comuna in enumerate(comunas_disponibles, 1):
            print(f"    {i:2d}. {comuna}")
        
        entrada_comuna = input("\n  Ingrese el número o nombre de la comuna: ").strip()
        
        # Verificar si ingresó un número o el nombre
        try:
            numero = int(entrada_comuna)
            if 1 <= numero <= len(comunas_disponibles):
                comuna = comunas_disponibles[numero - 1]
            else:
                print(f"\n  ✗ Error: Ingrese un número entre 1 y {len(comunas_disponibles)}")
                return True
        except ValueError:
            # Si no es un número, asumir que es el nombre de la comuna
            comuna = entrada_comuna
        
        resultado = predecir_clasificacion(metros, gastos_comunes, habitaciones, banos, comuna)
        
        if resultado is not None:
            clasificacion = resultado['clasificacion']
            precio_predicho = resultado['precio_predicho']
            
            print("\n" + "="*70)
            print(" " * 20 + "RESULTADO DE LA PREDICCIÓN")
            print("="*70)
            print("\n" + "─" * 70)
            print(" DATOS DE LA PROPIEDAD")
            print("─" * 70)
            print(f"  Metros cuadrados:     {metros:>10.2f} m²")
            print(f"  Gastos comunes:       ${gastos_comunes:>10,.2f}")
            print(f"  Habitaciones:         {habitaciones:>10}")
            print(f"  Baños:                {banos:>10}")
            print(f"  Comuna:                {comuna:>10}")
            
            print("\n" + "─" * 70)
            print(" PREDICCIÓN")
            print("─" * 70)
            print(f"  Precio predicho:       ${precio_predicho:>10,.2f}")
            print(f"  Clasificación:         {clasificacion:>10}")
            print(f"  Precio promedio:       ${precio_promedio:>10,.2f}")
            
            diferencia = precio_predicho - precio_promedio
            if diferencia >= 0:
                print(f"  Diferencia:            ${diferencia:>10,.2f} (por encima del promedio)")
            else:
                print(f"  Diferencia:            ${abs(diferencia):>10,.2f} (por debajo del promedio)")
            
            print("─" * 70)
            
            # Guardar la predicción
            prediccion_data = {
                'metros': metros,
                'gastos_comunes': gastos_comunes,
                'habitaciones': habitaciones,
                'banos': banos,
                'comuna': comuna,
                'precio_predicho': float(precio_predicho),
                'clasificacion': clasificacion
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
            
            print("\n  ✓ Predicción guardada en: predicciones.json")
            print("="*70 + "\n")
        
    except ValueError as e:
        print(f"\n  ✗ Error: Ingrese valores numéricos válidos. {e}")
    except KeyboardInterrupt:
        print("\n\n  ✗ Predicción cancelada.")
        return False
    
    return True

# Preguntar si desea hacer predicciones
while True:
    respuesta = input("\n  ¿Desea predecir el precio de una propiedad? (s/n): ").strip().lower()
    
    if respuesta == 's':
        if not solicitar_prediccion():
            break
    elif respuesta == 'n':
        print("\n" + "="*70)
        print(" " * 25 + "¡Gracias por usar el predictor!")
        print("="*70)
        break
    else:
        print("  ✗ Por favor, ingrese 's' para sí o 'n' para no.")

print("\n" + "="*70)
print(" " * 28 + "PROGRAMA FINALIZADO")
print("="*70)