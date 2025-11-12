import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
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
y = df_modelo['Precio ($)'].values

# Preparar características numéricas
X_numericas = df_modelo[['Metros (m²)', 'Gastos_Comunes ($)', 'Habitaciones', 'Baños']].values

# Preparar características categóricas (Comuna) usando one-hot encoding
X_categoricas = pd.get_dummies(df_modelo['Comuna'], prefix='Comuna')

# Combinar características numéricas y categóricas
X = np.hstack([X_numericas, X_categoricas.values])

# Entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Hacer predicciones con todos los datos
y_pred = modelo.predict(X)

# Calcular métricas
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
precio_promedio = y.mean()

# Obtener nombres de todas las características
nombres_numericas = ['Metros (m²)', 'Gastos_Comunes ($)', 'Habitaciones', 'Baños']
nombres_comunas = X_categoricas.columns.tolist()
nombres_caracteristicas = nombres_numericas + nombres_comunas

# Obtener coeficientes del modelo
coeficientes = modelo.coef_

# Crear DataFrames separados para comunas y variables numéricas
df_coef_comunas = pd.DataFrame({
    'Variable': [c.replace('Comuna_', '') for c in nombres_comunas],
    'Coeficiente': coeficientes[len(nombres_numericas):]
})
df_coef_comunas = df_coef_comunas.sort_values('Coeficiente', ascending=False)

df_coef_numericas = pd.DataFrame({
    'Variable': nombres_numericas,
    'Coeficiente': coeficientes[:len(nombres_numericas)]
})

# ============= SALIDA ORDENADA =============
print("\n")
print("RESULTADOS DEL MODELO DE REGRESIÓN LINEAL")
print("\n")

# 1. Promedio de precio
print("PROMEDIO DE PRECIO")
print(f"Precio promedio de arriendo: ${precio_promedio:,.2f}")
print("\n")

# 2. R²
print("COEFICIENTE DE DETERMINACIÓN (R²)")
print(f"R²: {r2:.4f}")
print(f"Interpretación: El modelo explica el {r2*100:.2f}% de la variabilidad en los precios")
print("\n")

# 3. MAE
print("ERROR ABSOLUTO MEDIO (MAE)")
print(f"MAE: ${mae:,.2f}")
print(f"Interpretación: En promedio, las predicciones difieren ${mae:,.2f} del precio real")
print("\n")

# 4. Coeficientes de las Comunas
print("COEFICIENTES POR COMUNA")
print(f"Intercepto base del modelo: ${modelo.intercept_:,.2f}")
print("\nEfecto de cada comuna sobre el precio (ordenado de mayor a menor impacto):")
print()
for _, row in df_coef_comunas.iterrows():
    comuna = row['Variable']
    coef = row['Coeficiente']
    signo = "+" if coef >= 0 else ""
    print(f"  {comuna:<30} {signo}${coef:,.2f}")
print("\n")

# 5. Coeficientes de las demás variables
print("COEFICIENTES DE VARIABLES NUMÉRICAS")
print("\nEfecto de cada variable numérica sobre el precio:")
print()
for _, row in df_coef_numericas.iterrows():
    variable = row['Variable']
    coef = row['Coeficiente']
    
    if variable == 'Metros (m²)':
        interpretacion = f"por cada metro cuadrado adicional"
    elif variable == 'Gastos_Comunes ($)':
        interpretacion = f"por cada peso adicional en gastos comunes"
    elif variable == 'Habitaciones':
        interpretacion = f"por cada habitación adicional"
    elif variable == 'Baños':
        interpretacion = f"por cada baño adicional"
    
    print(f"  {variable:<30} ${coef:,.2f}")
    print(f"    → {interpretacion}")
    print()

print("\nFIN DEL ANÁLISIS")

# ============= FUNCIÓN PARA HACER PREDICCIONES =============
def predecir_precio(metros, gastos_comunes, habitaciones, banos, comuna):
    """
    Predice el precio de un arriendo basándose en las características proporcionadas.
    
    Parámetros:
    - metros: Metros cuadrados
    - gastos_comunes: Gastos comunes en pesos chilenos
    - habitaciones: Número de habitaciones
    - banos: Número de baños
    - comuna: Nombre de la comuna (debe estar en el dataset original)
    
    Retorna:
    - Precio predicho en pesos chilenos
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
    
    # Predecir
    precio_predicho = modelo.predict(caracteristicas)[0]
    
    return precio_predicho

# ============= GUARDAR DATOS DEL MODELO =============
print("\n\nGUARDANDO DATOS DEL MODELO...")

# Guardar el modelo entrenado
with open('modelo_regresion.pkl', 'wb') as f:
    pickle.dump(modelo, f)

# Guardar información del modelo en JSON para usar en HTML
datos_modelo = {
    'metricas': {
        'precio_promedio': float(precio_promedio),
        'r2': float(r2),
        'mae': float(mae),
        'intercepto': float(modelo.intercept_)
    },
    'coeficientes_comunas': df_coef_comunas.to_dict('records'),
    'coeficientes_numericas': df_coef_numericas.to_dict('records'),
    'comunas_disponibles': [c.replace('Comuna_', '') for c in nombres_comunas],
    'nombres_caracteristicas': nombres_caracteristicas
}

with open('datos_modelo.json', 'w', encoding='utf-8') as f:
    json.dump(datos_modelo, f, ensure_ascii=False, indent=2)

# Guardar las columnas de las comunas para usar en predicciones
with open('comunas_columnas.pkl', 'wb') as f:
    pickle.dump(X_categoricas.columns.tolist(), f)

print("✓ Modelo guardado en: modelo_regresion.pkl")
print("✓ Datos del modelo guardados en: datos_modelo.json")
print("✓ Columnas de comunas guardadas en: comunas_columnas.pkl")

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