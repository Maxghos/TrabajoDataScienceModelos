import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

#----------- Cargar Datos ------------------
dt = pl.read_csv("DatosRevisadosLimpios.csv")
data = pl.DataFrame(dt)

#----------- Filtro para acotar los datos ------------------
data = data.filter(
    (pl.col("Precio ($)") >= 100000) &
    (pl.col("Precio ($)") <= 2000000) &
    (pl.col("Metros (m²)") >= 15) &
    (pl.col("Metros (m²)") <= 200)
)

#----------- Crear columnas para todas las comunas ------------------
data = data.to_dummies(columns=["Comuna"]) 
comuna_cols = [col for col in data.columns if col.startswith("Comuna_")]


#----------- Selección De Variables ------------------
xx = data.select([
    "Gastos_Comunes ($)",
    "Metros (m²)",
    "Habitaciones",
    "Baños",
    *comuna_cols
]).to_numpy()

yy = data.select(pl.col("Precio ($)")).to_numpy().ravel()

#----------- Separar y entrenar los datos ------------------
x_train,x_test,y_train,y_test = train_test_split(xx,yy, test_size=0.3,random_state=42)

#----------- Escalado de Datos ------------------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#----------- Crear y definir parámetros en Red Neuronal ------------------
modeloUsar = MLPRegressor(
    hidden_layer_sizes=(40, 20),   # Capas Ocultas (Dos Capas) -> 40 y 20 neuronas
    activation='relu',             # Función de activación moderna relu
    max_iter=7000,                 # N° De Iteraciones
    random_state=42
)


#----------- Entrenamiento De Datos ------------------
modeloUsar.fit(x_train,y_train)


#----------- Metricas Arrojadas (Evaluación) ------------------
predic = modeloUsar.predict(x_test)
print("\nR2 -> ", r2_score(y_test, predic))
print("MSE -> ", mean_squared_error(y_test, predic))
print("MAE -> ", mean_absolute_error(y_test,predic))
print("RMSE -> ", np.sqrt(mean_squared_error(y_test, predic)))


#----------- Gráfico 1 ------------------
#Caja Negra (Demo Rápida en MatPlotLib para demostración)
plt.scatter(y_test, predic, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Predicho vs Real")
plt.show()


#----------- Predicción Manual para probar modelo ------------------
print("\n----------- Inicio Modelo -----------")
gastosComun = int(input("\nIngrese gastos comunes aproximados -> "))
metrosC = int(input("\nIngrese metros cuadrados del arriendo -> "))
habitaciones = int(input("\nIngrese las habitaciones para el arriendo -> "))
baños = int(input("\nIngrese los baños para el arriendo -> "))
comuna = input("\nIngrese la comuna -> ").strip().title() #Solo toma el coso de titulo antes de columna



#----------- Agregar Columnas a dummies------------------
comuna_dummies = [1 if col == f"Comuna_{comuna}" else 0 for col in comuna_cols]


#----------- Advertencia en caso de Comuna no encontrada/existente ------------------
if sum(comuna_dummies) == 0: 
    print(f"\n Advertencia: la comuna '{comuna}' no existe en los datos. Se asumirá base (todas 0).")


#----------- Datos a variable (recientes) ------------------
predikkk = [gastosComun, metrosC, habitaciones, baños] + comuna_dummies


#----------- Escalado De Datos Recientes ------------------
predikkk_scaled = scaler.transform([predikkk])



#----------- Prediccion con datos recientes ------------------
ff = modeloUsar.predict(predikkk_scaled)
print(f"\nSu arriendo en {comuna} tendra el valor de -> ${round(ff[0],0)}") #Round pa redondear y ya evitar que aparezcan decimales







#Prueba Sugerida
#Gastos_Comunes -> 100000
#Metros -> 58
#Habitaciones -> 2
#Baños -> 2
#Comuna -> Santiago

#Resultado Esperado -> $390000
#Resultado Arrojado (Probado) -> $449157

#Analisis -> Probablemente este modelo no es el adecuado para una precisión completa, pero 
#se acerca al precio real al que deberia de haber arrojado (No se como comprobar si en verdad
#el modelo esta funcionando o no)
