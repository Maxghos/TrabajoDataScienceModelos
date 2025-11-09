#----------- Librerias a Usar ------------------
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression


#----------- Datos del Csv ------------------
dt = pl.read_csv("DatosRevisadosLimpios.csv")
data = pl.DataFrame(dt)


#----------- Crear columnas para todas las comunas ------------------
data = data.to_dummies(columns=["Comuna"]) #Se usa el dummies como en https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_dummies.html
comuna_cols = [col for col in data.columns if col.startswith("Comuna_")]
#Esto de acá crea una lista con los nombres de todas las columnas dummies de comuna que se generaron en la variable anterior
#Lo que se usará despues para poder simplificar la busqueda de la comuna en booleano (Según logica de to_dummies)


#----------- Extracción De Datos por columnas ------------------
xx = data.select([
    "Gastos_Comunes ($)",
    "Metros (m²)",
    "Habitaciones",
    "Baños",
    *[col for col in data.columns if col.startswith("Comuna_")]
    #Con el asterisco, desempaqueta o tambien extrae todas las comunas que comienzen con "Comuna_" en vez de extraer toda la columna "Comuna"
]).to_numpy()

yy = data.select(pl.col("Precio ($)")).to_numpy().ravel()


#----------- Division y entrenamiento de datos ------------------
x_train,x_test,y_train,y_test = train_test_split(xx,yy, test_size=0.3,random_state=42)


#----------- Modelo y entrenamiento ------------------
modeloUsar = LinearRegression()
modeloUsar.fit(x_train,y_train)


#----------- Predicción usando Modelo ------------------
predic = modeloUsar.predict(x_test)


#----------- Métricas Arrojadas ------------------
print("\nR2 -> ", r2_score(y_test, predic))
print("MSE -> ", mean_squared_error(y_test, predic))
print("MAE -> ", mean_absolute_error(y_test,predic))
print("RMSE -> ", np.sqrt(mean_squared_error(y_test, predic)))


#----------- Gráfico 1 ------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predic, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel("Precio Real ($)")
plt.ylabel("Precio Predicho ($)")
plt.title("Comparación entre Precio Real y Precio Predicho")
plt.grid(True)
plt.show()


#----------- Gráfico 2 ------------------
residuos = y_test - predic

plt.figure(figsize=(8, 6))
plt.scatter(predic, residuos, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Precio Predicho ($)")
plt.ylabel("Error (Residual)")
plt.title("Distribución de Errores del Modelo")
plt.grid(True)
plt.show()


#----------- Datos a agregar para probar modelo ------------------
print("\n----------- Inicio Modelo -----------")
gastosComun = int(input("\nIngrese gastos comunes aproximados -> "))
metrosC = int(input("\nIngrese metros cuadrados del arriendo -> "))
habitaciones = int(input("\nIngrese las habitaciones para el arriendo -> "))
baños = int(input("\nIngrese los baños para el arriendo -> "))
comuna = input("\nIngrese la comuna -> ").strip().title() #Solo toma el coso de titulo antes de columna



#----------- Agregar Columna a dummies ------------------
#Aquí basicamente agrega la comuna mencionada en la comuna de dummies (que empieza por Comuna_) para buscar y checar la comuna que se eligió (se puede ver imprimiento la variable predikkk)
comuna_dummies = [1 if col == f"Comuna_{comuna}" else 0 for col in comuna_cols]


#----------- Advertencia en caso de Comuna no encontrada/existente ------------------
#Esto es para evitar que si coloca una wea mal (como otra comuna o algo nada que ver) // Fuente: Chatgpgoat
if sum(comuna_dummies) == 0: 
    print(f"\n Advertencia: la comuna '{comuna}' no existe en los datos. Se asumirá base (todas 0).")



#----------- Datos a variable (recientes) ------------------
predikkk = [gastosComun, metrosC, habitaciones, baños] + comuna_dummies


#----------- prediccion con datos recientes ------------------
ff = modeloUsar.predict([predikkk])
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
