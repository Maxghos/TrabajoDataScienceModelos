import polars as pl

#||||||||||||------------DATOS PRINCIPALES----------||||||||||||| (LISTO)

astaCsv = pl.read_csv("ArriendosConValoresParaLimpiar.csv")
data = pl.DataFrame(astaCsv) #<-

data = data.with_columns([
    pl.when(pl.col("Precio").str.starts_with(r"USD")).then(pl.lit(None, dtype=pl.Utf8)).otherwise(pl.col("Precio")).alias("Precio")
])

data = data.drop_nulls()
print(len(data))

#AHORA HAY VALORES SIN USD
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


#||||||||||||------------LIMPIAR PRECIOS----------||||||||||||| (LISTO)


valorUF = 39597.67

limpiarPrecio1 = data.filter(pl.col("Precio").str.contains(r"\$"))
limpiarPrecio2 = data.filter(pl.col("Precio").str.contains(r"UF"))
#limpiarPrecio3 = data.filter(pl.col("Precio").str.contains(r"USD"))


#-----Limpiar Precios  con $$$$$$

limpiarPrecio1 = limpiarPrecio1.with_columns([
    pl.col("Precio").str.replace(r"\$","").str.replace(r"\.","").str.strip_chars().cast(pl.Float64)
])
limpiarPrecio1 = limpiarPrecio1.with_columns([
    pl.col("Precio").round(0).cast(pl.Int64)
])
print(len(limpiarPrecio1))


#-----Limpiar Precios  con UFUFUFUFUF 8ERROR8


limpiarPrecio2 = limpiarPrecio2.with_columns([
    pl.col("Precio").str.replace(r"UF","").str.replace(r"\.","").str.replace(r"\,",".").str.strip_chars().cast(pl.Float64)*valorUF
])

limpiarPrecio2 = limpiarPrecio2.with_columns([
    pl.col("Precio").round(0).cast(pl.Int64)
]) #<- Arreglar DATOS PASADOS DE PRECIO // Hay precios pasados de demasiados numeros, arreglar la conversion de UF

limpiarPrecio2 = limpiarPrecio2.with_columns([
    pl.when((pl.col("Precio") > 5000000) & (pl.col("Precio") < 100000000)).then(pl.col("Precio")/100)
    .when((pl.col("Precio") > 100000000) & (pl.col("Precio") < 1000000000)).then(pl.col("Precio")/100)
    .when((pl.col("Precio") > 1000000000) & (pl.col("Precio") < 15000000000)).then(pl.col("Precio")/10000)
    .otherwise(pl.col("Precio"))
])

limpiarPrecio2 = limpiarPrecio2.with_columns([
    pl.col("Precio").round(0).cast(pl.Int64)
]) 

print(len(limpiarPrecio2)) 


concatenarDatos1 = pl.concat([limpiarPrecio1,limpiarPrecio2], how="vertical")
#print(concatenarDatos1)

#DATOS HASTA AQUÍ TDO BIEN YA QUE LOS DATOS ESTAN BIEN CONCATENADOS


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#||||||||||||------------LIMPIAR GASTOS_COMUNES----------||||||||||||| (ERRORES)

concatenarDatos1 = concatenarDatos1.filter(pl.col("Gastos_Comunes").str.contains(r"\$"))

concatenarDatos1 = concatenarDatos1.with_columns([
    pl.col("Gastos_Comunes").str.replace(r"\$","").str.replace(r"\.","").str.strip_chars().cast(pl.Float64)
])

concatenarDatos1 = concatenarDatos1.with_columns([
    pl.col("Gastos_Comunes").round(0).cast(pl.Int64)
])



concatenarDatos1 = concatenarDatos1.with_columns([
    pl.when(pl.col("Gastos_Comunes") > 1000000).then(pl.col("Gastos_Comunes")/10)
    .when((pl.col("Gastos_Comunes") < 25) & (pl.col("Gastos_Comunes") > 2)).then(pl.col("Gastos_Comunes")*10000)
    .when(pl.col("Gastos_Comunes") < 3).then(pl.col("Gastos_Comunes")*100000)
    .when((pl.col("Gastos_Comunes") > 30) & (pl.col("Gastos_Comunes") < 300)).then(pl.col("Gastos_Comunes")*1000)
    .otherwise(pl.col("Gastos_Comunes"))
    .alias("Gastos_Comunes").cast(pl.Int64)
])

concatenarDatos1 = concatenarDatos1.with_columns([
    pl.col("Gastos_Comunes").round(0).cast(pl.Int64)
])


print(len(concatenarDatos1))
#DATOS HASTA AQUÍ TDO BIEN YA QUE LOS DATOS ESTAN BIEN CONCATENADOS


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#||||||||||||------------UBICACION SEPARACIÓN----------||||||||||||| (LISTO)

concatenarDatos1 = concatenarDatos1.with_columns([
    pl.col("Ubicacion").str.extract(r"^(.*?),").alias("Comuna") # Esto de acá ^(.*?) signficia que captura todo de forma no conficional desde el principio de la cadena
])

concatenarDatos1 = concatenarDatos1.rename({"Ubicacion": "Direccion"})

nuevoOrden = ["Nombre", "Precio", "Gastos_Comunes", "Comuna", "Direccion", "Metros", "Habitaciones", "Baños"]
concatenarDatos1 = concatenarDatos1.select(nuevoOrden)

concatenarDatos1 = concatenarDatos1.with_columns([
    pl.col("Direccion").str.extract(r', (.*)', 1) #Extrae lo que viene después de la coma, la primera coma, por eso el 1, y agarra todo con *
])

#DATOS HASTA AQUÍ TDO BIEN YA QUE LOS DATOS ESTAN BIEN CONCATENADOS


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#||||||||||||------------LIMPIAR METROS HACIA COLUMNA----------||||||||||||| (LISTO)

concatenarDatos1 = concatenarDatos1.with_columns([
    pl.col("Metros").str.replace(r"m²","")
])

concatenarDatos1 = concatenarDatos1.rename({"Metros": "Metros (m²)"})
concatenarDatos1 = concatenarDatos1.rename({"Precio": "Precio ($)"})
concatenarDatos1 = concatenarDatos1.rename({"Gastos_Comunes": "Gastos_Comunes ($)"})
concatenarDatos1 = concatenarDatos1.rename({"Nombre": "Nombre_Arriendo"})

concatenarDatos1 = concatenarDatos1.with_columns([
    pl.col("Metros (m²)").str.strip_chars().str.replace(r",",".").cast(pl.Float64)
])

concatenarDatos1 = concatenarDatos1.with_columns([
    pl.col("Metros (m²)").round(0).cast(pl.Int64)
])

print(concatenarDatos1.head(5))

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#PASAR A CSV COMPLETO

#concatenarDatos1.write_csv("DatosCompletosLimpios3303.csv") <- Exportar A Csv