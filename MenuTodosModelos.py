#----------- Librerías a Usar ------------------
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor 
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


#----------- Variables Globales para Modelo Linear Regression ------------------
modeloLinear = None
x_test_linear = None
y_test_linear = None
predic_linear = None
comuna_cols = None
data_linear = None


#----------- Variables Globales para Modelo KNN ------------------
knn_model = None
X_test_knn = None
y_test_knn = None
y_pred_knn = None
knn_metrics = None
umbral_precio = None


#----------- Variables Globales para Modelos de Árboles ------------------
tree_sin_reg = None
tree_con_reg = None
forest_model = None
X_test_trees = None
y_test_trees = None
y_pred_sin_reg = None
y_pred_con_reg = None
y_pred_forest = None
feature_names_trees = None
X_simple_trees = None
y_simple_trees = None

#----------- Variables Globales para Red Neuronal (MLP) ------------------
modelo_nn = None
scaler_nn = None
comuna_cols_nn = None
x_test_nn = None
y_test_nn = None
predic_nn = None
metrics_nn = {}


def cargar_y_entrenar_modelo_linear():
    """Carga los datos, entrena el modelo Linear Regression y lo deja listo para usar"""
    global modeloLinear, x_test_linear, y_test_linear, predic_linear, comuna_cols, data_linear
    
    print("\n🔄 Cargando datos y entrenando modelo Linear Regression...")
    
    #----------- Datos del Csv ------------------
    dt = pl.read_csv("DatosRevisadosLimpios.csv")
    data_linear = pl.DataFrame(dt)
    
    #----------- Crear columnas para todas las comunas ------------------
    data_linear = data_linear.to_dummies(columns=["Comuna"])
    comuna_cols = [col for col in data_linear.columns if col.startswith("Comuna_")]
    
    #----------- Extracción De Datos por columnas ------------------
    xx = data_linear.select([
        "Gastos_Comunes ($)",
        "Metros (m²)",
        "Habitaciones",
        "Baños",
        *[col for col in data_linear.columns if col.startswith("Comuna_")]
    ]).to_numpy()
    
    yy = data_linear.select(pl.col("Precio ($)")).to_numpy().ravel()
    
    #----------- División y entrenamiento de datos ------------------
    x_train, x_test_linear, y_train, y_test_linear = train_test_split(xx, yy, test_size=0.3, random_state=42)
    
    #----------- Modelo y entrenamiento ------------------
    modeloLinear = LinearRegression()
    modeloLinear.fit(x_train, y_train)
    
    #----------- Predicción usando Modelo ------------------
    predic_linear = modeloLinear.predict(x_test_linear)
    
    print("✅ Modelo Linear Regression entrenado exitosamente!\n")


def cargar_y_entrenar_modelo_knn():
    """Carga los datos, entrena el modelo KNN y lo deja listo para usar"""
    global knn_model, X_test_knn, y_test_knn, y_pred_knn, knn_metrics, umbral_precio
    
    print("\n🔄 Cargando datos y entrenando modelo KNN...")
    
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
    umbral_precio = df['Precio'].median()
    df = df.with_columns([
        (df['Precio'] > umbral_precio).cast(pl.Int8).alias('ArriendoAlto')
    ])
    
    # 5. Convertir a pandas (para sklearn)
    df_pd = df.to_pandas()
    
    # 6. Separar X e y
    X = df_pd[['MetrosCuadrados', 'Comuna', 'Baños', 'Habitaciones', 'GastosComunes']]
    y = df_pd['ArriendoAlto']
    
    # 7. División entrenamiento/prueba
    X_train, X_test_knn, y_train, y_test_knn = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 8. Preprocesamiento (escalar + codificar)
    num_cols = ['MetrosCuadrados', 'Baños', 'Habitaciones', 'GastosComunes']
    cat_cols = ['Comuna']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    # 9. Modelo KNN
    knn_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ])
    
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test_knn)
    
    # 10. Calcular métricas
    knn_metrics = {
        'Accuracy': accuracy_score(y_test_knn, y_pred_knn),
        'Precision': precision_score(y_test_knn, y_pred_knn),
        'Recall': recall_score(y_test_knn, y_pred_knn)
    }
    
    print(f"✅ Modelo KNN entrenado exitosamente!")
    print(f"📊 Umbral de precio (mediana): ${umbral_precio:,.0f}\n")


def cargar_y_entrenar_modelos_arboles():
    """Carga los datos, entrena los modelos de árboles (sin reg, con reg, Random Forest)"""
    global tree_sin_reg, tree_con_reg, forest_model
    global X_test_trees, y_test_trees
    global y_pred_sin_reg, y_pred_con_reg, y_pred_forest
    global feature_names_trees, X_simple_trees, y_simple_trees
    
    print("\n🔄 Cargando datos y entrenando modelos de Árboles de Decisión...")
    
    # Cargar datos
    df = pd.read_csv("DatosRevisadosLimpios.csv")
    
    # Renombrar columnas
    df = df.rename(columns={
        'Precio ($)': 'Precio',
        'Metros (m²)': 'MetrosCuadrados',
        'Gastos_Comunes ($)': 'GastosComunes'
    })
    
    # Seleccionar columnas relevantes
    df = df[['Precio', 'MetrosCuadrados', 'Comuna', 'Baños', 'Habitaciones', 'GastosComunes']].dropna()
    
    # Separar variables
    X = df[['MetrosCuadrados', 'Comuna', 'Baños', 'Habitaciones', 'GastosComunes']]
    y = df['Precio']
    
    # División entrenamiento/prueba
    X_train, X_test_trees, y_train, y_test_trees = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Preprocesamiento
    num_cols = ['MetrosCuadrados', 'Baños', 'Habitaciones', 'GastosComunes']
    cat_cols = ['Comuna']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    # 1. Árbol SIN regularización
    print("   🌳 Entrenando Árbol SIN regularización...")
    tree_sin_reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(
            max_depth=None,
            min_samples_leaf=1,
            random_state=42
        ))
    ])
    tree_sin_reg.fit(X_train, y_train)
    y_pred_sin_reg = tree_sin_reg.predict(X_test_trees)
    
    # 2. Árbol CON regularización
    print("   🌳 Entrenando Árbol CON regularización...")
    tree_con_reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=10,
            random_state=42
        ))
    ])
    tree_con_reg.fit(X_train, y_train)
    y_pred_con_reg = tree_con_reg.predict(X_test_trees)
    
    # 3. Random Forest
    print("   🌲 Entrenando Random Forest (100 árboles)...")
    forest_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    forest_model.fit(X_train, y_train)
    y_pred_forest = forest_model.predict(X_test_trees)
    
    # Preparar datos para visualización del árbol
    X_simple_trees = X.copy()
    X_simple_trees = pd.get_dummies(X_simple_trees, columns=['Comuna'], drop_first=False, prefix='Comuna')
    y_simple_trees = y
    
    # Nombres de características para importancia
    encoder = forest_model.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cols = list(encoder.get_feature_names_out(cat_cols))
    feature_names_trees = num_cols + encoded_cols
    
    print("✅ Modelos de Árboles entrenados exitosamente!\n")


def cargar_y_entrenar_modelo_red_neuronal():
    """Carga datos filtrados, entrena el MLP Regressor y lo guarda globalmente."""
    global modelo_nn, scaler_nn, comuna_cols_nn, x_test_nn, y_test_nn, predic_nn, metrics_nn
    
    print("\n🔄 Cargando datos y entrenando modelo Red Neuronal (MLP)...")
    
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
    comuna_cols_nn = [col for col in data.columns if col.startswith("Comuna_")] # Guardar comunas específicas de este modelo


    #----------- Selección De Variables ------------------
    xx = data.select([
        "Gastos_Comunes ($)",
        "Metros (m²)",
        "Habitaciones",
        "Baños",
        *[col for col in data.columns if col.startswith("Comuna_")]
    ]).to_numpy()

    yy = data.select(pl.col("Precio ($)")).to_numpy().ravel()

    #----------- Separar y entrenar los datos ------------------
    x_train, x_test_nn, y_train, y_test_nn = train_test_split(xx, yy, test_size=0.3, random_state=42)

    #----------- Escalado de Datos ------------------
    scaler_nn = StandardScaler()
    x_train_scaled = scaler_nn.fit_transform(x_train)
    x_test_scaled = scaler_nn.transform(x_test_nn)

    #----------- Crear y definir parámetros en Red Neuronal ------------------
    modelo_nn = MLPRegressor(
        hidden_layer_sizes=(40, 20),   # Capas Ocultas (Dos Capas) -> 40 y 20 neuronas
        activation='relu',             # Función de activación moderna relu
        max_iter=7000,                 # N° De Iteraciones
        random_state=42,
        early_stopping=True,           # Añadido para evitar sobreajuste y acortar entrenamiento
        n_iter_no_change=50
    )
    
    print("   🧠 Entrenando MLP Regressor (puede tomar un momento)...")
    #----------- Entrenamiento De Datos ------------------
    modelo_nn.fit(x_train_scaled, y_train)

    #----------- Metricas Arrojadas (Evaluación) ------------------
    predic_nn = modelo_nn.predict(x_test_scaled)
    
    metrics_nn['R2'] = r2_score(y_test_nn, predic_nn)
    metrics_nn['MSE'] = mean_squared_error(y_test_nn, predic_nn)
    metrics_nn['MAE'] = mean_absolute_error(y_test_nn, predic_nn)
    metrics_nn['RMSE'] = np.sqrt(mean_squared_error(y_test_nn, predic_nn))
    
    print(f"✅ Modelo Red Neuronal entrenado exitosamente!")
    print(f"   R² -> {metrics_nn['R2']:.4f}")
    print(f"   MAE -> ${metrics_nn['MAE']:,.0f}\n")


def accion1():
    """Predecir precio de arriendo con Linear Regression"""
    global modeloLinear, comuna_cols
    
    if modeloLinear is None:
        print("\n⚠️  El modelo Linear Regression aún no ha sido entrenado.")
        cargar_y_entrenar_modelo_linear()
    
    print("\n=========== PREDICCIÓN DE PRECIO DE ARRIENDO (LINEAR REGRESSION) ===========")
    
    try:
        gastosComun = int(input("\nIngrese gastos comunes aproximados -> $"))
        metrosC = int(input("Ingrese metros cuadrados del arriendo -> "))
        habitaciones = int(input("Ingrese las habitaciones para el arriendo -> "))
        baños = int(input("Ingrese los baños para el arriendo -> "))
        comuna = input("Ingrese la comuna -> ").strip().title()
        
        comuna_dummies = [1 if col == f"Comuna_{comuna}" else 0 for col in comuna_cols]
        
        if sum(comuna_dummies) == 0: 
            print(f"\n⚠️  Advertencia: la comuna '{comuna}' no existe en los datos. Se asumirá base (todas 0).")
        
        predikkk = [gastosComun, metrosC, habitaciones, baños] + comuna_dummies
        
        ff = modeloLinear.predict([predikkk])
        print(f"\n💰 Su arriendo en {comuna} tendrá el valor de -> ${round(ff[0], 0):,.0f}")
        
    except ValueError:
        print("\n❌ Error: Debe ingresar valores numéricos válidos.")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")


def accion2():
    """Clasificar arriendo como ALTO o BAJO con KNN"""
    global knn_model, umbral_precio
    
    if knn_model is None:
        print("\n⚠️  El modelo KNN aún no ha sido entrenado.")
        cargar_y_entrenar_modelo_knn()
    
    print("\n=========== CLASIFICACIÓN DE ARRIENDO (KNN) ===========")
    print(f"📊 Umbral de precio (mediana): ${umbral_precio:,.0f}")
    print("    • Arriendo ALTO: mayor a este valor")
    print("    • Arriendo BAJO: menor o igual a este valor\n")
    
    try:
        metros = float(input("Metros cuadrados: "))
        banos = int(input("Número de baños: "))
        habs = int(input("Número de habitaciones: "))
        gastos = float(input("Gastos comunes ($): "))
        comuna = input("Nombre de la comuna: ").strip().title()
        
        nuevo_dato = pd.DataFrame([{
            'MetrosCuadrados': metros,
            'Comuna': comuna,
            'Baños': banos,
            'Habitaciones': habs,
            'GastosComunes': gastos
        }])
        
        prediccion = knn_model.predict(nuevo_dato)[0]
        
        print("\n🔎 Resultado del modelo KNN:")
        if prediccion == 1:
            print("➡️  El modelo predice que el arriendo sería ALTO 💰")
            print(f"    (Precio estimado mayor a ${umbral_precio:,.0f})")
        else:
            print("➡️  El modelo predice que el arriendo sería BAJO 🏡")
            print(f"    (Precio estimado menor o igual a ${umbral_precio:,.0f})")
    
    except Exception as e:
        print(f"\n⚠️ Error al ingresar o procesar los datos: {e}")


def accion3():
    """Predecir precio con modelos de árboles (Decision Tree y Random Forest)"""
    global tree_sin_reg, tree_con_reg, forest_model
    
    if forest_model is None:
        print("\n⚠️  Los modelos de Árboles aún no han sido entrenados.")
        cargar_y_entrenar_modelos_arboles()
    
    print("\n=========== PREDICCIÓN CON ÁRBOLES DE DECISIÓN Y RANDOM FOREST ===========")
    
    try:
        metrosC = float(input("\nMetros cuadrados: "))
        comuna = input("Nombre de la comuna: ").strip().title()
        baños = int(input("Número de baños: "))
        habitaciones = int(input("Número de habitaciones: "))
        gastosComun = float(input("Gastos comunes ($): "))
        
        nuevo_dato = pd.DataFrame([{
            'MetrosCuadrados': metrosC,
            'Comuna': comuna,
            'Baños': baños,
            'Habitaciones': habitaciones,
            'GastosComunes': gastosComun
        }])
        
        # Predicciones
        pred_sin_reg = tree_sin_reg.predict(nuevo_dato)[0]
        pred_con_reg = tree_con_reg.predict(nuevo_dato)[0]
        pred_forest = forest_model.predict(nuevo_dato)[0]
        
        print("\n📊 RESULTADOS DE PREDICCIÓN:")
        print(f"\n🌳 Árbol SIN Regularización: ${pred_sin_reg:,.0f}")
        print(f"   (Puede tener sobreajuste - memoriza datos)")
        
        print(f"\n🌳 Árbol CON Regularización: ${pred_con_reg:,.0f}")
        print(f"   (Mejor balance - más confiable)")
        
        print(f"\n🌲 Random Forest (100 árboles): ${pred_forest:,.0f}")
        print(f"   (Más robusto - recomendado)")
        
        print(f"\n💡 Recomendación: Usar el valor de Random Forest (${pred_forest:,.0f})")
        
    except Exception as e:
        print(f"\n⚠️ Error al ingresar o procesar los datos: {e}")

def accion4_predecir_nn():
    """Predecir precio con Red Neuronal (MLP)"""
    global modelo_nn, scaler_nn, comuna_cols_nn
    
    if modelo_nn is None:
        print("\n⚠️  El modelo de Red Neuronal aún no ha sido entrenado.")
        cargar_y_entrenar_modelo_red_neuronal()
    
    print("\n=========== PREDICCIÓN CON RED NEURONAL (MLP REGRESSOR) ===========")
    print("ℹ️  Este modelo fue entrenado con datos filtrados:")
    print("   (Precio: $100k-$2M, Metros: 15-200 m²)")
    
    try:
        gastosComun = int(input("\nIngrese gastos comunes aproximados -> $"))
        metrosC = int(input("Ingrese metros cuadrados del arriendo -> "))
        habitaciones = int(input("Ingrese las habitaciones para el arriendo -> "))
        baños = int(input("Ingrese los baños para el arriendo -> "))
        comuna = input("Ingrese la comuna -> ").strip().title()

        # Usar las columnas de comuna específicas del modelo NN
        comuna_dummies = [1 if col == f"Comuna_{comuna}" else 0 for col in comuna_cols_nn]

        if sum(comuna_dummies) == 0: 
            print(f"\n⚠️  Advertencia: la comuna '{comuna}' no existe en los datos de este modelo. Se asumirá base (todas 0).")

        predikkk = [gastosComun, metrosC, habitaciones, baños] + comuna_dummies
        
        # Escalar los datos con el escalador específico del modelo NN
        predikkk_scaled = scaler_nn.transform([predikkk])

        # Prediccion con datos recientes
        ff = modelo_nn.predict(predikkk_scaled)
        print(f"\n🧠 Su arriendo en {comuna} tendrá el valor de -> ${round(ff[0],0):,.0f}")
        
    except ValueError:
        print("\n❌ Error: Debe ingresar valores numéricos válidos.")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")


def accion5_mostrar_metricas():
    """Mostrar métricas de todos los modelos"""
    print("\n=========== MÉTRICAS DE TODOS LOS MODELOS ===========")
    
    # Métricas Linear Regression
    if predic_linear is None:
        print("\n⚠️  Entrenando modelo Linear Regression...")
        cargar_y_entrenar_modelo_linear()
    
    print("\n📈 MODELO 1: LINEAR REGRESSION (Predicción de precio)")
    print(f"   R² (Coeficiente de determinación) -> {r2_score(y_test_linear, predic_linear):.4f}")
    print(f"   MSE (Error cuadrático medio) -> ${mean_squared_error(y_test_linear, predic_linear):,.2f}")
    print(f"   MAE (Error absoluto medio) -> ${mean_absolute_error(y_test_linear, predic_linear):,.2f}")
    print(f"   RMSE (Raíz del error cuadrático medio) -> ${np.sqrt(mean_squared_error(y_test_linear, predic_linear)):,.2f}")
    
    # Métricas KNN
    if knn_metrics is None:
        print("\n⚠️  Entrenando modelo KNN...")
        cargar_y_entrenar_modelo_knn()
    
    print("\n📊 MODELO 2: KNN (Clasificación Alto/Bajo)")
    for k, v in knn_metrics.items():
        print(f"   {k} -> {v:.3f}")
    
    # Métricas Árboles
    if forest_model is None:
        print("\n⚠️  Entrenando modelos de Árboles...")
        cargar_y_entrenar_modelos_arboles()
    
    print("\n🌳 MODELO 3: ÁRBOLES DE DECISIÓN Y RANDOM FOREST")
    
    # Árbol sin regularización
    mae_sin = mean_absolute_error(y_test_trees, y_pred_sin_reg)
    r2_sin = r2_score(y_test_trees, y_pred_sin_reg)
    print(f"\n   Árbol SIN Regularización:")
    print(f"       MAE -> ${mae_sin:,.0f}")
    print(f"       R² -> {r2_sin:.4f}")
    
    # Árbol con regularización
    mae_con = mean_absolute_error(y_test_trees, y_pred_con_reg)
    r2_con = r2_score(y_test_trees, y_pred_con_reg)
    print(f"\n   Árbol CON Regularización:")
    print(f"       MAE -> ${mae_con:,.0f}")
    print(f"       R² -> {r2_con:.4f}")
    
    # Random Forest
    mae_forest = mean_absolute_error(y_test_trees, y_pred_forest)
    r2_forest = r2_score(y_test_trees, y_pred_forest)
    rmse_forest = np.sqrt(mean_squared_error(y_test_trees, y_pred_forest))
    print(f"\n   Random Forest (100 árboles):")
    print(f"       MAE -> ${mae_forest:,.0f}")
    print(f"       RMSE -> ${rmse_forest:,.0f}")
    print(f"       R² -> {r2_forest:.4f}")

    # Métricas Red Neuronal
    if not metrics_nn:
        print("\n⚠️  Entrenando modelo Red Neuronal...")
        cargar_y_entrenar_modelo_red_neuronal()
        
    print("\n🧠 MODELO 4: RED NEURONAL (MLP REGRESSOR)")
    print(f"   Datos filtrados (Precio: $100k-$2M, Metros: 15-200 m²)")
    print(f"   R² -> {metrics_nn.get('R2', 0):.4f}")
    print(f"   MAE -> ${metrics_nn.get('MAE', 0):,.2f}")
    print(f"   RMSE -> ${metrics_nn.get('RMSE', 0):,.2f}")


def accion6_grafico_lr():
    """Mostrar gráficos de Linear Regression"""
    global y_test_linear, predic_linear
    
    if predic_linear is None:
        print("\n⚠️  El modelo Linear Regression aún no ha sido entrenado.")
        cargar_y_entrenar_modelo_linear()
    
    print("\n📊 Generando gráficos de Linear Regression...")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_linear, predic_linear, alpha=0.5)
    plt.plot([y_test_linear.min(), y_test_linear.max()], [y_test_linear.min(), y_test_linear.max()], color='red', linewidth=2)
    plt.xlabel("Precio Real ($)")
    plt.ylabel("Precio Predicho ($)")
    plt.title("Comparación entre Precio Real y Precio Predicho (Linear Regression)")
    plt.grid(True)
    plt.show()
    
    residuos = y_test_linear - predic_linear
    
    plt.figure(figsize=(8, 6))
    plt.scatter(predic_linear, residuos, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Precio Predicho ($)")
    plt.ylabel("Error (Residual)")
    plt.title("Distribución de Errores del Modelo (Linear Regression)")
    plt.grid(True)
    plt.show()


def accion7_grafico_knn():
    """Mostrar gráficos de KNN"""
    global knn_model, X_test_knn, y_test_knn, knn_metrics
    
    if knn_model is None:
        print("\n⚠️  El modelo KNN aún no ha sido entrenado.")
        cargar_y_entrenar_modelo_knn()
    
    print("\n📊 Generando gráficos de KNN...")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(knn_model, X_test_knn, y_test_knn, ax=ax, cmap='Greens')
    ax.set_title("Matriz de Confusión - KNN")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(knn_metrics.keys()), y=list(knn_metrics.values()), color="green", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Desempeño del Modelo KNN")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)
    plt.show()


def accion8_grafico_arboles():
    """Mostrar gráficos de Árboles de Decisión"""
    global tree_sin_reg, tree_con_reg, forest_model
    global y_test_trees, y_pred_sin_reg, y_pred_con_reg, y_pred_forest
    
    if forest_model is None:
        print("\n⚠️  Los modelos de Árboles aún no han sido entrenados.")
        cargar_y_entrenar_modelos_arboles()
    
    print("\n📊 Generando gráficos de Árboles de Decisión...")
    
    # Comparación de los 3 modelos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    r2_sin = r2_score(y_test_trees, y_pred_sin_reg)
    r2_con = r2_score(y_test_trees, y_pred_con_reg)
    r2_forest = r2_score(y_test_trees, y_pred_forest)
    
    # Árbol sin regularización
    axes[0].scatter(y_test_trees, y_pred_sin_reg, alpha=0.4, s=20)
    axes[0].plot([y_test_trees.min(), y_test_trees.max()], [y_test_trees.min(), y_test_trees.max()], 'r--', lw=2)
    axes[0].set_xlabel('Precio Real', fontsize=11)
    axes[0].set_ylabel('Precio Predicho', fontsize=11)
    axes[0].set_title(f'Árbol SIN Regularización\nR² = {r2_sin:.3f}', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Árbol con regularización
    axes[1].scatter(y_test_trees, y_pred_con_reg, alpha=0.4, s=20)
    axes[1].plot([y_test_trees.min(), y_test_trees.max()], [y_test_trees.min(), y_test_trees.max()], 'r--', lw=2)
    axes[1].set_xlabel('Precio Real', fontsize=11)
    axes[1].set_ylabel('Precio Predicho', fontsize=11)
    axes[1].set_title(f'Árbol CON Regularización\nR² = {r2_con:.3f}', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Random Forest
    axes[2].scatter(y_test_trees, y_pred_forest, alpha=0.4, s=20)
    axes[2].plot([y_test_trees.min(), y_test_trees.max()], [y_test_trees.min(), y_test_trees.max()], 'r--', lw=2)
    axes[2].set_xlabel('Precio Real', fontsize=11)
    axes[2].set_ylabel('Precio Predicho', fontsize=11)
    axes[2].set_title(f'Random Forest\nR² = {r2_forest:.3f}', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Gráfico de métricas comparativas
    mae_sin = mean_absolute_error(y_test_trees, y_pred_sin_reg)
    mae_con = mean_absolute_error(y_test_trees, y_pred_con_reg)
    mae_forest = mean_absolute_error(y_test_trees, y_pred_forest)
    
    comparacion = pd.DataFrame({
        'Modelo': ['Árbol\n(Sin Reg.)', 'Árbol\n(Con Reg.)', 'Random\nForest'],
        'MAE': [mae_sin, mae_con, mae_forest],
        'R²': [r2_sin, r2_con, r2_forest]
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].bar(comparacion['Modelo'], comparacion['MAE'], 
                color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
    axes[0].set_ylabel('MAE (Error Absoluto Medio)', fontsize=12)
    axes[0].set_title('Comparación de MAE\n(Menor es mejor)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(comparacion['MAE']):
        axes[0].text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=10)
    
    axes[1].bar(comparacion['Modelo'], comparacion['R²'], 
                color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
    axes[1].set_ylabel('R² (Coeficiente de Determinación)', fontsize=12)
    axes[1].set_title('Comparación de R²\n(Mayor es mejor)', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(comparacion['R²']):
        axes[1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def accion9_grafico_nn():
    """Mostrar gráficos de Red Neuronal (MLP)"""
    global y_test_nn, predic_nn
    
    if predic_nn is None:
        print("\n⚠️  El modelo de Red Neuronal aún no ha sido entrenado.")
        cargar_y_entrenar_modelo_red_neuronal()
    
    print("\n📊 Generando gráficos de Red Neuronal (MLP)...")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_nn, predic_nn, alpha=0.4)
    plt.plot([y_test_nn.min(), y_test_nn.max()], [y_test_nn.min(), y_test_nn.max()], 'r--')
    plt.xlabel("Precio real")
    plt.ylabel("Precio predicho")
    plt.title("Predicho vs Real (Red Neuronal MLP)")
    plt.grid(True)
    plt.show()


def accion10_importancia():
    """Mostrar importancia de variables y estructura del árbol"""
    global forest_model, feature_names_trees, X_simple_trees, y_simple_trees
    
    if forest_model is None:
        print("\n⚠️  Los modelos de Árboles aún no han sido entrenados.")
        cargar_y_entrenar_modelos_arboles()
    
    print("\n📊 Generando gráficos de Importancia de Variables...")
    
    # Importancia de variables
    rf_regressor = forest_model.named_steps['regressor']
    importances = pd.DataFrame({
        'Variable': feature_names_trees,
        'Importancia': rf_regressor.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    print("\nTOP 10 Variables Más Importantes:")
    print(importances.head(10).to_string(index=False))
    
    plt.figure(figsize=(12, 8))
    top_15 = importances.head(15)
    sns.barplot(data=top_15, y='Variable', x='Importancia', palette='viridis')
    plt.title('Importancia de las Variables - Random Forest\n(Top 15)', fontsize=16, fontweight='bold')
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Estructura del árbol
    print("\n📊 Generando visualización de la estructura del árbol")
    print("\n📊 Generando visualización de la estructura del árbol...")
    
    tree_vis = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
    tree_vis.fit(X_simple_trees, y_simple_trees)
    
    plt.figure(figsize=(20, 10))
    plot_tree(tree_vis, feature_names=X_simple_trees.columns, filled=True, fontsize=9, 
              max_depth=3, rounded=True, precision=2)
    plt.title('Estructura de un Árbol de Decisión Regularizado\n(max_depth=3, min_samples_leaf=20)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def salir():
    print('\n👋 Saliendo del sistema...')


def mostrar_menu(opciones):
    print('\n' + '='*70)
    print('       SISTEMA COMPLETO DE PREDICCIÓN Y CLASIFICACIÓN DE ARRIENDOS')
    print('='*70)
    print('\nSeleccione una opción:')
    for clave in sorted(opciones, key=int):
        print(f' {clave}) {opciones[clave][0]}')


def leer_opcion(opciones):
    while (a := input('\nOpción: ')) not in opciones:
        print('❌ Opción incorrecta, vuelva a intentarlo.')
    return a


def ejecutar_opcion(opcion, opciones):
    opciones[opcion][1]()


def generar_menu(opciones, opcion_salida):
    opcion = None
    while opcion != opcion_salida:
        mostrar_menu(opciones)
        opcion = leer_opcion(opciones)
        ejecutar_opcion(opcion, opciones)
        print()


def menu_principal():
    opciones = {
        '1': ('🏠 Predecir precio (Linear Regression)', accion1),
        '2': ('🔍 Clasificar arriendo ALTO/BAJO (KNN)', accion2),
        '3': ('🌳 Predecir precio (Árboles y Random Forest)', accion3),
        '4': ('🧠 Predecir precio (Red Neuronal MLP)', accion4_predecir_nn),
        '5': ('📊 Ver métricas de TODOS los modelos', accion5_mostrar_metricas),
        '6': ('📈 Gráficos Linear Regression', accion6_grafico_lr),
        '7': ('📉 Gráficos KNN', accion7_grafico_knn),
        '8': ('🌲 Gráficos Árboles de Decisión', accion8_grafico_arboles),
        '9': ('🧠 Gráficos Red Neuronal', accion9_grafico_nn),
        '10': ('🔬 Importancia de variables y estructura del árbol', accion10_importancia),
        '11': ('🚪 Salir', salir)
    }
    
    generar_menu(opciones, '11')


if __name__ == '__main__':
    print("\n" + "="*70)
    print("   SISTEMA DE MACHINE LEARNING PARA PREDICCIÓN DE ARRIENDOS")
    print("="*70)
    print("\n📚 MODELOS DISPONIBLES:")
    print("   1️⃣  Linear Regression - Predicción de precio exacto")
    print("   2️⃣  KNN - Clasificación Alto/Bajo")
    print("   3️⃣  Árboles de Decisión y Random Forest")
    print("   4️⃣  Red Neuronal (MLP) - Predicción de precio (con filtros)")
    print("\n💡 Los modelos se entrenan automáticamente al usarlos")
    print("="*70)
    
    menu_principal()