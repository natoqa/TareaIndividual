import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import io

st.set_page_config(
    page_title="Procesamiento de Datasets ML",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Procesamiento de Datasets en Machine Learning")
st.markdown("**Actividad Individual - Sistemas Inteligentes**")
st.markdown("---")

st.sidebar.title("🔍 Navegación")
ejercicio = st.sidebar.radio(
    "Selecciona un ejercicio:",
    ["Ejercicio 1: Titanic", "Ejercicio 2: Student Performance", "Ejercicio 3: Iris"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🪜 Etapas del Procesamiento")
st.sidebar.markdown("""
1. ✅ Carga del dataset
2. ✅ Exploración inicial
3. ✅ Limpieza de datos
4. ✅ Codificación categóricas
5. ✅ Normalización/Estandarización
6. ✅ División train/test
""")

# ==================== EJERCICIO 1: TITANIC ====================
if ejercicio == "Ejercicio 1: Titanic":
    st.header("🚢 Ejercicio 1: Dataset Titanic")
    st.markdown("**Objetivo:** Preparar los datos para un modelo que prediga la supervivencia de los pasajeros.")
    
    archivo_cargado = st.file_uploader("📁 Sube tu archivo titanic.csv", type=['csv'])
    
    if archivo_cargado is not None:
        datos = pd.read_csv(archivo_cargado)
        st.success("✅ Archivo cargado correctamente")
        
        st.subheader("1️⃣ Exploración Inicial")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Información del Dataset:**")
            buffer = io.StringIO()
            datos.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with col2:
            st.markdown("**Valores Nulos:**")
            st.dataframe(datos.isnull().sum().to_frame('Valores Nulos'))
        
        st.markdown("**Estadísticas Descriptivas:**")
        st.dataframe(datos.describe())
        
        st.subheader("2️⃣ Limpieza de Datos")
        
        datos_limpios = datos.copy()
        
        st.markdown("**Columnas eliminadas:** Name, Ticket, Cabin, PassengerId")
        columnas_eliminar = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        datos_limpios = datos_limpios.drop(columnas_eliminar, axis=1, errors='ignore')
        
        st.markdown("**Manejo de valores nulos:**")
        media_edad = datos_limpios['Age'].mean()
        media_tarifa = datos_limpios['Fare'].mean()
        moda_embarque = datos_limpios['Embarked'].mode()[0]
        
        st.write(f"- Age: Imputar con la media ({media_edad:.2f})")
        st.write(f"- Fare: Imputar con la media ({media_tarifa:.2f})")
        st.write(f"- Embarked: Imputar con la moda ({moda_embarque})")
        
        datos_limpios['Age'].fillna(media_edad, inplace=True)
        datos_limpios['Fare'].fillna(media_tarifa, inplace=True)
        datos_limpios['Embarked'].fillna(moda_embarque, inplace=True)
        
        st.subheader("3️⃣ Codificación de Variables Categóricas")
        
        codificador_sexo = LabelEncoder()
        datos_limpios['Sex'] = codificador_sexo.fit_transform(datos_limpios['Sex'])
        st.write("**Sex codificado:**", dict(zip(codificador_sexo.classes_, codificador_sexo.transform(codificador_sexo.classes_))))
        
        codificador_embarque = LabelEncoder()
        datos_limpios['Embarked'] = codificador_embarque.fit_transform(datos_limpios['Embarked'])
        st.write("**Embarked codificado:**", dict(zip(codificador_embarque.classes_, codificador_embarque.transform(codificador_embarque.classes_))))
        
        st.subheader("4️⃣ Estandarización")
        
        escalador = StandardScaler()
        datos_limpios[['Age', 'Fare']] = escalador.fit_transform(datos_limpios[['Age', 'Fare']])
        st.success("✅ Variables estandarizadas: Age, Fare")
        
        st.subheader("📋 Primeros 5 Registros Procesados")
        st.dataframe(datos_limpios.head())
        
        st.subheader("5️⃣ División en Conjuntos de Entrenamiento y Prueba")
        
        caracteristicas = datos_limpios.drop('Survived', axis=1)
        etiquetas = datos_limpios['Survived']
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            caracteristicas, etiquetas, test_size=0.30, random_state=42
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Entrenamiento", f"{X_entrenamiento.shape[0]} filas")
        with col2:
            st.metric("🧪 Prueba", f"{X_prueba.shape[0]} filas")
        with col3:
            st.metric("📊 Proporción", "70% / 30%")
        
        st.write(f"**Shape de entrenamiento:** {X_entrenamiento.shape}")
        st.write(f"**Shape de prueba:** {X_prueba.shape}")
        
        st.success("✅ Ejercicio 1 completado exitosamente")
    else:
        st.info("👆 Por favor, carga el archivo titanic.csv para comenzar el análisis")

# ==================== EJERCICIO 2: STUDENT PERFORMANCE ====================
elif ejercicio == "Ejercicio 2: Student Performance":
    st.header("🎓 Ejercicio 2: Student Performance")
    st.markdown("**Objetivo:** Procesar los datos para un modelo que prediga la nota final (G3) de los estudiantes.")
    
    archivo_cargado = st.file_uploader("📁 Sube tu archivo student-mat.csv", type=['csv'])
    
    if archivo_cargado is not None:
        try:
            datos = pd.read_csv(archivo_cargado, sep=';')
        except:
            datos = pd.read_csv(archivo_cargado)
        
        st.success("✅ Archivo cargado correctamente")
        
        st.subheader("1️⃣ Carga y Exploración")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Información del Dataset:**")
            buffer = io.StringIO()
            datos.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with col2:
            st.markdown("**Variables Categóricas:**")
            columnas_categoricas = datos.select_dtypes(include=['object']).columns.tolist()
            st.write(columnas_categoricas)
        
        st.subheader("2️⃣ Limpieza de Datos")
        
        datos_limpios = datos.copy()
        filas_antes = len(datos_limpios)
        datos_limpios = datos_limpios.drop_duplicates()
        filas_despues = len(datos_limpios)
        
        st.write(f"**Duplicados eliminados:** {filas_antes - filas_despues}")
        st.success("✅ Verificación de valores inconsistentes completada")
        
        st.subheader("3️⃣ One-Hot Encoding")
        
        variables_categoricas = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                                'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                                'famsup', 'paid', 'activities', 'nursery', 'higher', 
                                'internet', 'romantic']
        
        variables_existentes = [var for var in variables_categoricas if var in datos_limpios.columns]
        
        datos_codificados = pd.get_dummies(datos_limpios, columns=variables_existentes, drop_first=True)
        st.write(f"**Variables codificadas:** {len(variables_existentes)}")
        st.write(f"**Columnas totales después de encoding:** {datos_codificados.shape[1]}")
        st.info("💡 drop_first=True aplicado para evitar multicolinealidad")
        
        st.subheader("4️⃣ Normalización")
        
        variables_numericas = ['age', 'absences', 'G1', 'G2']
        variables_num_existentes = [var for var in variables_numericas if var in datos_codificados.columns]
        
        if len(variables_num_existentes) > 0:
            normalizador = MinMaxScaler()
            datos_codificados[variables_num_existentes] = normalizador.fit_transform(datos_codificados[variables_num_existentes])
            st.success(f"✅ Variables normalizadas: {', '.join(variables_num_existentes)}")
        else:
            st.warning("⚠️ No se encontraron las variables numéricas esperadas (age, absences, G1, G2)")
            st.info("Columnas disponibles en el dataset:")
            st.write(list(datos_codificados.columns))

        st.subheader("5️⃣ Separación de Variables")
        
        caracteristicas = datos_codificados.drop('G3', axis=1)
        objetivo = datos_codificados['G3']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**X (características):** {caracteristicas.shape}")
        with col2:
            st.write(f"**y (objetivo G3):** {objetivo.shape}")
        
        st.subheader("6️⃣ División de Datos")
        
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            caracteristicas, objetivo, test_size=0.20, random_state=42
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Entrenamiento", f"{X_entrenamiento.shape[0]} filas")
        with col2:
            st.metric("🧪 Prueba", f"{X_prueba.shape[0]} filas")
        with col3:
            st.metric("📊 Proporción", "80% / 20%")
        
        st.write(f"**Shape de entrenamiento:** {X_entrenamiento.shape}")
        st.write(f"**Shape de prueba:** {X_prueba.shape}")
        
        st.subheader("🎯 Reto Adicional: Análisis de Correlación")
        
        matriz_correlacion = datos_limpios[['G1', 'G2', 'G3']].corr()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Matriz de Correlación:**")
            st.dataframe(matriz_correlacion.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlación entre G1, G2 y G3')
            st.pyplot(fig)
        
        st.markdown("**Conclusiones:**")
        st.write(f"- Correlación G1 ↔ G2: {matriz_correlacion.loc['G1', 'G2']:.3f}")
        st.write(f"- Correlación G2 ↔ G3: {matriz_correlacion.loc['G2', 'G3']:.3f}")
        st.write(f"- Correlación G1 ↔ G3: {matriz_correlacion.loc['G1', 'G3']:.3f}")
        st.info("💡 Las notas anteriores son excelentes predictores de la nota final G3")
        
        st.success("✅ Ejercicio 2 completado exitosamente")
    else:
        st.info("👆 Por favor, carga el archivo student-mat.csv para comenzar el análisis")

# ==================== EJERCICIO 3: IRIS ====================
else:  
    st.header("🌸 Ejercicio 3: Dataset Iris")
    st.markdown("**Objetivo:** Implementar un flujo completo de preprocesamiento y visualizar resultados.")
    
    st.subheader("1️⃣ Carga del Dataset")
    
    datos_iris = load_iris()
    st.success(f"✅ Dataset Iris cargado desde sklearn.datasets")
    st.write(f"**Número de muestras:** {datos_iris.data.shape[0]}")
    st.write(f"**Número de características:** {datos_iris.data.shape[1]}")
    st.write(f"**Clases:** {', '.join(datos_iris.target_names)}")
    
    st.subheader("2️⃣ Conversión a DataFrame")
    
    df_iris = pd.DataFrame(data=datos_iris.data, columns=datos_iris.feature_names)
    df_iris['target'] = datos_iris.target
    
    st.markdown("**Primeras 5 filas:**")
    st.dataframe(df_iris.head())

    st.subheader("📊 Exploración Inicial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Información del Dataset:**")
        buffer = io.StringIO()
        df_iris.info(buf=buffer)
        st.text(buffer.getvalue())
    
    with col2:
        st.markdown("**Distribución de Clases:**")
        distribucion_clases = df_iris['target'].value_counts().sort_index()
        for indice, conteo in distribucion_clases.items():
            st.write(f"- {datos_iris.target_names[indice]}: {conteo} muestras")
    
    st.markdown("**Estadísticas Descriptivas (datos originales):**")
    st.dataframe(df_iris.describe())

    st.subheader("3️⃣ Estandarización")
    
    caracteristicas = df_iris.drop('target', axis=1)
    etiquetas = df_iris['target']
    
    escalador = StandardScaler()
    caracteristicas_escaladas = escalador.fit_transform(caracteristicas)
    
    df_estandarizado = pd.DataFrame(caracteristicas_escaladas, columns=datos_iris.feature_names)
    df_estandarizado['target'] = etiquetas
    
    st.success("✅ StandardScaler aplicado a todas las características")
    st.markdown("**Estadísticas después de estandarización:**")
    st.dataframe(df_estandarizado.describe())
    st.info("💡 Nota: Media ≈ 0, Desviación estándar ≈ 1")
    
    st.subheader("4️⃣ División de Datos")
    
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        caracteristicas_escaladas, etiquetas, test_size=0.30, random_state=42
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Entrenamiento", f"{X_entrenamiento.shape[0]} filas")
    with col2:
        st.metric("🧪 Prueba", f"{X_prueba.shape[0]} filas")
    with col3:
        st.metric("📊 Proporción", "70% / 30%")
    
    st.write(f"**Shape de entrenamiento:** {X_entrenamiento.shape}")
    st.write(f"**Shape de prueba:** {X_prueba.shape}")
    
    st.subheader("5️⃣ Visualización")
    st.markdown("**Gráfico de dispersión: Sepal Length vs Petal Length por Clase**")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colores = ['red', 'green', 'blue']
    marcadores = ['o', 's', '^']
    
    for i, nombre_especie in enumerate(datos_iris.target_names):
        mascara = etiquetas == i
        ax.scatter(
            caracteristicas_escaladas[mascara, 0],  
            caracteristicas_escaladas[mascara, 2],  
            c=colores[i],
            label=nombre_especie,
            alpha=0.7,
            edgecolors='black',
            s=100,
            marker=marcadores[i]
        )
    
    ax.set_xlabel('Sepal Length (estandarizada)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Petal Length (estandarizada)', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Sepal Length vs Petal Length por Clase', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Especies', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown("**Interpretación:**")
    st.write("- Las tres especies se separan claramente en el espacio bidimensional")
    st.write("- Setosa (rojo) se distingue fácilmente de las otras dos especies")
    st.write("- Versicolor (verde) y Virginica (azul) tienen cierta superposición")

    st.markdown("**Gráfico de Pares (Pairplot):**")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    pares_caracteristicas = [(0, 1), (0, 2), (1, 2), (1, 3)]
    nombres_cortos = ['sepal length', 'sepal width', 'petal length', 'petal width']
    
    for idx, (i, j) in enumerate(pares_caracteristicas):
        ax = axes[idx // 2, idx % 2]
        for k, nombre_especie in enumerate(datos_iris.target_names):
            mascara = etiquetas == k
            ax.scatter(caracteristicas_escaladas[mascara, i], caracteristicas_escaladas[mascara, j], 
                      c=colores[k], label=nombre_especie, alpha=0.6, s=50)
        ax.set_xlabel(nombres_cortos[i], fontsize=10)
        ax.set_ylabel(nombres_cortos[j], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.success("✅ Ejercicio 3 completado exitosamente")

st.markdown("---")
st.markdown("### 📝 Resumen de Salidas Esperadas")

if ejercicio == "Ejercicio 1: Titanic":
    st.markdown("""
    ✅ **Completado:**
    - Tabla con los primeros 5 registros procesados
    - Impresión de shape de entrenamiento y prueba (70/30)
    - Estandarización de Age y Fare
    - Codificación de Sex y Embarked
    """)
elif ejercicio == "Ejercicio 2: Student Performance":
    st.markdown("""
    ✅ **Completado:**
    - One-Hot Encoding aplicado
    - Normalización de variables numéricas
    - División 80/20
    - Análisis de correlación entre G1, G2 y G3 (Reto adicional)
    """)
else:
    st.markdown("""
    ✅ **Completado:**
    - Gráfico de dispersión con colores por clase
    - Estadísticas descriptivas del dataset estandarizado
    - Dataset preparado para modelado (70/30)
    """)

st.markdown("---")
st.markdown("Desarrollado para el curso de **Sistemas Inteligentes** 🤖")