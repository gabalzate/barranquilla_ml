# app.py - Predictor Electoral Barranquilla 2014
# Paso 1: Configuración inicial y estructura base

# Importar las bibliotecas necesarias
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Predictor Electoral Barranquilla",
    page_icon="📊",
    layout="wide",  # Usar diseño ancho para aprovechar mejor el espacio
    initial_sidebar_state="collapsed"  # Sidebar colapsado inicialmente
)

# Título principal de la aplicación
st.title("Predictor de Votos - Barranquilla")

# Estructura básica de la interfaz con dos columnas
col1, col2 = st.columns(2)

# Panel izquierdo - Parámetros de Predicción
with col1:
    st.header("Parámetros de Predicción")
    # Aquí irán los controles para ingresar los parámetros
    st.write("En este panel se incluirán los controles para seleccionar los parámetros.")
    
# Panel derecho - Resultados de la Predicción
with col2:
    st.header("Resultados de la Predicción")
    # Aquí irá la visualización de los resultados
    st.write("En este panel se mostrarán los resultados de la predicción.")

# Mensaje informativo sobre la aplicación
st.markdown("---")
st.info("Esta aplicación utiliza un modelo XGBoost optimizado para predecir votos en las elecciones de Barranquilla.")


# Funciones de carga y preprocesamiento
# Estas funciones se ejecutarán una sola vez y se almacenarán en caché

@st.cache_resource
def load_model():
    """
    Carga el modelo XGBoost guardado.
    Usa cache_resource para cargar el modelo solo una vez.
    """
    try:
        with open('modelo_xgboost_optimizado_barranquilla.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

@st.cache_data
def load_data():
    """
    Carga el dataset original.
    Usa cache_data para cargar los datos solo una vez.
    """
    try:
        # Ajusta esta ruta según donde tengas los datos
        data = pd.read_csv('./out/dataset_barranquilla_2014.csv')
        return data
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

def preprocess_input(input_df, train_data):
    """
    Preprocesa las entradas del usuario de la misma manera que se hizo en el entrenamiento.
    
    Args:
        input_df: DataFrame con los datos ingresados por el usuario.
        train_data: DataFrame con los datos de entrenamiento para referencia.
        
    Returns:
        DataFrame preprocesado listo para la predicción.
    """
    # Codificar variables categóricas
    categorical_columns = ['GÉNERO', 'IDEOLOGÍA', 'CAMBIO DE POLÍTICAS', 'APOYADO POR PARTIDOS MAYORITARIOS']
    
    # Creamos una copia para no modificar el original
    processed_df = input_df.copy()
    
    for column in categorical_columns:
        if column in processed_df.columns and processed_df[column].dtype == 'object':
            le = LabelEncoder()
            # Ajustar el encoder con los datos de entrenamiento para mantener la coherencia
            le.fit(train_data[column].astype(str))
            # Transformar los datos de entrada
            processed_df[column] = le.transform(processed_df[column].astype(str))
    
    # Normalizar características numéricas
    numerical_columns = ['ZONA', 'COD_PUESTO', 'MESA', 'EDAD', '# CANDIDATOS', 
                       'AÑOS DE TRAYECTORIA', 'ENCUESTA']
    
    # Crear y ajustar el scaler con los datos de entrenamiento
    scaler = StandardScaler()
    scaler.fit(train_data[numerical_columns])
    
    # Aplicar la transformación a los datos de entrada
    processed_df[numerical_columns] = scaler.transform(processed_df[numerical_columns])
    
    return processed_df

# Intentar cargar el modelo y los datos
try:
    model = load_model()
    data = load_data()
    model_loaded = True
    
    # Solo para depuración - mostrar las primeras filas del dataset
    if data is not None:
        st.sidebar.write("Dataset cargado correctamente")
        with st.sidebar.expander("Ver las primeras filas del dataset"):
            st.dataframe(data.head())
    else:
        model_loaded = False
        
except Exception as e:
    st.error(f"Error durante la inicialización: {e}")
    model_loaded = False


# Panel izquierdo - Parámetros de Predicción
with col1:
    st.header("Parámetros de Predicción")
    
    # Crear un formulario para los parámetros
    # El formulario agrupa los inputs y sólo ejecuta la predicción cuando se presiona el botón
    with st.form(key="prediction_form"):
        # Obtener valores únicos de los datos para poblar los selectores
        if model_loaded and data is not None:
            zonas = sorted(data['ZONA'].unique())
            puestos = sorted(data['COD_PUESTO'].unique())
            mesas = sorted(data['MESA'].unique())
            generos = sorted(data['GÉNERO'].unique())
            ideologias = sorted(data['IDEOLOGÍA'].unique())
            
            # Inputs para el usuario
            zona = st.selectbox("Zona:", zonas)
            puesto = st.selectbox("Puesto de votación:", puestos)
            mesa = st.selectbox("Mesa:", mesas)
            genero = st.selectbox("Género:", generos)
            ideologia = st.selectbox("Ideología:", ideologias)
            
            # Sliders para variables numéricas
            edad = st.slider("Edad:", min_value=30, max_value=80, value=60, step=1)
            
            # Radio buttons para opciones binarias
            cambio_politicas = st.radio("¿Propone cambio de políticas?", ["Si", "No"])
            
            anos_trayectoria = st.slider("Años de trayectoria:", min_value=5, max_value=30, value=15, step=1)
            apoyo_mayoritario = st.radio("¿Apoyado por partidos mayoritarios?", ["Si", "No"])
            
            # Slider para el porcentaje en encuesta
            encuesta = st.slider("Porcentaje en encuesta:", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
            
            # Botón para hacer la predicción
            predict_button = st.form_submit_button("Predecir")
        else:
            st.error("No se pudo cargar el modelo o los datos. Por favor verifica que los archivos existan y sean accesibles.")
            predict_button = st.form_submit_button("Predecir", disabled=True)



# : Implementar la función de predicción
# Función para realizar la predicción
def predict_votes(input_features, model, train_data):
    """
    Realiza la predicción de votos utilizando el modelo cargado.
    
    Args:
        input_features: DataFrame con los parámetros ingresados por el usuario.
        model: Modelo XGBoost cargado.
        train_data: DataFrame con los datos de entrenamiento para preprocesamiento.
        
    Returns:
        int: Número de votos predicho.
    """
    try:
        # Preprocesar las características de entrada
        processed_features = preprocess_input(input_features, train_data)
        
        # Realizar la predicción
        prediction = model.predict(processed_features)[0]
        
        # Redondear al entero más cercano ya que los votos son valores enteros
        votes = int(round(prediction))
        
        # Asegurar que no tengamos valores negativos
        votes = max(0, votes)
        
        return votes
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        return None

# Panel derecho - Mostrar los resultados si se hizo clic en el botón de predicción
with col2:
    if predict_button and model_loaded:
        st.header("Resultados de la Predicción")
        
        # Crear un DataFrame con los datos ingresados
        input_data = pd.DataFrame({
            'ZONA': [zona],
            'COD_PUESTO': [puesto],
            'MESA': [mesa],
            'GÉNERO': [genero],
            '# CANDIDATOS': [5],  # Valor fijo según el dataset
            'IDEOLOGÍA': [ideologia],
            'EDAD': [edad],
            'CAMBIO DE POLÍTICAS': [cambio_politicas],
            'AÑOS DE TRAYECTORIA': [anos_trayectoria],
            'APOYADO POR PARTIDOS MAYORITARIOS': [apoyo_mayoritario],
            'ENCUESTA': [encuesta]
        })
        
        # Realizar la predicción
        predicted_votes = predict_votes(input_data, model, data)
        
        if predicted_votes is not None:
            # Crear un contenedor para mostrar el resultado principal
            result_container = st.container()
            
            with result_container:
                # Mostrar el resultado con un formato destacado
                st.markdown(f"## 🗳️ Predicción: {predicted_votes} votos")
                
                # Calcular el porcentaje para la barra de progreso basado en el máximo de votos
                max_votes = int(data['VOTOS'].max())
                progress_pct = min(1.0, predicted_votes / max_votes)
                
                # Determinar el color según el porcentaje
                if progress_pct > 0.7:
                    bar_color = "green"
                elif progress_pct > 0.3:
                    bar_color = "orange"
                else:
                    bar_color = "red"
                
                # Crear una barra de progreso simple
                st.markdown(f"### Escala de Votos (máximo: {max_votes})")
                st.progress(progress_pct)
                
                # Texto interpretativo sobre la posición en la escala
                if progress_pct > 0.7:
                    st.markdown("🌟 **Resultado alto** en la escala de votos")
                elif progress_pct > 0.3:
                    st.markdown("✅ **Resultado medio** en la escala de votos")
                else:
                    st.markdown("⚠️ **Resultado bajo** en la escala de votos")
                
                # Mostrar información contextual sobre el resultado
                avg_votes = int(data['VOTOS'].mean())
                median_votes = int(data['VOTOS'].median())
                
                # Crear columnas para estadísticas comparativas
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    st.metric(label="Predicción", value=f"{predicted_votes}", 
                             delta=f"{predicted_votes - avg_votes} vs promedio")
                
                with stat_col2:
                    st.metric(label="Promedio", value=f"{avg_votes}")
                    
                with stat_col3:
                    st.metric(label="Mediana", value=f"{median_votes}")
                
                # Añadir interpretación del resultado
                if predicted_votes > avg_votes * 1.5:
                    st.success("🌟 **Resultado excepcional**: La predicción sugiere un rendimiento muy por encima del promedio.")
                elif predicted_votes > avg_votes:
                    st.success("✅ **Buen resultado**: La predicción está por encima del promedio de votos.")
                elif predicted_votes > avg_votes * 0.5:
                    st.warning("⚠️ **Resultado moderado**: La predicción está por debajo del promedio pero dentro de rangos normales.")
                else:
                    st.error("❗ **Resultado bajo**: La predicción muestra un número de votos significativamente por debajo del promedio.")
                
                # Mostrar información adicional
                with st.expander("Más información sobre esta predicción"):
                    st.info("""
                    Esta predicción se basa en un modelo de XGBoost optimizado, entrenado 
                    con datos históricos de las elecciones de Barranquilla 2014.
                    
                    La precisión general del modelo es:
                    - **R² (coeficiente de determinación)**: 0.80, lo que significa que el modelo explica 
                      aproximadamente el 80% de la variabilidad en el número de votos.
                    - **RMSE (error cuadrático medio)**: 5.94 votos, que representa el error promedio en las predicciones.
                    """)
                    
                # Generar un gráfico de barras simple para visualizar la predicción y compararla
                st.subheader("Comparación con promedios")
                
                comparison_data = pd.DataFrame({
                    'Categoría': ['Predicción', 'Promedio', 'Mediana'],
                    'Votos': [predicted_votes, avg_votes, median_votes]
                })
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(
                    comparison_data['Categoría'],
                    comparison_data['Votos'],
                    color=['crimson', 'steelblue', 'lightgreen']
                )
                
                # Añadir etiquetas con valores
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.5,
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontweight='bold'
                    )
                
                ax.set_ylabel('Votos')
                ax.set_title('Comparación de la predicción con valores de referencia')
                
                st.pyplot(fig)
        else:
            st.error("No se pudo realizar la predicción. Verifica los parámetros ingresados.")







#_________________________________________________________________________________
# BLOQUE HORIZONTAL DEBAJO

# Paso 4: Análisis de características (layout horizontal)

# Este código debe ir DESPUÉS de todos los bloques anteriores de visualización
# Se coloca como una nueva sección completa debajo de los resultados principales

# Añadir un separador para la siguiente sección
st.markdown("---")
st.header("Análisis de Características del Modelo")
st.markdown("Esta sección te permite entender qué factores tienen mayor influencia en la predicción y cómo cambiaría el resultado al modificarlos.")

# Obtener la importancia de características del modelo
feature_importance = pd.DataFrame({
    'Feature': model.feature_names_in_,  # Accedemos a los nombres de características
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Definir una función para traducir nombres de características a nombres más amigables
def get_feature_display_name(feature_name):
    """Convierte nombres de características técnicos a nombres más amigables para mostrar."""
    name_mapping = {
        'ZONA': 'Zona Electoral',
        'COD_PUESTO': 'Código de Puesto',
        'MESA': 'Mesa Electoral',
        'GÉNERO': 'Género',
        '# CANDIDATOS': 'Número de Candidatos',
        'IDEOLOGÍA': 'Ideología Política',
        'EDAD': 'Edad del Candidato',
        'CAMBIO DE POLÍTICAS': 'Propone Cambio de Políticas',
        'AÑOS DE TRAYECTORIA': 'Años de Trayectoria',
        'APOYADO POR PARTIDOS MAYORITARIOS': 'Apoyo de Partidos Mayoritarios',
        'ENCUESTA': 'Porcentaje en Encuestas'
    }
    return name_mapping.get(feature_name, feature_name)

# Aplicar la función para obtener nombres más amigables
feature_importance['Display Name'] = feature_importance['Feature'].apply(get_feature_display_name)

# Mostrar solo las 5 características más importantes para no sobrecargar la visualización
top_features = feature_importance.head(5).copy()

# Crear dos columnas para el layout horizontal
col_importance, col_sensitivity = st.columns([1, 1])

# Primera columna: Importancia de características
with col_importance:
    st.subheader("Importancia de Características")
    
    # Crear visualización de la importancia de características
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Crear barras horizontales para la importancia de características
    bars = ax.barh(
        top_features['Display Name'],
        top_features['Importance'],
        color='skyblue'
    )
    
    # Añadir etiquetas y título
    ax.set_xlabel('Importancia Relativa')
    ax.set_ylabel('Característica')
    ax.set_title('Las 5 Características Más Importantes')
    
    # Añadir valores numéricos a las barras
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01, 
            bar.get_y() + bar.get_height()/2., 
            f'{width:.3f}',
            ha='left', 
            va='center',
            fontweight='bold'
        )
    
    # Invertir el eje y para que la característica más importante esté arriba
    ax.invert_yaxis()
    
    # Ajustar layout
    fig.tight_layout()
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)
    
    # Mostrar interpretación general
    st.markdown("""
    ### Interpretación
    - **Mayor importancia**: Las características en la parte superior tienen mayor influencia en la predicción
    - **Influencia porcentual**: Los valores numéricos representan el peso relativo de cada factor
    - **Enfoque estratégico**: Concentrar esfuerzos en mejorar los factores más importantes
    """)

# Segunda columna: Análisis de sensibilidad
with col_sensitivity:
    st.subheader("Análisis de Sensibilidad")
    
    # Filtrar solo características numéricas importantes
    numeric_features = [f for f in top_features['Feature'] if f in ['ZONA', 'COD_PUESTO', 'MESA', 'EDAD', 'AÑOS DE TRAYECTORIA', 'ENCUESTA']]
    
    # Crear una función para simular cambios en las características
    def simulate_feature_change(input_df, feature_name, change_pct, model, train_data):
        """
        Simula el cambio en la predicción al modificar una característica.
        """
        # Crear una copia para no modificar los datos originales
        modified_df = input_df.copy()
        
        # Obtener el valor original
        original_value = modified_df[feature_name].iloc[0]
        
        # Calcular el nuevo valor (solo para características numéricas)
        if feature_name in numeric_features:
            # Aplicar el cambio porcentual
            new_value = original_value * (1 + change_pct/100)
            modified_df[feature_name] = new_value
        
        # Preprocesar ambos conjuntos de datos
        processed_original = preprocess_input(input_df, train_data)
        processed_modified = preprocess_input(modified_df, train_data)
        
        # Realizar predicciones
        original_pred = model.predict(processed_original)[0]
        modified_pred = model.predict(processed_modified)[0]
        
        return round(original_pred), round(modified_pred)
    
    # Si tenemos características numéricas importantes, mostrar análisis de sensibilidad
    if numeric_features:
        st.markdown("Selecciona una característica y ajusta su valor para ver cómo afecta a la predicción:")
        
        # Selector para elegir la característica a analizar
        selected_feature = st.selectbox(
            "Característica a modificar:",
            options=numeric_features,
            format_func=get_feature_display_name
        )
        
        # Slider para ajustar el porcentaje de cambio
        change_pct = st.slider(
            f"Modificar {get_feature_display_name(selected_feature)} (%)", 
            min_value=-50, 
            max_value=50, 
            value=0, 
            step=5
        )
        
        # Mostrar el valor original y el modificado
        original_val = input_data[selected_feature].iloc[0]
        modified_val = original_val * (1 + change_pct/100)
        
        # Formato para mostrar valores con la precisión adecuada
        if selected_feature == 'ENCUESTA':
            st.markdown(f"**Valor original:** {original_val:.1f}% → **Valor modificado:** {modified_val:.1f}%")
        else:
            st.markdown(f"**Valor original:** {int(original_val)} → **Valor modificado:** {int(modified_val)}")
        
        # Calcular el efecto del cambio
        if change_pct != 0:
            original, modified = simulate_feature_change(
                input_data, selected_feature, change_pct, model, data
            )
            
            # Mostrar el efecto del cambio
            change = modified - original
            change_text = f"+{change}" if change > 0 else f"{change}"
            
            # Crear columnas para mostrar valores actual y modificado
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Votos originales", 
                    value=original
                )
            
            with col2:
                st.metric(
                    label="Votos con cambio", 
                    value=modified,
                    delta=change_text
                )
            
            # Visualizar el cambio en un gráfico
            compare_data = pd.DataFrame({
                'Escenario': ['Original', 'Modificado'],
                'Votos': [original, modified]
            })
            
            fig, ax = plt.subplots()
            bars = ax.bar(
                compare_data['Escenario'],
                compare_data['Votos'],
                color=['blue', 'orange']
            )
            
            # Añadir etiquetas con valores
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.3,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )
            
            ax.set_ylabel('Votos')
            ax.set_title(f'Efecto de cambiar {get_feature_display_name(selected_feature)}')
            
            # Ajustar el rango del eje y para mejor visualización
            max_value = max(original, modified) * 1.1
            ax.set_ylim(0, max_value)
            
            st.pyplot(fig)
            
            # Añadir interpretación
            impact_level = ""
            if abs(change) > 5:
                impact_level = "**muy significativo**"
            elif abs(change) > 2:
                impact_level = "**moderado**"
            else:
                impact_level = "**limitado**"
            
            st.markdown(f"El cambio del {change_pct}% en '{get_feature_display_name(selected_feature)}' tiene un impacto {impact_level} en la predicción ({change_text} votos).")
        else:
            st.info("Ajusta el porcentaje para ver el impacto en la predicción.")
    else:
        st.info("No hay características numéricas entre las más importantes para realizar este análisis.")

# Sección final con recomendaciones - Estilo horizontal con toda la anchura de la página
st.markdown("---")
st.subheader("Recomendaciones Estratégicas")

# Crear layout de 3 columnas para recomendaciones
rec_col1, rec_col2, rec_col3 = st.columns(3)

# Obtener las 3 características más importantes
top_3_features = feature_importance.head(3)['Feature'].tolist()
top_3_names = feature_importance.head(3)['Display Name'].tolist()

# Primera columna de recomendación
with rec_col1:
    feature = top_3_features[0] if len(top_3_features) > 0 else ""
    display_name = top_3_names[0] if len(top_3_names) > 0 else ""
    
    st.markdown(f"### 1️⃣ {display_name}")
    
    if feature == 'ENCUESTA':
        st.markdown("""
        📊 **Recomendación:**
        - Invertir en campañas de visibilidad pública
        - Trabajar con grupos de encuestas reconocidos
        - Aumentar presencia en medios y redes sociales
        """)
    elif feature == 'IDEOLOGÍA':
        st.markdown("""
        🧭 **Recomendación:**
        - Clarificar mensajes alineados con la ideología
        - Fortalecer relaciones con grupos afines
        - Crear propuestas emblemáticas ideológicas
        """)
    elif feature in ['ZONA', 'COD_PUESTO', 'MESA']:
        st.markdown("""
        🗺️ **Recomendación:**
        - Identificar zonas de mayor potencial electoral
        - Enfocar recursos en áreas prioritarias
        - Desarrollar estrategias por localidad
        """)
    else:
        st.markdown(f"""
        ⭐ **Recomendación:**
        - Factor de alto impacto en resultados
        - Desarrollar estrategias específicas
        - Monitorear constantemente esta variable
        """)

# Segunda columna de recomendación
with rec_col2:
    feature = top_3_features[1] if len(top_3_features) > 1 else ""
    display_name = top_3_names[1] if len(top_3_names) > 1 else ""
    
    if feature:
        st.markdown(f"### 2️⃣ {display_name}")
        
        if feature == 'ENCUESTA':
            st.markdown("""
            📊 **Recomendación:**
            - Mejorar posicionamiento en encuestas
            - Identificar factores que influyen en percepción
            - Segmentar mensajes por grupos demográficos
            """)
        elif feature == 'IDEOLOGÍA':
            st.markdown("""
            🧭 **Recomendación:**
            - Definir postura clara en temas clave
            - Destacar diferencias con oponentes
            - Mantener coherencia en comunicaciones
            """)
        elif feature in ['ZONA', 'COD_PUESTO', 'MESA']:
            st.markdown("""
            🗺️ **Recomendación:**
            - Analizar patrones de votación por zona
            - Reforzar presencia en áreas estratégicas
            - Adaptar mensajes a realidades locales
            """)
        else:
            st.markdown(f"""
            ⭐ **Recomendación:**
            - Optimizar este factor para mejorar resultados
            - Comparar con valores de candidatos exitosos
            - Establecer metas incrementales
            """)

# Tercera columna de recomendación
with rec_col3:
    feature = top_3_features[2] if len(top_3_features) > 2 else ""
    display_name = top_3_names[2] if len(top_3_names) > 2 else ""
    
    if feature:
        st.markdown(f"### 3️⃣ {display_name}")
        
        if feature == 'ENCUESTA':
            st.markdown("""
            📊 **Recomendación:**
            - Realizar encuestas internas periódicas
            - Identificar tendencias y cambios de opinión
            - Ajustar estrategia según resultados
            """)
        elif feature == 'IDEOLOGÍA':
            st.markdown("""
            🧭 **Recomendación:**
            - Alinear propuestas con valores ideológicos
            - Construir coaliciones con grupos afines
            - Evitar contradicciones en discurso
            """)
        elif feature in ['ZONA', 'COD_PUESTO', 'MESA']:
            st.markdown("""
            🗺️ **Recomendación:**
            - Implementar estrategias territoriales
            - Formar equipos locales de apoyo
            - Monitorear competencia por zonas
            """)
        else:
            st.markdown(f"""
            ⭐ **Recomendación:**
            - Evaluar el impacto de cambios en este factor
            - Implementar mejoras graduales
            - Comunicar efectivamente los avances
            """)