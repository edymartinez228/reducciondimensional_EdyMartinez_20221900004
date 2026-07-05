import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

# Configuración de la página de Streamlit
st.set_page_config(page_title="MNIST PCA-KMeans-SVM", layout="wide")

st.title("🔢 Clasificador de Dígitos MNIST con Reducción de Dimensionalidad")
st.markdown("""
Esta aplicación demuestra el flujo completo de un proyecto de IA:
1. **PCA** para reducir las dimensiones de imágenes de 28x28 (784 píxeles).
2. **K-Means** para agrupar visualmente los dígitos de forma no supervisada.
3. **SVM (Support Vector Machine)** para clasificar y predecir el dígito real.
""")

# Funciones para cargar modelos y datos de prueba optimizados
@st.cache_resource
def cargar_recursos():
    try:
        pca = joblib.load("models/pca_mnist.pkl")
        kmeans = joblib.load("models/kmeans_mnist.pkl")
        svm = joblib.load("models/svm_mnist.pkl")
        
        # Cargar imágenes de muestra pre-guardadas por el notebook
        X_test_muestras = np.load("outputs/X_test_muestras.npy")
        y_test_muestras = np.load("outputs/y_test_muestras.npy")
        
        # Cargar metadatos
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
            
        return pca, kmeans, svm, X_test_muestras, y_test_muestras, metadata
    except Exception as e:
        st.error(f"Error al cargar los archivos del modelo: {e}")
        return None, None, None, None, None, None

pca, kmeans, svm, X_test, y_test, meta = cargar_recursos()

if pca is not None:
    # --- BARRA LATERAL (CONTROLES) ---
    st.sidebar.header("⚙️ Configuración del Modelo")
    
    # El usuario selecciona el número de componentes a usar en la app (máximo lo entrenado en el notebook)
    max_comp = meta.get("n_componentes_max", 50)
    n_componentes = st.sidebar.slider(
        "Componentes Principales (PCA)", 
        min_value=2, 
        max_value=max_comp, 
        value=15,
        help="Número de componentes para limitar el análisis en la visualización o inferencia."
    )
    
    # --- PANEL PRINCIPAL: MÉTRICAS Y COMPARATIVA ---
    st.header("📈 Desempeño y Métricas Generales")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Exactitud Global del Modelo (SVM Accuracy)", value=f"{meta['accuracy_test']*100:.2f}%")
    with col_m2:
        st.metric(label="Muestras Utilizadas para Entrenamiento", value=f"{meta['muestras_entrenamiento']} imágenes")
        
    # --- SECCIÓN DE VISUALIZACIÓN EN 2D (REQUERIMIENTO DE LA RÚBRICA) ---
    st.header("🖼️ Proyección de Datos y Clústeres (K-Means)")
    
    # Reducimos una porción pequeña de test a los componentes elegidos para graficar rápido
    X_test_pca_full = pca.transform(X_test)
    X_test_pca_reducido = X_test_pca_full[:, :n_componentes]
    
    # Asignar grupos de K-means a estos datos proyectados
    clusters_test = kmeans.predict(X_test_pca_full) # K-Means original requiere las dimensiones completas de su entreno
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Proyección PCA (Primeros 2 Componentes)")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        scatter1 = ax1.scatter(X_test_pca_full[:, 0], X_test_pca_full[:, 1], c=y_test, cmap="tab10", alpha=0.6, s=15)
        fig1.colorbar(scatter1, ax=ax1, label="Dígito Real (0-9)")
        ax1.set_xlabel("Componente Principal 1")
        ax1.set_ylabel("Componente Principal 2")
        st.pyplot(fig1)
        st.caption("Distribución de las imágenes basada en sus etiquetas reales.")

    with col_g2:
        st.subheader("Grupos Encontrados por K-Means")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        scatter2 = ax2.scatter(X_test_pca_full[:, 0], X_test_pca_full[:, 1], c=clusters_test, cmap="Set1", alpha=0.6, s=15)
        fig2.colorbar(scatter2, ax=ax2, label="Clúster Asignado")
        ax2.set_xlabel("Componente Principal 1")
        ax2.set_ylabel("Componente Principal 2")
        st.pyplot(fig2)
        st.caption("Agrupación puramente matemática realizada por K-Means de forma no supervisada.")


    # --- SECCIÓN DE CLASIFICACIÓN INTERACTIVA (REQUERIMIENTO DE LA RÚBRICA) ---
    st.header("🎯 Prueba Interactiva de Clasificación (SVM)")
    st.markdown("Selecciona un índice de imagen para extraerla del conjunto de prueba, ver cómo la "
                "procesa el modelo y observar la predicción final de la Máquina de Vectores de Soporte.")
    
    idx_imagen = st.number_input("Selecciona el índice de la imagen (0 a 299):", min_value=0, max_value=299, value=42, step=1)
    
    col_img, col_pred = st.columns([1, 2])
    
    with col_img:
        st.write("**Imagen Original (28x28 píxeles):**")
        # Reconstruir la matriz de 28x28 para graficar el dígito manuscrito
        imagen_matriz = X_test[idx_imagen].reshape(28, 28)
        
        fig_img, ax_img = plt.subplots(figsize=(3, 3))
        ax_img.imshow(imagen_matriz, cmap="gray")
        ax_img.axis("off")
        st.pyplot(fig_img)
        
    with col_pred:
        st.write("**Análisis de Predicción:**")
        
        # Extraer características reducidas correspondientes a esta imagen en específico
        vector_pca_completo = X_test_pca_full[idx_imagen].reshape(1, -1)
        
        # Clasificar usando el modelo SVM entrenado
        clase_predicha = svm.predict(vector_pca_completo)[0]
        clase_real = y_test[idx_imagen]
        cluster_perteneciente = clusters_test[idx_imagen]
        
        st.info(f"🔹 **Clúster asignado por K-Means:** Grupo {cluster_perteneciente}")
        
        if clase_predicha == clase_real:
            st.success(f"✅ **Predicción SVM:** ¡Dígito **{clase_predicha}**!")
            st.write(f"El modelo clasificó correctamente la imagen. (Clase Real: {clase_real})")
        else:
            st.error(f"❌ **Predicción SVM:** Dígito **{clase_predicha}**")
            st.write(f"El modelo falló en esta imagen. (Clase Real original: {clase_real})")
            
else:
    st.warning("Por favor, asegúrate de correr el notebook primero para generar los archivos requeridos en la carpeta 'models/' y 'outputs/'.")
