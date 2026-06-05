import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from pathlib import Path


st.set_page_config(
    page_title="Clasificación de Intensidad de Mortalidad COVID-19",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mlp_model.keras"
CONFIG_PATH = BASE_DIR / "models" / "preprocessing_config.json"
DATA_PATH = BASE_DIR / "data_processed" / "dataset_modelado.csv"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value):
    return str(value).strip().upper()


def transform_input_manual(input_df, config):
    """
    Replica manualmente el preprocesamiento usado en Colab:
    - OneHotEncoder para variables categóricas.
    - StandardScaler para variables numéricas.
    Esto evita errores de incompatibilidad de scikit-learn/joblib.
    """

    output_values = []

    cat_cols = config["cat_cols"]
    num_cols = config["num_cols"]
    categories = config["categories"]
    scaler_mean = config["scaler_mean"]
    scaler_scale = config["scaler_scale"]

    # One-Hot manual en el mismo orden del entrenamiento
    for col, category_list in zip(cat_cols, categories):
        value = normalize_text(input_df.iloc[0][col])

        for category in category_list:
            output_values.append(1.0 if value == normalize_text(category) else 0.0)

    # StandardScaler manual
    for i, col in enumerate(num_cols):
        value = float(input_df.iloc[0][col])
        scaled_value = (value - scaler_mean[i]) / scaler_scale[i]
        output_values.append(scaled_value)

    return np.array([output_values], dtype=np.float32)


try:
    model = load_model()
    df_model = load_data()
    config = load_config()
except Exception as e:
    st.error("No se pudieron cargar los recursos necesarios para la demo.")
    st.write("Verifica que existan:")
    st.code(
        """
models/mlp_model.keras
models/preprocessing_config.json
data_processed/dataset_modelado.csv
        """
    )
    st.exception(e)
    st.stop()


st.title("Clasificación de Patrones de Intensidad de Mortalidad por COVID-19")
st.subheader("Modelo Deep Learning MLP — Fase 3")

st.info(
    "Esta demo no predice si una persona fallecerá o sobrevivirá. "
    "El sistema clasifica patrones históricos de intensidad de mortalidad "
    "en grupos demográficos, geográficos y temporales."
)

st.caption(
    f"El modelo espera {model.input_shape[-1]} características después del preprocesamiento."
)

col1, col2 = st.columns([1, 1])


with col1:
    st.markdown("### Ingreso de variables")

    # Opciones desde el dataset procesado
    departamentos = sorted(df_model["DEPARTAMENTO"].dropna().astype(str).unique())
    departamento = st.selectbox("Departamento", departamentos)

    provincias_filtradas = sorted(
        df_model[df_model["DEPARTAMENTO"].astype(str) == str(departamento)]["PROVINCIA"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not provincias_filtradas:
        provincias_filtradas = sorted(df_model["PROVINCIA"].dropna().astype(str).unique())

    provincia = st.selectbox("Provincia", provincias_filtradas)

    sexo = st.selectbox(
        "Sexo",
        sorted(df_model["SEXO"].dropna().astype(str).unique())
    )

    rango_edad = st.selectbox(
        "Rango de edad",
        ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]
    )

    mes_fallecimiento = st.selectbox(
        "Mes de fallecimiento",
        list(range(1, 13)),
        format_func=lambda x: {
            1: "Enero",
            2: "Febrero",
            3: "Marzo",
            4: "Abril",
            5: "Mayo",
            6: "Junio",
            7: "Julio",
            8: "Agosto",
            9: "Septiembre",
            10: "Octubre",
            11: "Noviembre",
            12: "Diciembre"
        }[x]
    )

    predict_button = st.button("Clasificar intensidad")


with col2:
    st.markdown("### Resultado del modelo")

    if predict_button:
        input_data = pd.DataFrame([{
            "DEPARTAMENTO": normalize_text(departamento),
            "PROVINCIA": normalize_text(provincia),
            "SEXO": normalize_text(sexo),
            "rango_edad": str(rango_edad),
            "mes_fallecimiento": int(mes_fallecimiento)
        }])

        X_input = transform_input_manual(input_data, config)

        st.caption(f"Características generadas para la predicción: {X_input.shape[1]}")

        if X_input.shape[1] != model.input_shape[-1]:
            st.error(
                f"Error de dimensiones: el modelo espera {model.input_shape[-1]} columnas, "
                f"pero se generaron {X_input.shape[1]}."
            )
            st.stop()

        probabilities = model.predict(X_input, verbose=0)

        predicted_class_index = np.argmax(probabilities, axis=1)[0]
        predicted_class = config["classes"][predicted_class_index]

        confidence = probabilities[0][predicted_class_index]

        if predicted_class == "ALTO":
            st.error(f"Nivel estimado de intensidad: {predicted_class}")
        elif predicted_class == "MEDIO":
            st.warning(f"Nivel estimado de intensidad: {predicted_class}")
        else:
            st.success(f"Nivel estimado de intensidad: {predicted_class}")

        st.metric("Confianza del modelo", f"{confidence:.2%}")

        prob_df = pd.DataFrame({
            "Clase": config["classes"],
            "Probabilidad": probabilities[0]
        })

        st.markdown("#### Probabilidades por clase")
        st.dataframe(prob_df, use_container_width=True)
        st.bar_chart(prob_df.set_index("Clase"))

    else:
        st.write("Completa las variables y presiona **Clasificar intensidad**.")


st.divider()

st.markdown("### Información metodológica")

st.write(
    """
    El modelo fue entrenado a partir de registros oficiales de defunciones por COVID-19 en el Perú.
    Debido a que el dataset contiene únicamente personas fallecidas, el problema fue reformulado
    como clasificación de patrones de intensidad de mortalidad y no como predicción individual
    de muerte o supervivencia.
    """
)

st.markdown(
    """
    **Variables utilizadas por el modelo:**
    - Departamento
    - Provincia
    - Sexo
    - Rango de edad
    - Mes de fallecimiento

    **Clases de salida:**
    - BAJO
    - MEDIO
    - ALTO
    """
)