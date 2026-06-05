# Clasificación de patrones de intensidad de mortalidad por COVID-19 en el Perú

Proyecto académico desarrollado para la asignatura **Inteligencia Artificial** de la Escuela Profesional de Ingeniería de Sistemas de la Universidad Andina del Cusco.

## Descripción del proyecto

Este proyecto analiza registros oficiales de defunciones por COVID-19 en el Perú con el objetivo de clasificar patrones históricos de intensidad de mortalidad utilizando técnicas de Machine Learning y Deep Learning.

Debido a que el dataset contiene únicamente registros de personas fallecidas, el sistema **no predice muerte o supervivencia individual**. En su lugar, el problema fue reformulado como una clasificación de patrones de intensidad de mortalidad en tres categorías:

* **BAJO**
* **MEDIO**
* **ALTO**

La clasificación se realiza a partir de variables demográficas, geográficas y temporales, como departamento, provincia, sexo, rango de edad y mes de fallecimiento.

## Objetivo general

Implementar un sistema basado en Inteligencia Artificial que permita clasificar patrones de intensidad de mortalidad por COVID-19 en el Perú, integrando modelos de Machine Learning, Deep Learning y una interfaz funcional de demostración.

## Variables utilizadas

Las principales variables empleadas en el modelo son:

* Departamento
* Provincia
* Sexo
* Rango de edad
* Mes de fallecimiento

A partir de estas variables se generaron grupos demográficos, geográficos y temporales. Luego se calculó la cantidad de fallecidos por grupo y se creó la variable objetivo `nivel_intensidad`.

## Modelos implementados

| Modelo              | Tipo                         | Rol en el proyecto                             |
| ------------------- | ---------------------------- | ---------------------------------------------- |
| Random Forest       | Machine Learning supervisado | Línea base de la Fase 1                        |
| MLP                 | Deep Learning                | Modelo principal de la Fase 2                  |
| CNN 1D              | Deep Learning alternativo    | Modelo comparativo                             |
| Transformer tabular | Exploración Fase 3           | Modelo basado en atención para datos tabulares |

## Resultados principales

El modelo MLP fue el mejor modelo confirmado durante la Fase 2.

| Modelo              |       Accuracy | F1 Macro | F1 Weighted | Observación             |
| ------------------- | -------------: | -------: | ----------: | ----------------------- |
| MLP                 |           0.68 |     0.65 |        0.67 | Mejor modelo confirmado |
| CNN 1D              |           0.33 |     0.17 |        0.17 | Bajo desempeño          |
| Random Forest       |     Línea base |        - |           - | Modelo de referencia    |
| Transformer tabular | En exploración |        - |           - | Fase 3                  |

El modelo MLP presentó mejor desempeño en las clases **ALTO** y **BAJO**, mientras que la clase **MEDIO** fue la más difícil de clasificar debido a su naturaleza intermedia.

## Estructura del repositorio

```text
FASE2/
│
├── app/
│   └── streamlit_app.py
│
├── data_processed/
│   └── dataset_modelado.csv
│
├── models/
│   ├── mlp_model.keras
│   ├── preprocessing_config.json
│   └── otros modelos o archivos auxiliares
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Streamlit
* GitHub

## Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/Ritual01/RandomForestGrupo4
cd RandomForestGrupo4
```

### 2. Crear entorno virtual

En Windows:

```bash
python -m venv venv
```

### 3. Activar entorno virtual

```bash
venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecutar la aplicación Streamlit

```bash
python -m streamlit run app/streamlit_app.py
```

La aplicación se abrirá normalmente en:

```text
http://localhost:8501
```

## Funcionamiento de la demo

La demo permite ingresar variables como:

* Departamento
* Provincia
* Sexo
* Rango de edad
* Mes de fallecimiento

Luego, el sistema procesa los datos y devuelve una clasificación estimada:

```text
BAJO / MEDIO / ALTO
```

Además, muestra las probabilidades asociadas a cada clase.

## Advertencia ética

Este sistema no debe interpretarse como una herramienta médica ni como un sistema de diagnóstico individual. El modelo no predice si una persona fallecerá o sobrevivirá.

Su propósito es académico y analítico: clasificar patrones históricos de intensidad de mortalidad a partir de datos agrupados. Los resultados deben analizarse con cuidado debido a posibles sesgos del dataset, subregistro de casos, falta de variables clínicas y diferencias regionales en el acceso a servicios de salud.

## Integrantes

* Yanin Mamani Gutierrez
* Andre Marcelo Manrique Rojas
* Diego Roberto Macedo Yañac

## Curso

**Inteligencia Artificial**
Universidad Andina del Cusco
Escuela Profesional de Ingeniería de Sistemas
