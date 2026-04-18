# app.py
import streamlit as st
import numpy as np
import pandas as pd

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Estadístico - Prueba Z",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Análisis Estadístico · Prueba Z")
st.markdown("Proyecto de estadística inferencial con visualización e IA interpretativa.")

# ── Sidebar: fuente de datos ─────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración de datos")

fuente = st.sidebar.radio(
    "Fuente de datos",
    ["Generar datos sintéticos", "Cargar CSV"],
)

data = None  # se llenará en cada rama

# ── Rama A: datos sintéticos ─────────────────────────────────────────────────
if fuente == "Generar datos sintéticos":
    st.sidebar.subheader("Parámetros de la distribución")

    mu = st.sidebar.number_input(
        "Media (μ)", value=50.0, step=1.0,
        help="Media poblacional de la distribución normal."
    )
    sigma = st.sidebar.number_input(
        "Desviación estándar (σ)", value=10.0, min_value=0.1, step=0.5,
        help="Desviación estándar; debe ser > 0."
    )
    n = st.sidebar.slider(
        "Tamaño de muestra (n)", min_value=30, max_value=2000,
        value=200, step=10,
        help="Mínimo 30 para cumplir el supuesto de la Prueba Z."
    )
    seed = st.sidebar.number_input(
        "Semilla aleatoria", value=42, step=1,
        help="Fija la semilla para reproducibilidad."
    )

    if st.sidebar.button("🎲 Generar datos"):
        rng = np.random.default_rng(int(seed))
        muestra = rng.normal(loc=mu, scale=sigma, size=n)
        data = pd.DataFrame({"valor": muestra})
        st.session_state["data"] = data
        st.session_state["sigma_pob"] = sigma   # varianza poblacional conocida
        st.sidebar.success(f"✅ {n} observaciones generadas.")

# ── Rama B: carga de CSV ─────────────────────────────────────────────────────
else:
    archivo = st.sidebar.file_uploader(
        "Sube tu archivo CSV", type=["csv"],
        help="El archivo debe tener al menos una columna numérica."
    )

    if archivo:
        try:
            df_raw = pd.read_csv(archivo)
            columnas_num = df_raw.select_dtypes(include="number").columns.tolist()

            if not columnas_num:
                st.sidebar.error("❌ El CSV no contiene columnas numéricas.")
            else:
                col_sel = st.sidebar.selectbox(
                    "Selecciona la columna de análisis", columnas_num
                )
                sigma_usuario = st.sidebar.number_input(
                    "Desviación estándar poblacional conocida (σ)",
                    value=1.0, min_value=0.01, step=0.1,
                    help="Requerida para la Prueba Z."
                )

                if st.sidebar.button("📂 Cargar columna"):
                    muestra = df_raw[col_sel].dropna().values
                    if len(muestra) < 30:
                        st.sidebar.warning(
                            f"⚠️ n = {len(muestra)} < 30. "
                            "Los resultados de la Prueba Z pueden no ser válidos."
                        )
                    data = pd.DataFrame({"valor": muestra})
                    st.session_state["data"] = data
                    st.session_state["sigma_pob"] = sigma_usuario
                    st.sidebar.success(
                        f"✅ {len(muestra)} observaciones cargadas desde '{col_sel}'."
                    )
        except Exception as e:
            st.sidebar.error(f"Error al leer el CSV: {e}")

# ── Panel principal: previsualización ────────────────────────────────────────
if "data" in st.session_state:
    data = st.session_state["data"]

    st.subheader("🔍 Vista previa de los datos")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observaciones (n)", len(data))
    col2.metric("Media muestral (x̄)", f"{data['valor'].mean():.4f}")
    col3.metric("Desv. estándar muestral (s)", f"{data['valor'].std(ddof=1):.4f}")
    col4.metric("σ poblacional (conocida)", f"{st.session_state['sigma_pob']:.4f}")

    st.dataframe(
        data.describe().T.rename(columns=str.title),
        use_container_width=True,
    )
else:
    st.info(
        "👈 Configura la fuente de datos en el panel lateral y pulsa el botón "
        "correspondiente para comenzar."
    )