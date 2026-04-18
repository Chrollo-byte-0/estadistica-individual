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

    
# ── FASE 2: Análisis Exploratorio ────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

if "data" in st.session_state:
    data = st.session_state["data"]
    valores = data["valor"].values

    st.markdown("---")
    st.header("📈 Fase 2 · Análisis Exploratorio de Datos")

    # ── Controles de visualización ───────────────────────────────────────────
    with st.expander("🎨 Opciones de visualización", expanded=False):
        col_v1, col_v2, col_v3 = st.columns(3)
        n_bins = col_v1.slider("Número de bins (histograma)", 10, 100, 30, 5)
        color_hist = col_v2.color_picker("Color histograma", "#4F8EF7")
        color_box  = col_v3.color_picker("Color boxplot",    "#F7824F")

    # ── Estadísticos auxiliares ──────────────────────────────────────────────
    n        = len(valores)
    media    = np.mean(valores)
    mediana  = np.median(valores)
    s        = np.std(valores, ddof=1)
    skewness = stats.skew(valores)
    kurtosis = stats.kurtosis(valores)          # exceso de curtosis (Fisher)
    q1, q3   = np.percentile(valores, [25, 75])
    iqr      = q3 - q1
    lim_inf  = q1 - 1.5 * iqr
    lim_sup  = q3 + 1.5 * iqr
    outliers = valores[(valores < lim_inf) | (valores > lim_sup)]

    # ── Test de normalidad (Shapiro-Wilk o KS) ───────────────────────────────
    if n <= 5000:
        stat_norm, p_norm = stats.shapiro(valores)
        test_nombre = "Shapiro-Wilk"
    else:
        stat_norm, p_norm = stats.kstest(
            (valores - media) / s, "norm"
        )
        test_nombre = "Kolmogorov-Smirnov"

    # ── Figura principal: Histograma + KDE ───────────────────────────────────
    kde        = stats.gaussian_kde(valores)
    x_kde      = np.linspace(valores.min() - s, valores.max() + s, 400)
    y_kde      = kde(x_kde)

    # Escalar KDE al eje de frecuencia del histograma
    counts, bin_edges = np.histogram(valores, bins=n_bins)
    bin_width  = bin_edges[1] - bin_edges[0]
    y_kde_freq = y_kde * n * bin_width

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=("Histograma con KDE", "Boxplot"),
        horizontal_spacing=0.08,
    )

    # Histograma
    fig.add_trace(
        go.Histogram(
            x=valores, nbinsx=n_bins,
            name="Frecuencia",
            marker_color=color_hist,
            opacity=0.75,
            showlegend=True,
        ),
        row=1, col=1,
    )

    # KDE superpuesta (en escala de frecuencia)
    fig.add_trace(
        go.Scatter(
            x=x_kde, y=y_kde_freq,
            mode="lines", name="KDE",
            line=dict(color="#E63946", width=2.5),
        ),
        row=1, col=1,
    )

    # Líneas verticales: media y mediana
    for val, label, color_line in [
        (media,   "Media",   "#2EC4B6"),
        (mediana, "Mediana", "#FF9F1C"),
    ]:
        fig.add_vline(
            x=val, row=1, col=1,
            line=dict(color=color_line, dash="dash", width=1.8),
            annotation_text=f"{label}: {val:.2f}",
            annotation_position="top right",
            annotation_font_size=11,
        )

    # Boxplot
    fig.add_trace(
        go.Box(
            y=valores, name="Distribución",
            marker_color=color_box,
            boxmean="sd",
            boxpoints="outliers",
            jitter=0.3,
            pointpos=0,
            line_width=2,
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=480,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50, b=60, l=40, r=20),
        title_text="",
    )
    fig.update_xaxes(title_text="Valor", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig.update_yaxes(title_text="Valor", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # ── Q-Q Plot ─────────────────────────────────────────────────────────────
    with st.expander("📐 Q-Q Plot (verificación visual de normalidad)", expanded=False):
        qq = stats.probplot(valores, dist="norm")
        qq_x = np.array([pt[0] for pt in zip(qq[0][0], qq[0][0])])
        qq_theoretical = qq[0][0]
        qq_sample      = qq[0][1]
        slope, intercept = qq[1][0], qq[1][1]
        line_y = slope * qq_theoretical + intercept

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=qq_theoretical, y=qq_sample,
            mode="markers", name="Cuantiles",
            marker=dict(color=color_hist, size=5, opacity=0.8),
        ))
        fig_qq.add_trace(go.Scatter(
            x=qq_theoretical, y=line_y,
            mode="lines", name="Línea de referencia normal",
            line=dict(color="#E63946", width=2, dash="dot"),
        ))
        fig_qq.update_layout(
            height=380,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Cuantiles teóricos (Normal)",
            yaxis_title="Cuantiles muestrales",
            title="Q-Q Plot",
            margin=dict(t=50, b=50),
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    # ── Panel de diagnóstico ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🩺 Panel de Diagnóstico Automático")

    # ── Diagnóstico 1: Sesgo ─────────────────────────────────────────────────
    if abs(skewness) < 0.5:
        sesgo_label, sesgo_color, sesgo_txt = (
            "✅ Aproximadamente simétrica",
            "normal",
            f"El coeficiente de asimetría es **{skewness:.4f}** (|skew| < 0.5), "
            "indicando una distribución prácticamente simétrica.",
        )
    elif skewness >= 0.5:
        sesgo_label, sesgo_color, sesgo_txt = (
            "⚠️ Sesgo positivo (cola derecha)",
            "warning",
            f"El coeficiente de asimetría es **{skewness:.4f}** (skew ≥ 0.5). "
            "La cola derecha es más larga; la media supera a la mediana.",
        )
    else:
        sesgo_label, sesgo_color, sesgo_txt = (
            "⚠️ Sesgo negativo (cola izquierda)",
            "warning",
            f"El coeficiente de asimetría es **{skewness:.4f}** (skew ≤ −0.5). "
            "La cola izquierda es más larga; la mediana supera a la media.",
        )

    # ── Diagnóstico 2: Outliers ──────────────────────────────────────────────
    pct_out = len(outliers) / n * 100
    if len(outliers) == 0:
        out_label, out_color, out_txt = (
            "✅ Sin outliers detectados",
            "normal",
            "No se encontraron valores fuera del rango IQR × 1.5.",
        )
    elif pct_out <= 5:
        out_label, out_color, out_txt = (
            f"⚠️ {len(outliers)} outlier(s) moderados ({pct_out:.1f}%)",
            "warning",
            f"Se detectaron **{len(outliers)}** valor(es) atípico(s) "
            f"({pct_out:.1f}% de la muestra) usando el criterio IQR × 1.5.",
        )
    else:
        out_label, out_color, out_txt = (
            f"🔴 {len(outliers)} outliers severos ({pct_out:.1f}%)",
            "error",
            f"El **{pct_out:.1f}%** de los datos son atípicos. "
            "Revisar la calidad de los datos antes de proceder.",
        )

    # ── Diagnóstico 3: Normalidad ────────────────────────────────────────────
    alpha_norm = 0.05
    if p_norm > alpha_norm:
        norm_label, norm_color, norm_txt = (
            "✅ Distribución compatible con la Normal",
            "normal",
            f"**{test_nombre}**: estadístico = {stat_norm:.4f}, "
            f"p-value = {p_norm:.4f} > {alpha_norm}. "
            "No hay evidencia suficiente para rechazar la normalidad.",
        )
    else:
        norm_label, norm_color, norm_txt = (
            "⚠️ Posible desviación de la Normal",
            "warning",
            f"**{test_nombre}**: estadístico = {stat_norm:.4f}, "
            f"p-value = {p_norm:.4f} ≤ {alpha_norm}. "
            "Se rechaza la normalidad estricta; sin embargo, con n ≥ 30 "
            "el TCL respalda el uso de la Prueba Z.",
        )

    # Renderizar tarjetas de diagnóstico
    col_d1, col_d2, col_d3 = st.columns(3)

    for col, label, color, txt in [
        (col_d1, sesgo_label, sesgo_color, sesgo_txt),
        (col_d2, out_label,   out_color,   out_txt),
        (col_d3, norm_label,  norm_color,  norm_txt),
    ]:
        with col:
            if color == "normal":
                st.success(f"**{label}**\n\n{txt}")
            elif color == "warning":
                st.warning(f"**{label}**\n\n{txt}")
            else:
                st.error(f"**{label}**\n\n{txt}")

    # ── Tabla de estadísticos descriptivos ───────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Estadísticos descriptivos")

    tabla = pd.DataFrame({
        "Estadístico": [
            "n", "Media (x̄)", "Mediana", "Desv. Estándar (s)",
            "Asimetría (skewness)", "Curtosis (exceso)",
            "Q1", "Q3", "IQR", "Límite inf. IQR", "Límite sup. IQR",
            "Outliers detectados",
            f"Test de normalidad ({test_nombre})",
            "p-value (normalidad)",
        ],
        "Valor": [
            n, f"{media:.4f}", f"{mediana:.4f}", f"{s:.4f}",
            f"{skewness:.4f}", f"{kurtosis:.4f}",
            f"{q1:.4f}", f"{q3:.4f}", f"{iqr:.4f}",
            f"{lim_inf:.4f}", f"{lim_sup:.4f}",
            f"{len(outliers)} ({pct_out:.1f}%)",
            f"{stat_norm:.4f}",
            f"{p_norm:.4f}",
        ],
    })
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    # Guardar diagnóstico en session_state para la fase de IA
    st.session_state["diagnostico"] = {
        "n": n, "media": media, "mediana": mediana, "s": s,
        "skewness": skewness, "kurtosis": kurtosis,
        "outliers_n": len(outliers), "outliers_pct": pct_out,
        "test_normalidad": test_nombre,
        "stat_norm": stat_norm, "p_norm": p_norm,
        "sesgo_label": sesgo_label,
        "out_label": out_label,
        "norm_label": norm_label,
    }