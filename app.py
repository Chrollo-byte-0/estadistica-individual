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


# ── FASE 3: Prueba de Hipótesis · Prueba Z ───────────────────────────────────
if "data" in st.session_state:
    data    = st.session_state["data"]
    valores = data["valor"].values
    sigma   = st.session_state["sigma_pob"]
    n       = len(valores)
    x_bar   = np.mean(valores)

    st.markdown("---")
    st.header("🔬 Fase 3 · Prueba de Hipótesis — Prueba Z")

    # ── Supuesto recordatorio ────────────────────────────────────────────────
    with st.expander("📌 Supuestos de la Prueba Z", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.info(f"**σ poblacional conocida**\nσ = {sigma:.4f}")
        col_s2.info(f"**Tamaño de muestra**\nn = {n} {'✅ ≥ 30' if n >= 30 else '⚠️ < 30'}")
        col_s3.info("**TCL aplicable**\nDistribución de x̄ ~ Normal")

    st.markdown("---")

    # ── Configuración de hipótesis ───────────────────────────────────────────
    st.subheader("⚙️ Configuración de hipótesis")

    col_h1, col_h2, col_h3 = st.columns([1.2, 1.2, 1])

    with col_h1:
        mu_0 = st.number_input(
            "Valor hipotético de la media (μ₀)",
            value=round(float(x_bar), 2),
            step=0.1,
            help="Media poblacional bajo H₀.",
        )

    with col_h2:
        tipo_prueba = st.selectbox(
            "Tipo de prueba (H₁)",
            [
                "Bilateral (μ ≠ μ₀)",
                "Cola izquierda (μ < μ₀)",
                "Cola derecha (μ > μ₀)",
            ],
        )

    with col_h3:
        alpha = st.select_slider(
            "Nivel de significancia (α)",
            options=[0.01, 0.05, 0.10],
            value=0.05,
        )

    # ── Mostrar hipótesis en LaTeX ───────────────────────────────────────────
    tipo_map = {
        "Bilateral (μ ≠ μ₀)":       ("μ = μ₀", "μ ≠ μ₀", "bilateral"),
        "Cola izquierda (μ < μ₀)":  ("μ ≥ μ₀", "μ < μ₀", "left"),
        "Cola derecha (μ > μ₀)":    ("μ ≤ μ₀", "μ > μ₀", "right"),
    }
    h0_tex, h1_tex, cola = tipo_map[tipo_prueba]

    col_latex1, col_latex2 = st.columns(2)
    col_latex1.latex(rf"H_0: \quad {h0_tex.replace('μ', r'\mu').replace('μ₀', r'\mu_0')}")
    col_latex2.latex(rf"H_1: \quad {h1_tex.replace('μ', r'\mu').replace('μ₀', r'\mu_0')}")

    st.markdown("---")

    # ── Cálculo del estadístico Z ────────────────────────────────────────────
    error_est = sigma / np.sqrt(n)
    z_stat    = (x_bar - mu_0) / error_est

    if cola == "bilateral":
        p_value   = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        z_critico = stats.norm.ppf(1 - alpha / 2)
        rechaza   = abs(z_stat) > z_critico
    elif cola == "left":
        p_value   = stats.norm.cdf(z_stat)
        z_critico = stats.norm.ppf(alpha)
        rechaza   = z_stat < z_critico
    else:  # right
        p_value   = 1 - stats.norm.cdf(z_stat)
        z_critico = stats.norm.ppf(1 - alpha)
        rechaza   = z_stat > z_critico

    # ── Métricas principales ─────────────────────────────────────────────────
    st.subheader("📊 Resultados del contraste")

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Media muestral (x̄)",    f"{x_bar:.4f}")
    col_m2.metric("Error estándar (σ/√n)", f"{error_est:.4f}")
    col_m3.metric("Estadístico Z",          f"{z_stat:.4f}")
    col_m4.metric("p-value",                f"{p_value:.4f}")
    col_m5.metric("Z crítico",
                  f"±{z_critico:.4f}" if cola == "bilateral" else f"{z_critico:.4f}")

    # Veredicto
    if rechaza:
        st.error(
            f"### 🔴 Se RECHAZA H₀  (α = {alpha})\n"
            f"p-value = {p_value:.4f} < α = {alpha}  |  "
            f"Z = {z_stat:.4f} cae en la zona de rechazo."
        )
    else:
        st.success(
            f"### ✅ No se rechaza H₀  (α = {alpha})\n"
            f"p-value = {p_value:.4f} ≥ α = {alpha}  |  "
            f"Z = {z_stat:.4f} cae en la zona de no rechazo."
        )

    # ── Fórmula con valores sustituidos ─────────────────────────────────────
    with st.expander("🧮 Desarrollo del cálculo", expanded=False):
        st.latex(
            rf"Z = \frac{{\bar{{x}} - \mu_0}}{{\sigma / \sqrt{{n}}}} "
            rf"= \frac{{{x_bar:.4f} - {mu_0:.4f}}}{{{sigma:.4f} / \sqrt{{{n}}}}} "
            rf"= \frac{{{x_bar - mu_0:.4f}}}{{{error_est:.4f}}} "
            rf"= {z_stat:.4f}"
        )
        st.latex(
            rf"\text{{p-value}} = {p_value:.6f} \quad "
            rf"\alpha = {alpha} \quad "
            rf"\Rightarrow \quad "
            rf"{'\\text{{Rechazar}} H_0' if rechaza else '\\text{{No rechazar}} H_0'}"
        )

    # ── Gráfico: curva normal con zonas de rechazo ───────────────────────────
    st.subheader("📉 Curva normal estándar · Zonas de rechazo")

    x_plot = np.linspace(-4.2, 4.2, 800)
    y_plot = stats.norm.pdf(x_plot)

    ROJO   = "rgba(220, 53, 69, 0.55)"
    VERDE  = "rgba(40, 167, 69, 0.30)"
    BORDE  = "#DC3545"

    fig_z = go.Figure()

    # ── Zona(s) de NO rechazo (verde) ────────────────────────────────────────
    if cola == "bilateral":
        mask_nr = (x_plot >= -z_critico) & (x_plot <= z_critico)
    elif cola == "left":
        mask_nr = x_plot >= z_critico
    else:
        mask_nr = x_plot <= z_critico

    fig_z.add_trace(go.Scatter(
        x=np.concatenate([[x_plot[mask_nr][0]], x_plot[mask_nr], [x_plot[mask_nr][-1]]]),
        y=np.concatenate([[0], y_plot[mask_nr], [0]]),
        fill="toself", fillcolor=VERDE,
        line=dict(width=0), name="No rechazo H₀",
        hoverinfo="skip",
    ))

    # ── Zona(s) de rechazo (rojo) ────────────────────────────────────────────
    def add_rejection_area(fig, x_arr, y_arr, mask, name, showlegend=True):
        xs = x_arr[mask]
        ys = y_arr[mask]
        if len(xs) == 0:
            return
        fig.add_trace(go.Scatter(
            x=np.concatenate([[xs[0]], xs, [xs[-1]]]),
            y=np.concatenate([[0], ys, [0]]),
            fill="toself", fillcolor=ROJO,
            line=dict(color=BORDE, width=1.2),
            name=name, showlegend=showlegend,
            hoverinfo="skip",
        ))

    if cola == "bilateral":
        add_rejection_area(fig_z, x_plot, y_plot, x_plot <= -z_critico, "Rechazo H₀")
        add_rejection_area(fig_z, x_plot, y_plot, x_plot >=  z_critico, "Rechazo H₀", showlegend=False)
    elif cola == "left":
        add_rejection_area(fig_z, x_plot, y_plot, x_plot <= z_critico, "Rechazo H₀")
    else:
        add_rejection_area(fig_z, x_plot, y_plot, x_plot >= z_critico, "Rechazo H₀")

    # ── Curva normal completa ─────────────────────────────────────────────────
    fig_z.add_trace(go.Scatter(
        x=x_plot, y=y_plot,
        mode="lines", name="N(0,1)",
        line=dict(color="#4F8EF7", width=2.5),
    ))

    # ── Línea del estadístico Z observado ────────────────────────────────────
    z_plot_val = max(min(z_stat, 4.1), -4.1)   # clamp para visibilidad
    fig_z.add_vline(
        x=z_plot_val,
        line=dict(
            color="#FFD700", width=2.5, dash="dash",
        ),
        annotation_text=f"Z = {z_stat:.3f}",
        annotation_position="top" if z_stat >= 0 else "top left",
        annotation_font=dict(color="#FFD700", size=13),
    )

    # ── Línea(s) crítica(s) ───────────────────────────────────────────────────
    criticos = ([-z_critico, z_critico] if cola == "bilateral"
                else [z_critico])
    for zc in criticos:
        zc_clamp = max(min(zc, 4.1), -4.1)
        fig_z.add_vline(
            x=zc_clamp,
            line=dict(color=BORDE, width=1.8, dash="dot"),
            annotation_text=f"Zc = {zc:.3f}",
            annotation_position="top right" if zc > 0 else "top left",
            annotation_font=dict(color=BORDE, size=11),
        )

    fig_z.update_layout(
        height=430,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Z", range=[-4.3, 4.3], zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(title="f(Z)", showgrid=False),
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=30, b=60, l=40, r=20),
    )

    st.plotly_chart(fig_z, use_container_width=True)

    # ── Interpretación textual ────────────────────────────────────────────────
    st.subheader("📝 Interpretación estadística")

    signo_map = {"bilateral": "≠", "left": "<", "right": ">"}
    cola_txt  = {
        "bilateral": "bilateral",
        "left":      "de cola izquierda",
        "right":     "de cola derecha",
    }

    interpretacion = f"""
**Contraste {cola_txt[cola]} con α = {alpha}**

Se realizó una Prueba Z {cola_txt[cola]} para evaluar si la media poblacional
difiere de **μ₀ = {mu_0:.4f}**, asumiendo una desviación estándar poblacional
conocida de **σ = {sigma:.4f}** y una muestra de **n = {n}** observaciones.

- La media muestral observada fue **x̄ = {x_bar:.4f}**.
- El estadístico de prueba resultó **Z = {z_stat:.4f}**.
- El valor crítico para α = {alpha} es **Zc = {'±' if cola == 'bilateral' else ''}{abs(z_critico):.4f}**.
- El p-value obtenido es **{p_value:.4f}**.

{'🔴 **Conclusión:** Con un nivel de significancia del ' + str(int(alpha*100)) + '%, existe evidencia estadística suficiente para **rechazar H₀**. Los datos sugieren que la media poblacional ' + signo_map[cola] + ' ' + str(mu_0) + '.' if rechaza else '✅ **Conclusión:** Con un nivel de significancia del ' + str(int(alpha*100)) + '%, **no existe evidencia suficiente para rechazar H₀**. Los datos son consistentes con una media poblacional de ' + str(mu_0) + '.'}
"""
    st.markdown(interpretacion)

    # ── Guardar en session_state para Fase 4 ────────────────────────────────
    st.session_state["prueba_z"] = {
        "mu_0":       mu_0,
        "alpha":      alpha,
        "cola":       cola,
        "x_bar":      x_bar,
        "sigma":      sigma,
        "n":          n,
        "error_est":  error_est,
        "z_stat":     z_stat,
        "z_critico":  z_critico,
        "p_value":    p_value,
        "rechaza":    rechaza,
        "h0_tex":     h0_tex,
        "h1_tex":     h1_tex,
    }


# ── FASE 4: Módulo de IA · Google Gemini ─────────────────────────────────────
import google.generativeai as genai

if "prueba_z" in st.session_state and "diagnostico" in st.session_state:

    pz   = st.session_state["prueba_z"]
    diag = st.session_state["diagnostico"]

    st.markdown("---")
    st.header("🤖 Fase 4 · Interpretación con IA — Google Gemini")

    # ── Configuración de API Key ─────────────────────────────────────────────
    with st.expander("🔑 Configuración de API Key", expanded=False):
        st.markdown(
            "Obtén tu clave en [Google AI Studio](https://aistudio.google.com/app/apikey). "
            "La clave **no se almacena** fuera de esta sesión."
        )
        api_key_input = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="La clave se usa únicamente para esta consulta.",
        )
        if api_key_input:
            st.session_state["gemini_api_key"] = api_key_input
            st.success("✅ API Key registrada para esta sesión.")

    api_key = st.session_state.get("gemini_api_key", "")

    # ── Selector de modelo ───────────────────────────────────────────────────
    col_mod1, col_mod2 = st.columns([1, 2])
    with col_mod1:
        modelo_gemini = st.selectbox(
            "Modelo Gemini",
            ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            index=0,
            help="Flash: rápido y económico. Pro: mayor razonamiento.",
        )

    # ── Construcción del prompt ──────────────────────────────────────────────
    cola_labels = {
        "bilateral": "bilateral (H₁: μ ≠ μ₀)",
        "left":      "cola izquierda (H₁: μ < μ₀)",
        "right":     "cola derecha (H₁: μ > μ₀)",
    }

    prompt_estadistico = f"""
Eres un experto en estadística inferencial. Analiza los siguientes resultados
de un análisis estadístico completo y proporciona una interpretación profesional,
clara y detallada en español.

════════════════════════════════════════
 SECCIÓN 1 — ANÁLISIS EXPLORATORIO
════════════════════════════════════════
- Tamaño de muestra (n):           {diag['n']}
- Media muestral (x̄):             {diag['media']:.4f}
- Mediana:                         {diag['mediana']:.4f}
- Desviación estándar muestral (s):{diag['s']:.4f}
- Coeficiente de asimetría:        {diag['skewness']:.4f}
- Curtosis (exceso Fisher):        {diag['kurtosis']:.4f}
- Outliers detectados:             {diag['outliers_n']} ({diag['outliers_pct']:.1f}%)
- Test de normalidad ({diag['test_normalidad']}):
    - Estadístico:                 {diag['stat_norm']:.4f}
    - p-value:                     {diag['p_norm']:.4f}
- Diagnóstico de sesgo:            {diag['sesgo_label']}
- Diagnóstico de outliers:         {diag['out_label']}
- Diagnóstico de normalidad:       {diag['norm_label']}

════════════════════════════════════════
 SECCIÓN 2 — PRUEBA DE HIPÓTESIS (Z)
════════════════════════════════════════
- H₀:                              {pz['h0_tex']}
- H₁:                              {pz['h1_tex']}
- Tipo de prueba:                   {cola_labels[pz['cola']]}
- Valor hipotético (μ₀):           {pz['mu_0']:.4f}
- σ poblacional conocida:          {pz['sigma']:.4f}
- Media muestral (x̄):             {pz['x_bar']:.4f}
- Error estándar (σ/√n):           {pz['error_est']:.4f}
- Estadístico Z calculado:         {pz['z_stat']:.4f}
- Valor crítico (Zc):              {pz['z_critico']:.4f}
- p-value:                         {pz['p_value']:.6f}
- Nivel de significancia (α):      {pz['alpha']}
- Conclusión:                      {'Se RECHAZA H₀' if pz['rechaza'] else 'No se rechaza H₀'}

════════════════════════════════════════
 INSTRUCCIONES DE RESPUESTA
════════════════════════════════════════
Por favor, estructura tu respuesta en las siguientes secciones con sus títulos
en negrita y usando viñetas donde corresponda:

1. **Resumen ejecutivo** (2-3 oraciones con el hallazgo principal)

2. **Interpretación del Análisis Exploratorio**
   - Comenta la forma de la distribución (sesgo, curtosis)
   - Evalúa si la presencia/ausencia de outliers es preocupante
   - Opina sobre la normalidad y su impacto en los supuestos

3. **Interpretación de la Prueba Z**
   - Explica qué significa el valor Z = {pz['z_stat']:.4f} en contexto
   - Interpreta el p-value = {pz['p_value']:.6f} en lenguaje no técnico
   - Indica claramente si se rechaza o no H₀ y qué implica

4. **Validez de los supuestos**
   - ¿Es razonable asumir σ poblacional conocida?
   - ¿El tamaño muestral justifica el uso de la Prueba Z?
   - ¿Algún hallazgo del EDA compromete la validez del contraste?

5. **Conclusión y recomendaciones**
   - Conclusión final en 2-3 oraciones
   - 2 o 3 recomendaciones concretas para el analista

Usa un tono académico pero accesible. No incluyas fórmulas matemáticas en la
respuesta, solo interpretación en lenguaje natural.
""".strip()

    # ── Vista previa del prompt ──────────────────────────────────────────────
    with st.expander("🔍 Ver prompt enviado a Gemini", expanded=False):
        st.code(prompt_estadistico, language="markdown")

    # ── Botón de consulta ────────────────────────────────────────────────────
    st.markdown("###")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])

    with col_btn2:
        consultar = st.button(
            "🚀 Consultar a Gemini",
            use_container_width=True,
            disabled=not bool(api_key),
            help="Configura tu API Key primero." if not api_key else "",
        )

    if not api_key:
        st.warning("⚠️ Ingresa tu API Key de Gemini para habilitar este módulo.")

    # ── Llamada a la API y respuesta ─────────────────────────────────────────
    if consultar and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=modelo_gemini,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,          # respuestas consistentes
                    max_output_tokens=1800,
                ),
                system_instruction=(
                    "Eres un estadístico experto y profesor universitario. "
                    "Tus respuestas son rigurosas, estructuradas y didácticas. "
                    "Siempre respondes en español y en formato Markdown."
                ),
            )

            with st.spinner("⏳ Consultando a Gemini, espera un momento..."):
                response = model.generate_content(prompt_estadistico)
                respuesta_texto = response.text

            st.session_state["gemini_respuesta"] = respuesta_texto
            st.session_state["gemini_modelo"]    = modelo_gemini

        except genai.types.BlockedPromptException:
            st.error("❌ El prompt fue bloqueado por los filtros de seguridad de Gemini.")
        except Exception as e:
            st.error(f"❌ Error al consultar Gemini: {e}")
            st.info(
                "Verifica que tu API Key sea válida y que el modelo seleccionado "
                "esté disponible en tu cuenta de Google AI Studio."
            )

    # ── Renderizado de la respuesta ──────────────────────────────────────────
    if "gemini_respuesta" in st.session_state:
        st.markdown("---")
        st.subheader(f"💬 Interpretación de Gemini · `{st.session_state.get('gemini_modelo', '')}`")

        # Tarjeta contenedora con fondo sutil
        with st.container(border=True):
            st.markdown(st.session_state["gemini_respuesta"])

        # ── Acciones post-respuesta ──────────────────────────────────────────
        st.markdown("---")
        col_acc1, col_acc2, col_acc3 = st.columns(3)

        with col_acc1:
            st.download_button(
                label="⬇️ Descargar interpretación (.txt)",
                data=st.session_state["gemini_respuesta"],
                file_name="interpretacion_gemini.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_acc2:
            reporte_completo = f"""
REPORTE DE ANÁLISIS ESTADÍSTICO — PRUEBA Z
==========================================
Generado con soporte de IA (Google Gemini · {st.session_state.get('gemini_modelo','')})

── DATOS DEL CONTRASTE ──────────────────
H₀: {pz['h0_tex']}
H₁: {pz['h1_tex']}
μ₀ = {pz['mu_0']:.4f} | α = {pz['alpha']} | n = {pz['n']}
x̄  = {pz['x_bar']:.4f} | σ = {pz['sigma']:.4f}
Z  = {pz['z_stat']:.4f} | Zc = {pz['z_critico']:.4f}
p-value = {pz['p_value']:.6f}
Conclusión: {'Se RECHAZA H₀' if pz['rechaza'] else 'No se rechaza H₀'}

── DIAGNÓSTICO EDA ───────────────────────
Sesgo:       {diag['sesgo_label']}
Outliers:    {diag['out_label']}
Normalidad:  {diag['norm_label']}

── INTERPRETACIÓN IA ─────────────────────
{st.session_state['gemini_respuesta']}
""".strip()

            st.download_button(
                label="📄 Descargar reporte completo (.txt)",
                data=reporte_completo,
                file_name="reporte_estadistico_completo.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_acc3:
            if st.button("🔄 Nueva consulta", use_container_width=True):
                del st.session_state["gemini_respuesta"]
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "Análisis Estadístico · Prueba Z &nbsp;|&nbsp; "
    "Powered by Streamlit, SciPy & Google Gemini"
    "</div>",
    unsafe_allow_html=True,
)