import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Calculadora Scoring Cliente", layout="wide")
st.title("🧮 Calculadora de Scoring de Cliente (borrador)")
st.caption("Borrador: valores inventados (0–1). Score = Σ(Peso% · x). Puede calcular 1 cliente (inputs) o un archivo con muchos clientes.")

# ---------------- Pesos (los vuestros) ----------------
WEIGHTS = {
    "Antigüedad 1ª contratación": 7.5,
    "Vinculación: Nº de Ramos con nosotros": 7.5,
    "Rentabilidad de la póliza actual": 7.5,
    "Descuentos o Recargos aplicados sobre tarifa": 5.5,
    "Morosidad": 5.0,
    "Engagement comercial / Uso de canales propios": 4.5,
    "Frecuencia uso coberturas complementarias (sin siniestralidad)": 4.5,
    "Total asegurados / media asegurados por póliza": 4.5,
    "Edad": 4.5,
    "Rentabilidad histórica (LTV)": 4.5,
    "Tipo de distribución": 4.5,
    "Vinculación: Coberturas complementarias opcionales": 4.5,
    "Contactabilidad": 4.0,
    "Edad del asegurado más mayor": 4.0,
    "Vinculación familiar": 3.0,
    "Prescriptor": 3.0,
    "Exposición a comunicaciones de marca": 3.0,
    "Descendencia": 3.0,
    "Medio de pago": 2.5,
    "Frecuencia de pago (Periodicidad)": 2.0,
    "Probabilidad de desglose": 1.5,
    "Tipo de producto": 1.5,
    "NPS": 1.5,
    "Mascotas": 1.5,
    "Localización (potencial de compra)": 1.5,
    "Autónomo": 1.0,
    "Siniestralidad (Salud)": 1.0,
    "Grado de digitalización de la póliza": 0.5,
    "Profesión": 0.5,
    "Nivel educativo": 0.5,
    "Sexo": 0.0,  # no aporta
}

# ---------------- Subcategorías borrador + valores x ----------------
CONFIG = {
    "Antigüedad 1ª contratación": (["<1 año", "1–3", "3–5", "5–10", ">10"], [0.0, 0.25, 0.5, 0.75, 1.0]),
    "Vinculación: Nº de Ramos con nosotros": (["1 ramo", "2", "3", "4", "5"], [0.0, 0.25, 0.5, 0.75, 1.0]),
    "Rentabilidad de la póliza actual": (["Negativa", "Baja", "Media", "Alta", "Muy alta"], [0.0, 0.25, 0.5, 0.75, 1.0]),

    # cuanto más descuento peor
    "Descuentos o Recargos aplicados sobre tarifa": (
        [">20% desc", "10–20% desc", "0–10% desc", "Tarifa neutra", "Recargo / sin desc"],
        [0.0, 0.25, 0.5, 0.75, 1.0]
    ),

    # más moroso peor
    "Morosidad": (["Reincidente", "Varias incidencias", "Alguna incidencia", "Sin incidencias"], [0.0, 0.33, 0.66, 1.0]),

    "Engagement comercial / Uso de canales propios": (["Nulo", "Bajo", "Medio", "Alto"], [0.0, 0.33, 0.66, 1.0]),
    "Frecuencia uso coberturas complementarias (sin siniestralidad)": (["Nunca", "Baja", "Media", "Alta"], [0.0, 0.33, 0.66, 1.0]),
    "Total asegurados / media asegurados por póliza": (["1", "2", "3", "4", "5+"], [0.0, 0.25, 0.5, 0.75, 1.0]),

    # óptimo en el medio
    "Edad": (["<30", "30–50", ">50"], [0.6, 1.0, 0.5]),

    "Rentabilidad histórica (LTV)": (["Muy baja", "Baja", "Media", "Alta", "Muy alta"], [0.0, 0.25, 0.5, 0.75, 1.0]),

    # corredor malo, mediador bueno, propio buenísimo
    "Tipo de distribución": (["Corredor", "Mediador", "Propio"], [0.0, 0.7, 1.0]),

    "Vinculación: Coberturas complementarias opcionales": (["Ninguna", "1", "2", "3+"], [0.0, 0.33, 0.66, 1.0]),
    "Contactabilidad": (["Baja (1 canal)", "Media (2 canales)", "Alta (3+ canales)"], [0.2, 0.6, 1.0]),
    "Edad del asegurado más mayor": (["<50", "50–65", ">65"], [1.0, 0.6, 0.3]),

    "Vinculación familiar": (["No", "Sí"], [0.4, 1.0]),
    "Prescriptor": (["No", "Sí"], [0.5, 1.0]),
    "Exposición a comunicaciones de marca": (["Baja", "Media", "Alta"], [0.3, 0.6, 1.0]),
    "Descendencia": (["No", "Sí"], [0.6, 1.0]),

    "Medio de pago": (["Efectivo/otros", "Tarjeta", "Domiciliación"], [0.4, 0.7, 1.0]),
    "Frecuencia de pago (Periodicidad)": (["Mensual", "Trimestral", "Semestral", "Anual"], [0.4, 0.6, 0.8, 1.0]),

    "Probabilidad de desglose": (["Alta", "Media", "Baja"], [0.2, 0.6, 1.0]),
    "Tipo de producto": (["Básico", "Medio", "Premium"], [0.4, 0.7, 1.0]),
    "NPS": (["Detractor", "Pasivo", "Promotor"], [0.0, 0.6, 1.0]),
    "Mascotas": (["No", "Sí"], [0.6, 1.0]),
    "Localización (potencial de compra)": (["Bajo", "Medio", "Alto"], [0.4, 0.7, 1.0]),

    "Autónomo": (["No", "Sí"], [0.7, 1.0]),
    "Siniestralidad (Salud)": (["Alta", "Media", "Baja", "Sin siniestros"], [0.0, 0.4, 0.7, 1.0]),
    "Grado de digitalización de la póliza": (["Bajo", "Medio", "Alto"], [0.5, 0.75, 1.0]),
    "Profesión": (["Sin dato / otros", "Estable"], [0.6, 1.0]),
    "Nivel educativo": (["Sin dato", "Medio", "Alto"], [0.6, 0.8, 1.0]),
    "Sexo": (["No aplica"], [0.0]),
}

VAR_LIST = list(WEIGHTS.keys())

# ---------------- Tipos A/B/C (umbrales) ----------------
# Ajusta estos cortes como quieras
DEFAULT_A_MIN = 65.0
DEFAULT_B_MIN = 45.0

st.sidebar.header("Clasificación A/B/C")
a_min = st.sidebar.slider("Tipo A si Score ≥", 0.0, 100.0, DEFAULT_A_MIN, 1.0)
b_min = st.sidebar.slider("Tipo B si Score ≥", 0.0, 100.0, DEFAULT_B_MIN, 1.0)
st.sidebar.caption("Tipo C si Score < umbral de B")

def classify(score: float) -> str:
    if score >= a_min:
        return "A"
    if score >= b_min:
        return "B"
    return "C"

# ---------------- Helpers ----------------
def x_from_value(var: str, val) -> float:
    """Convierte el valor del archivo (texto o índice) en x. Si falta/incorrecto, usa 0."""
    labels, xs = CONFIG[var]

    if pd.isna(val):
        return 0.0

    # si llega como número -> índice
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        idx = int(val)
        if 0 <= idx < len(labels):
            return float(xs[idx])
        return 0.0

    # si llega como texto -> match label
    if isinstance(val, str):
        v = val.strip()
        if v in labels:
            return float(xs[labels.index(v)])
        # si es "3" como texto
        try:
            idx = int(v)
            if 0 <= idx < len(labels):
                return float(xs[idx])
        except:
            pass

    return 0.0

def score_row(row: pd.Series) -> float:
    total = 0.0
    for var, weight in WEIGHTS.items():
        x = x_from_value(var, row.get(var, None))
        total += weight * x
    return float(total)

# ---------------- Modo archivo (batch) ----------------
st.markdown("## 📤 Subir archivo para scoring masivo (varias filas)")
uploaded = st.file_uploader("Sube CSV o Excel con una fila por cliente (columnas = variables)", type=["csv", "xlsx"])

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_up = pd.read_csv(uploaded)
        else:
            df_up = pd.read_excel(uploaded)

        if df_up.empty:
            st.error("El archivo está vacío.")
        else:
            st.success(f"Archivo cargado: {df_up.shape[0]} clientes, {df_up.shape[1]} columnas.")

            # Calcula score por fila
            df_res = df_up.copy()
            df_res["Score_total_%"] = df_res.apply(score_row, axis=1)
            df_res["Tipo"] = df_res["Score_total_%"].apply(classify)

            # Porcentajes A/B/C
            dist = df_res["Tipo"].value_counts(normalize=True).reindex(["A", "B", "C"]).fillna(0) * 100

            cA, cB, cC = st.columns(3)
            cA.metric("% Tipo A", f"{dist['A']:.1f}%")
            cB.metric("% Tipo B", f"{dist['B']:.1f}%")
            cC.metric("% Tipo C", f"{dist['C']:.1f}%")

            st.markdown("### Resultados por cliente")
            # Si el archivo tiene una columna identificadora, puedes escogerla
            id_col = st.selectbox(
                "Columna identificadora (opcional, para mostrar primero)",
                options=["(ninguna)"] + list(df_up.columns),
                index=0
            )

            show_cols = []
            if id_col != "(ninguna)":
                show_cols.append(id_col)

            show_cols += ["Score_total_%", "Tipo"]
            # añade algunas variables al final (no todas) para no saturar
            sample_vars = [v for v in VAR_LIST if v in df_res.columns][:6]
            show_cols += sample_vars

            st.dataframe(df_res[show_cols].sort_values("Score_total_%", ascending=False), use_container_width=True)

            # Descarga CSV con resultados completos
            csv_bytes = df_res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar resultados (CSV)",
                data=csv_bytes,
                file_name="resultados_scoring_clientes.csv",
                mime="text/csv"
            )

            st.markdown("#### Nota")
            st.write(
                "Si alguna variable no está en el archivo o no coincide exactamente con una opción, "
                "esa variable cuenta como x=0 en ese cliente."
            )

    except Exception as e:
        st.error(f"No he podido leer/procesar el archivo: {e}")

st.divider()

# ---------------- Modo 1 cliente (manual) ----------------
st.markdown("## 🧍 Scoring manual de 1 cliente (inputs)")

def ensure_state():
    for v in VAR_LIST:
        key = f"sel_{v}"
        if key not in st.session_state:
            st.session_state[key] = 0

def choose_index_tipo(n: int, tipo: str) -> int:
    if n <= 1:
        return 0
    if tipo == "A":
        weights = [(i + 1) ** 3 for i in range(n)]
    elif tipo == "B":
        mid = (n - 1) / 2
        weights = [1 / (1 + abs(i - mid)) for i in range(n)]
    else:
        weights = [(n - i) ** 3 for i in range(n)]
    return random.choices(range(n), weights=weights, k=1)[0]

def fill_random_client(tipo: str):
    for v in VAR_LIST:
        labels, _ = CONFIG[v]
        st.session_state[f"sel_{v}"] = choose_index_tipo(len(labels), tipo)

ensure_state()

c2, c3, c4 = st.columns(3)
with c2:
    if st.button("🎲 Cliente aleatorio Tipo A (alto)", key="btnA"):
        fill_random_client("A")
        st.rerun()
with c3:
    if st.button("🎲 Cliente aleatorio Tipo B (medio)", key="btnB"):
        fill_random_client("B")
        st.rerun()
with c4:
    if st.button("🎲 Cliente aleatorio Tipo C (bajo)", key="btnC"):
        fill_random_client("C")
        st.rerun()

left, right = st.columns([1.3, 1])
rows = []
total = 0.0

with left:
    for var, weight in WEIGHTS.items():
        labels, xs = CONFIG[var]
        idx = st.selectbox(
            f"{var}  —  Peso {weight}%",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            key=f"sel_{var}",
        )
        x = float(xs[int(idx)])
        contrib = weight * x
        total += contrib
        rows.append({
            "Variable": var,
            "Selección": labels[int(idx)],
            "Peso (%)": weight,
            "x (0-1)": round(x, 3),
            "Contribución (%)": round(contrib, 3),
        })

with right:
    st.metric("Score total del cliente (%)", f"{total:.2f}")
    st.metric("Tipo", classify(total))
    st.caption("Tipo A/B/C se calcula con los umbrales configurados en la barra lateral.")

st.dataframe(pd.DataFrame(rows).sort_values("Contribución (%)", ascending=False), use_container_width=True)

st.markdown("## Fórmula")
st.latex(r"Score=\sum_i (Peso_i \cdot x_i)")
