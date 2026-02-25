import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Calculadora Scoring Cliente", layout="wide")
st.title("🧮 Calculadora de Scoring de Cliente (borrador)")
st.caption("Borrador: valores inventados (0–1) para probar el funcionamiento. Score = Σ(Peso% · x).")

# Pesos (los vuestros)
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

# Subcategorías borrador + valores x (inventados)
CONFIG = {
    "Antigüedad 1ª contratación": (["<1 año", "1–3", "3–5", "5–10", ">10"], [0.0, 0.25, 0.5, 0.75, 1.0]),
    "Vinculación: Nº de Ramos con nosotros": (["1 ramo", "2", "3", "4", "5"], [0.0, 0.25, 0.5, 0.75, 1.0]),
    "Rentabilidad de la póliza actual": (["Negativa", "Baja", "Media", "Alta", "Muy alta"], [0.0, 0.25, 0.5, 0.75, 1.0]),

    # cuanto más descuento peor
    "Descuentos o Recargos aplicados sobre tarifa": (
        [">20% desc", "10–20% desc", "0–10% desc", "Tarifa neutra", "Recargo / sin desc"],
        [0.0, 0.25, 0.5, 0.75, 1.0]
    ),

    # más moroso peor (4)
    "Morosidad": (["Reincidente", "Varias incidencias", "Alguna incidencia", "Sin incidencias"], [0.0, 0.33, 0.66, 1.0]),

    "Engagement comercial / Uso de canales propios": (["Nulo", "Bajo", "Medio", "Alto"], [0.0, 0.33, 0.66, 1.0]),
    "Frecuencia uso coberturas complementarias (sin siniestralidad)": (["Nunca", "Baja", "Media", "Alta"], [0.0, 0.33, 0.66, 1.0]),
    "Total asegurados / media asegurados por póliza": (["1", "2", "3", "4", "5+"], [0.0, 0.25, 0.5, 0.75, 1.0]),

    # Edad: óptimo en el medio
    "Edad": (["<30", "30–50", ">50"], [0.6, 1.0, 0.5]),

    "Rentabilidad histórica (LTV)": (["Muy baja", "Baja", "Media", "Alta", "Muy alta"], [0.0, 0.25, 0.5, 0.75, 1.0]),

    # distribución: corredor malo, mediador bueno, propio buenísimo
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


# ---------- Helpers estado + aleatorio ----------
def ensure_state():
    # inicializa las keys de los selectbox (sel_<var>)
    for v in VAR_LIST:
        key = f"sel_{v}"
        if key not in st.session_state:
            st.session_state[key] = 0


def choose_index_tipo(n: int, tipo: str) -> int:
    """Devuelve un índice 0..n-1 con sesgo según tipo A/B/C."""
    if n <= 1:
        return 0

    if tipo == "A":       # alto: sesgo fuerte a índices altos
        weights = [(i + 1) ** 3 for i in range(n)]
    elif tipo == "B":     # medio: pico en el centro
        mid = (n - 1) / 2
        weights = [1 / (1 + abs(i - mid)) for i in range(n)]
    else:                 # C bajo: sesgo fuerte a índices bajos
        weights = [(n - i) ** 3 for i in range(n)]

    return random.choices(range(n), weights=weights, k=1)[0]


def fill_random_client(tipo: str):
    """Rellena los selectbox en session_state y luego se rerunea."""
    for v in VAR_LIST:
        labels, _ = CONFIG[v]
        st.session_state[f"sel_{v}"] = choose_index_tipo(len(labels), tipo)


def load_client_from_df(df: pd.DataFrame):
    """Carga primera fila: valores pueden ser índice o texto exacto de opción."""
    row = df.iloc[0].to_dict()

    for v in VAR_LIST:
        if v not in row:
            continue

        val = row[v]
        labels, _ = CONFIG[v]

        # número = índice
        if isinstance(val, (int, float)) and pd.notna(val):
            st.session_state[f"sel_{v}"] = max(0, min(int(val), len(labels) - 1))
            continue

        # texto = label
        if isinstance(val, str):
            val_clean = val.strip()
            if val_clean in labels:
                st.session_state[f"sel_{v}"] = labels.index(val_clean)
                continue
            # si viene como "3" texto
            try:
                idx = int(val_clean)
                st.session_state[f"sel_{v}"] = max(0, min(idx, len(labels) - 1))
            except:
                pass


ensure_state()

# ---------- Barra superior ----------
st.markdown("## Acciones rápidas")
c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])

with c1:
    uploaded = st.file_uploader("📤 Subir cliente (CSV o Excel)", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            if df_up.empty:
                st.error("El archivo está vacío.")
            else:
                load_client_from_df(df_up)
                st.success("Cliente cargado desde archivo (primera fila).")
                st.rerun()
        except Exception as e:
            st.error(f"No he podido leer el archivo: {e}")

with c2:
    if st.button("🎲 Cliente aleatorio Tipo A (alto)"):
        fill_random_client("A")
        st.rerun()

with c3:
    if st.button("🎲 Cliente aleatorio Tipo B (medio)"):
        fill_random_client("B")
        st.rerun()

with c4:
    if st.button("🎲 Cliente aleatorio Tipo C (bajo)"):
        fill_random_client("C")
        st.rerun()

# ---------- Inputs + cálculo ----------
st.markdown("## Inputs del cliente")
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
            key=f"sel_{var}",  # clave controlada por session_state
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
    st.markdown("## Resultado")
    st.metric("Score total del cliente (%)", f"{total:.2f}")
    st.info(
        "Notas:\n"
        "- Los valores x (0–1) son **inventados** para este borrador.\n"
        "- Los botones A/B/C rellenan los inputs con sesgo (alto/medio/bajo).\n"
        "- El archivo cargado debe tener columnas con nombres iguales a las variables."
    )

st.markdown("## Desglose por variable")
df = pd.DataFrame(rows).sort_values("Contribución (%)", ascending=False)
st.dataframe(df, use_container_width=True)

st.markdown("## Fórmula")
st.latex(r"Score=\sum_i (Peso_i \cdot x_i)")
st.markdown("- **Pesoᵢ** en porcentaje (ej. 7,5)\n- **xᵢ** entre 0 y 1 según la categoría elegida")
