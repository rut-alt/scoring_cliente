import streamlit as st
import pandas as pd

st.set_page_config(page_title="Scoring Cliente", layout="wide")
st.title("🧮 Calculadora de Scoring de Cliente (borrador)")

st.caption("Inputs por variable → calcula Score total = Σ(w · x). Subcategorías borrador (editables).")

# Pesos (los vuestros)
WEIGHTS = {
    "Antigüedad 1ª contratación": 7.5,
    "Vinculación: Nº de Ramos con nosotros": 7.5,
    "Rentabilidad de la póliza actual": 7.5,
    "Descuentos o Recargos aplicados sobre tarifa": 5.5,
    "Morosidad": 5.0,
    "Engagement comercial / uso canales propios": 4.5,
    "Frecuencia uso coberturas complementarias sin siniestralidad": 4.5,
    "Total asegurados (media por póliza)": 4.5,
    "Edad": 4.5,
    "Rentabilidad histórica (LTV)": 4.5,
    "Tipo de distribución": 4.5,
    # Puedes añadir más después
}

# Subcategorías borrador (puedes cambiarlas cuando queráis)
CONFIG = {
    "Antigüedad 1ª contratación": dict(
        k=5,
        labels=["< 1 año", "1–3 años", "3–5 años", "5–10 años", "> 10 años"],
        invert=False,   # más antigüedad mejor
        special=None
    ),
    "Vinculación: Nº de Ramos con nosotros": dict(
        k=5,
        labels=["1 ramo", "2 ramos", "3 ramos", "4 ramos", "5 ramos"],
        invert=False,   # más ramos mejor
        special=None
    ),
    "Rentabilidad de la póliza actual": dict(
        k=5,
        labels=["Muy baja / negativa", "Baja", "Media", "Alta", "Muy alta"],
        invert=False,
        special=None
    ),
    "Descuentos o Recargos aplicados sobre tarifa": dict(
        k=5,
        labels=[">20% descuento", "10–20% descuento", "0–10% descuento", "Tarifa neutra", "Recargo / sin descuento"],
        invert=True,    # más descuento = peor (como pediste)
        special=None
    ),
    "Morosidad": dict(
        k=4,
        labels=["Reincidente / incidencias recientes", "Alguna incidencia", "Histórico antiguo leve", "Sin incidencias"],
        invert=True,    # más moroso peor
        special=None
    ),
    "Engagement comercial / uso canales propios": dict(
        k=4,
        labels=["Nulo (no usa canales)", "Bajo (algún contacto)", "Medio (uso ocasional app/web)", "Alto (uso frecuente + responde)"],
        invert=False,
        special=None
    ),
    "Frecuencia uso coberturas complementarias sin siniestralidad": dict(
        k=4,
        labels=["Nunca", "Baja", "Media", "Alta"],
        invert=False,
        special=None
    ),
    "Total asegurados (media por póliza)": dict(
        k=5,
        labels=["1", "2", "3", "4", "5 o más"],
        invert=False,
        special=None
    ),
    "Edad": dict(
        k=3,
        labels=["<30", "30–50", ">50"],
        invert=False,
        # Caso especial: “óptimo en el medio”
        special="edad_optimo_medio"
    ),
    "Rentabilidad histórica (LTV)": dict(
        k=5,
        labels=["Muy baja", "Baja", "Media", "Alta", "Muy alta"],
        invert=False,
        special=None
    ),
    "Tipo de distribución": dict(
        k=3,
        labels=["Corredor", "Mediador", "Propio"],
        invert=False,
        special=None
    ),
}

def x_for_special(variable: str, peso_pct: float, choice_index: int) -> float:
    """
    Casos especiales no monotónicos.
    Aquí: Edad (mejor en el centro).
    """
    w = peso_pct / 100.0
    xmin = xmin_by_weight(w)

    if variable == "Edad":
        # 3 categorías: <30, 30–50, >50
        # Queremos 30–50 = 1 (mejor), extremos “peor”.
        # Manteniendo x >= xmin.
        xs = [
            max(xmin, 0.55),  # <30
            1.00,             # 30–50
            max(xmin, 0.45),  # >50
        ]
        return xs[choice_index]

    # fallback seguro
    return x_value_for_choice(peso_pct=peso_pct, k=3, choice_index=choice_index, invert=False)

st.markdown("## Inputs del cliente")

left, right = st.columns([1.2, 1])

rows = []
total_score_pct = 0.0

with left:
    for var, peso in WEIGHTS.items():
        conf = CONFIG[var]
        labels = conf["labels"]
        k = conf["k"]
        invert = conf["invert"]
        special = conf.get("special")

        choice = st.selectbox(f"{var}  —  Peso {peso}%", options=list(range(len(labels))), format_func=lambda i: labels[i])

        if special == "edad_optimo_medio":
            x = x_for_special("Edad", peso, choice)
        else:
            x = x_value_for_choice(peso_pct=peso, k=k, choice_index=choice, invert=invert)

        contrib_pct = peso * x  # porque peso ya está en % y x en [0,1]
        total_score_pct += contrib_pct

        rows.append({
            "Variable": var,
            "Selección": labels[choice],
            "Peso (%)": peso,
            "x": round(x, 4),
            "Contribución (%)": round(contrib_pct, 4),
        })

with right:
    st.markdown("## Resultado")
    st.metric("Score total del cliente (%)", f"{total_score_pct:.2f}")

    st.info(
        "Interpretación sugerida (borrador):\n"
        "- > 70%: Cliente muy prioritario\n"
        "- 50–70%: Cliente prioritario\n"
        "- < 50%: Cliente estándar\n\n"
        "⚠️ Los cortes son orientativos: se ajustan cuando validéis el modelo."
    )

st.markdown("## Desglose por variable")
df = pd.DataFrame(rows).sort_values(by="Contribución (%)", ascending=False)
st.dataframe(df, use_container_width=True)

st.markdown("## Fórmulas")
st.latex(r"Score=\sum_i (w_i \cdot x_i)")
st.latex(r"x(j)=x_{min} + \frac{(j-1)(1-x_{min})}{k-1}")
st.markdown("""
- **wᵢ**: peso de la variable (en %).
- **xᵢ**: valor normalizado (0–1), calculado según el peso vía **x_min**.
- La **Contribución (%)** que ves en la tabla es: **Peso (%) × x**.
""")
