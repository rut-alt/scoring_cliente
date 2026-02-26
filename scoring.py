# calculadora.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openpyxl import Workbook


# =========================================================
# 1) LÓGICA DEL TALLER (generate_scale + xmin_floor)
# =========================================================

@dataclass(frozen=True)
class CategoryResult:
    j: int
    x: float                 # valor normalizado en [xmin, 1]
    contribution: float      # w * x (proporción)
    contribution_pct: float  # w * x en %
    delta_from_prev: float   # incremento vs categoría anterior (proporción)
    delta_from_prev_pct: float


def xmin_by_weight(w: float) -> float:
    peso_pct = round(w * 100, 1)
    if peso_pct <= 0:
        return 0.0

    mapping = {
        7.5: 0.00,
        5.5: 0.05,
        5.0: 0.10,
        4.5: 0.15,
        4.0: 0.20,
        3.0: 0.25,
        2.5: 0.30,
        2.0: 0.35,
        1.5: 0.40,
        1.0: 0.45,
        0.5: 0.50,
        0.0: 0.0,
    }
    return mapping.get(peso_pct, 0.20)


def generate_scale(
    peso_pct: float,
    k: int,
    xmin: Optional[float] = None,
    xmin_floor: float = 0.01,
) -> Dict:
    if k < 2:
        raise ValueError("k debe ser >= 2 (mínimo 2 categorías).")

    w = peso_pct / 100.0

    if xmin is None:
        xmin = xmin_by_weight(w)

    xmin = max(float(xmin_floor), float(xmin))
    xmin = max(0.0, min(1.0, float(xmin)))

    results: List[CategoryResult] = []
    prev_contrib = 0.0

    for j in range(1, k + 1):
        if w == 0:
            x = 0.0
        else:
            x = xmin + (j - 1) * (1.0 - xmin) / (k - 1)

        contrib = w * x
        delta = contrib - prev_contrib if j > 1 else 0.0

        results.append(
            CategoryResult(
                j=j,
                x=round(x, 6),
                contribution=round(contrib, 6),
                contribution_pct=round(contrib * 100.0, 4),
                delta_from_prev=round(delta, 6),
                delta_from_prev_pct=round(delta * 100.0, 4),
            )
        )
        prev_contrib = contrib

    x_min_effective = results[0].x if results else 0.0
    x_max_effective = results[-1].x if results else 0.0
    delta_max = w * (x_max_effective - x_min_effective)

    return {
        "peso_pct": float(peso_pct),
        "w": float(w),
        "k": int(k),
        "xmin": float(xmin),
        "x_min_effective": float(x_min_effective),
        "x_max_effective": float(x_max_effective),
        "delta_max": round(float(delta_max), 6),
        "delta_max_pct": round(float(delta_max) * 100.0, 4),
        "categories": results,
    }


def build_model_from_taller_json(
    model_dict: dict,
    xmin_floor: float,
    invert_map: Dict[str, bool],
) -> Tuple[Dict[str, float], Dict[str, Tuple[List[str], List[float]]]]:
    weights: Dict[str, float] = {}
    config: Dict[str, Tuple[List[str], List[float]]] = {}

    for v in model_dict.get("variables", []):
        name = str(v.get("name", "")).strip()
        if not name:
            continue

        peso = float(v.get("peso_pct", 0.0))
        k = int(v.get("k", 3))
        labels = list(v.get("labels") or [])

        if len(labels) < k:
            labels = labels + [""] * (k - len(labels))
        if len(labels) > k:
            labels = labels[:k]

        labels = [
            lab.strip() if isinstance(lab, str) and lab.strip() else f"K={i+1}"
            for i, lab in enumerate(labels)
        ]

        scale = generate_scale(peso_pct=peso, k=k, xmin=None, xmin_floor=xmin_floor)
        xs = [c.x for c in scale["categories"]]  # peor -> mejor

        # Si "más = peor", invertimos orden visual/índices
        if invert_map.get(name, False):
            xs = list(reversed(xs))
            labels = list(reversed(labels))

        weights[name] = peso
        config[name] = (labels, xs)

    return weights, config


# =========================================================
# 2) APP
# =========================================================

st.set_page_config(page_title="Calculadora Scoring Cliente", layout="wide")
st.title("Calculadora de Scoring de Cliente (con modelo del taller)")
st.caption("Carga el JSON exportado del taller. Score = Σ(Peso% · x). Modo 1 cliente o archivo masivo.")

# --- LOGO ---
st.sidebar.image("LOGOTIPO-AES-05.png", use_container_width=True)
st.sidebar.markdown("---")

st.sidebar.header("Modelo (JSON del taller)")

uploaded_model = st.sidebar.file_uploader("Sube el JSON exportado del taller", type=["json"])

xmin_floor = st.sidebar.slider(
    "Suelo mínimo xmin",
    min_value=0.0,
    max_value=0.30,
    value=0.01,
    step=0.01,
    help="Debe ser el mismo que usaste (o quieras usar) en el taller.",
)

st.sidebar.divider()
st.sidebar.subheader("Variables invertidas (más = peor)")
st.sidebar.caption("Marca las variables donde el orden natural es “más es peor”.")

DEFAULT_INVERT = {
    "Descuentos o Recargos aplicados sobre tarifa": True,
    "Morosidad: Históricos sin incidencia en devolución (Anotaciones de póliza)": True,
}

invert_map: Dict[str, bool] = {}

if uploaded_model is None:
    st.info("Sube el JSON del taller en la barra lateral para cargar pesos/k/etiquetas.")
    st.stop()

try:
    model = json.load(uploaded_model)
except Exception as e:
    st.error(f"No he podido leer el JSON: {e}")
    st.stop()

var_names = [v.get("name") for v in model.get("variables", []) if isinstance(v, dict) and v.get("name")]
for name in var_names:
    invert_map[name] = st.sidebar.checkbox(
        name,
        value=DEFAULT_INVERT.get(name, False),
        key=f"inv_{name}",
    )

WEIGHTS, CONFIG = build_model_from_taller_json(model, xmin_floor=xmin_floor, invert_map=invert_map)
VAR_LIST = list(WEIGHTS.keys())

if not VAR_LIST:
    st.error("El JSON no contiene variables válidas.")
    st.stop()

# ----- Tipos A/B/C -----
st.sidebar.divider()
st.sidebar.header("Clasificación A/B/C")
DEFAULT_A_MIN = 65.0
DEFAULT_B_MIN = 45.0
a_min = st.sidebar.slider("Tipo A si Score ≥", 0.0, 100.0, DEFAULT_A_MIN, 1.0)
b_min = st.sidebar.slider("Tipo B si Score ≥", 0.0, 100.0, DEFAULT_B_MIN, 1.0)
st.sidebar.caption("Tipo C si Score < umbral de B")


def classify(score: float) -> str:
    if score >= a_min:
        return "A"
    if score >= b_min:
        return "B"
    return "C"


# =========================================================
# Helpers scoring
# =========================================================

def x_from_value(var: str, val) -> float:
    labels, xs = CONFIG[var]

    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0

    if isinstance(val, (int, float)) and not isinstance(val, bool):
        idx = int(val)
        if 0 <= idx < len(labels):
            return float(xs[idx])
        return 0.0

    if isinstance(val, str):
        v = val.strip()
        if v in labels:
            return float(xs[labels.index(v)])
        try:
            idx = int(v)
            if 0 <= idx < len(labels):
                return float(xs[idx])
        except Exception:
            pass

    return 0.0


def score_row(row: pd.Series) -> float:
    total = 0.0
    for var, weight in WEIGHTS.items():
        x = x_from_value(var, row.get(var, None))
        total += weight * x
    return float(total)


# --- nuevos helpers para generación por rango ---

def ensure_state():
    for v in VAR_LIST:
        key = f"sel_{v}"
        if key not in st.session_state:
            st.session_state[key] = 0


def score_from_session() -> float:
    total = 0.0
    for var, weight in WEIGHTS.items():
        _, xs = CONFIG[var]
        idx = int(st.session_state.get(f"sel_{var}", 0))
        idx = max(0, min(idx, len(xs) - 1))
        total += weight * float(xs[idx])
    return float(total)


def min_max_score_possible() -> Tuple[float, float]:
    """Score mínimo y máximo teórico con el CONFIG actual (teniendo en cuenta invertidas)."""
    min_s, max_s = 0.0, 0.0
    for var, w in WEIGHTS.items():
        _, xs = CONFIG[var]
        if not xs:
            continue
        min_s += w * float(min(xs))
        max_s += w * float(max(xs))
    return float(min_s), float(max_s)


def set_all_indices(mode: str):
    """
    mode:
      'max' -> mejor categoría (índice n-1)
      'min' -> peor categoría (índice 0)
      'mid' -> categoría intermedia (n//2)
    """
    for var in VAR_LIST:
        _, xs = CONFIG[var]
        n = len(xs)
        if n <= 1:
            st.session_state[f"sel_{var}"] = 0
        else:
            if mode == "max":
                st.session_state[f"sel_{var}"] = n - 1
            elif mode == "min":
                st.session_state[f"sel_{var}"] = 0
            else:
                st.session_state[f"sel_{var}"] = n // 2


def current_state_dict() -> Dict[str, int]:
    return {f"sel_{v}": int(st.session_state.get(f"sel_{v}", 0)) for v in VAR_LIST}


def score_from_state(state: Dict[str, int]) -> float:
    total = 0.0
    for var, w in WEIGHTS.items():
        _, xs = CONFIG[var]
        idx = int(state.get(f"sel_{var}", 0))
        idx = max(0, min(idx, len(xs) - 1))
        total += w * float(xs[idx])
    return float(total)


def fill_random_client(tipo: str, tries: int = 400) -> Tuple[bool, float, str]:
    """
    Genera un cliente que cumpla los umbrales del sidebar:
      A: score >= a_min
      B: b_min <= score < a_min
      C: score < b_min

    Devuelve: (ok, score, msg)
    """
    # Umbrales incoherentes
    if b_min > a_min:
        set_all_indices("mid")
        s = score_from_session()
        return True, s, "Umbrales incoherentes (B > A). He generado un cliente intermedio."

    min_s, max_s = min_max_score_possible()

    # Rangos objetivo
    if tipo == "A":
        lo, hi = a_min, max_s
    elif tipo == "B":
        lo, hi = b_min, a_min - 1e-9
    else:
        lo, hi = min_s, b_min - 1e-9

    # Si el rango pedido es imposible
    if lo > max_s + 1e-9:
        set_all_indices("max")
        s = score_from_session()
        return False, s, f"Imposible llegar a {lo:.2f}%. Máximo teórico: {max_s:.2f}%."
    if hi < min_s - 1e-9:
        set_all_indices("min")
        s = score_from_session()
        return False, s, f"Imposible bajar a {hi:.2f}%. Mínimo teórico: {min_s:.2f}%."

    # ---- Tipo A: construir desde arriba ----
    if tipo == "A":
        set_all_indices("max")
        s = score_from_session()
        if s < a_min - 1e-9:
            return False, s, f"Máximo {s:.2f}% no llega al umbral A {a_min:.2f}%."

        # Variabilidad: baja algunas variables sin caer por debajo de a_min
        vars_shuffled = VAR_LIST[:]
        random.shuffle(vars_shuffled)
        for var in vars_shuffled:
            key = f"sel_{var}"
            _, xs = CONFIG[var]
            if len(xs) <= 1:
                continue
            idx = int(st.session_state[key])
            if idx <= 0:
                continue

            st.session_state[key] = idx - 1
            s2 = score_from_session()
            if s2 >= a_min:
                s = s2
            else:
                st.session_state[key] = idx  # revert

        return True, s, f"Cliente A generado (≥ {a_min:.2f}%)."

    # ---- Tipo C: construir desde abajo ----
    if tipo == "C":
        set_all_indices("min")
        s = score_from_session()
        if s >= b_min:
            return False, s, f"Mínimo {s:.2f}% no baja del umbral B {b_min:.2f}%."
        return True, s, f"Cliente C generado (< {b_min:.2f}%)."

    # ---- Tipo B: bajar desde arriba hasta caer en rango ----
    target_mid = (a_min + b_min) / 2.0
    best_state = None
    best_dist = None
    best_score = None

    for _ in range(tries):
        # arrancamos en max
        set_all_indices("max")
        state = current_state_dict()
        s = score_from_state(state)

        vars_order = VAR_LIST[:]
        random.shuffle(vars_order)

        # bajar hasta quedar < a_min
        for var in vars_order:
            if s < a_min:
                break
            key = f"sel_{var}"
            _, xs = CONFIG[var]
            idx = state[key]
            if idx <= 0:
                continue
            state[key] = idx - 1
            s = score_from_state(state)

        # si nos pasamos por debajo de b_min, reintenta
        if b_min <= s < a_min:
            for k, v in state.items():
                st.session_state[k] = v
            return True, s, f"Cliente B generado (entre {b_min:.2f}% y {a_min:.2f}%)."

        dist = abs(s - target_mid)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_state = state
            best_score = s

    if best_state is not None:
        for k, v in best_state.items():
            st.session_state[k] = v
        return False, float(best_score), "No encontré un B perfecto, te dejo el más cercano."

    set_all_indices("mid")
    s = score_from_session()
    return False, s, "No pude generar cliente (fallback a intermedio)."


# =========================================================
# Modo archivo (batch)
# =========================================================

st.markdown("## Subir archivo para scoring masivo (varias filas)")
uploaded = st.file_uploader("Sube CSV o Excel con una fila por cliente (columnas = variables)", type=["csv", "xlsx"])


def build_template_xlsx() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Plantilla"
    ws.append(list(WEIGHTS.keys()))
    ws.append(["(elige una opción exacta o índice 0..k-1)"] + [""] * (len(WEIGHTS) - 1))

    bio = BytesIO()
    wb.save(bio)
    return bio.getvalue()


st.download_button(
    "⬇️ Descargar plantilla Excel (vacía)",
    data=build_template_xlsx(),
    file_name="plantilla_clientes_scoring.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

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

            df_res = df_up.copy()
            df_res["Score_total_%"] = df_res.apply(score_row, axis=1)
            df_res["Tipo"] = df_res["Score_total_%"].apply(classify)

            dist = df_res["Tipo"].value_counts(normalize=True).reindex(["A", "B", "C"]).fillna(0) * 100
            cA, cB, cC = st.columns(3)
            cA.metric("% Tipo A", f"{dist['A']:.1f}%")
            cB.metric("% Tipo B", f"{dist['B']:.1f}%")
            cC.metric("% Tipo C", f"{dist['C']:.1f}%")

            st.markdown("### Resultados por cliente")
            id_col = st.selectbox(
                "Columna identificadora (opcional, para mostrar primero)",
                options=["(ninguna)"] + list(df_up.columns),
                index=0
            )

            show_cols = []
            if id_col != "(ninguna)":
                show_cols.append(id_col)

            show_cols += ["Score_total_%", "Tipo"]
            sample_vars = [v for v in VAR_LIST if v in df_res.columns][:6]
            show_cols += sample_vars

            st.dataframe(df_res[show_cols].sort_values("Score_total_%", ascending=False), use_container_width=True)

            csv_bytes = df_res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar resultados (CSV)",
                data=csv_bytes,
                file_name="resultados_scoring_clientes.csv",
                mime="text/csv"
            )

            st.markdown("#### Nota")
            st.write(
                "Si alguna variable no está en el archivo o no coincide exactamente con una opción/índice, "
                "esa variable cuenta como x=0 en ese cliente."
            )

    except Exception as e:
        st.error(f"No he podido leer/procesar el archivo: {e}")

st.divider()

# =========================================================
# Modo 1 cliente (manual)
# =========================================================

st.markdown("## Scoring manual de 1 cliente (inputs)")

ensure_state()

min_s, max_s = min_max_score_possible()
st.caption(f"Rango teórico con este modelo: mínimo {min_s:.2f}% · máximo {max_s:.2f}%")

c2, c3, c4 = st.columns(3)
with c2:
    if st.button("🎲 Cliente aleatorio Tipo A (alto)", key="btnA"):
        ok, s, msg = fill_random_client("A")
        (st.success if ok else st.warning)(f"{msg} Score: {s:.2f}%")
with c3:
    if st.button("🎲 Cliente aleatorio Tipo B (medio)", key="btnB"):
        ok, s, msg = fill_random_client("B")
        (st.success if ok else st.warning)(f"{msg} Score: {s:.2f}%")
with c4:
    if st.button("🎲 Cliente aleatorio Tipo C (bajo)", key="btnC"):
        ok, s, msg = fill_random_client("C")
        (st.success if ok else st.warning)(f"{msg} Score: {s:.2f}%")

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
            "x (0-1)": round(x, 6),
            "Contribución (%)": round(contrib, 4),
        })

with right:
    st.metric("Score total del cliente (%)", f"{total:.2f}")
    st.metric("Tipo", classify(total))
    st.caption("Tipo A/B/C se calcula con los umbrales configurados en la barra lateral.")

st.dataframe(pd.DataFrame(rows).sort_values("Contribución (%)", ascending=False), use_container_width=True)

st.markdown("## Fórmula")
st.latex(r"Score=\sum_i (Peso_i \cdot x_i)")
