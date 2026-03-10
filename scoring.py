from __future__ import annotations

import json
import random
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import streamlit as st

st.set_page_config(
    page_title="Taller Scoring - Europea Seguros",
    page_icon="REDUCCION-AES-01.png",
    layout="wide"
)

import pandas as pd
from openpyxl import Workbook


# =========================================================
# 1) LÓGICA (usar gaps del JSON del taller)
# =========================================================

@dataclass(frozen=True)
class CategoryResult:
    j: int
    x: float                 # valor normalizado
    contribution: float      # w * x (proporción)
    contribution_pct: float  # w * x en %
    delta_from_prev: float   # incremento vs categoría anterior (proporción)
    delta_from_prev_pct: float


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def gaps_to_x(k: int, gaps: List[float], cap_mode: str = "clip") -> Dict:
    """
    Construye x(j) con x(k)=1 y gaps (penalizaciones) no negativas entre categorías.

    - K=1 = peor, K=k = mejor
    - gaps: longitud k-1
      gap_t = caída entre K=t (peor) y K=t+1 (mejor), medido en escala 0..1
    - Restricción deseada: sum(gaps) <= 1
    - cap_mode:
        clip  -> si sum>1 recorta el último gap para que sum=1
        scale -> si sum>1 reescala todos los gaps proporcionalmente para que sum=1
    """
    if k < 2:
        raise ValueError("k debe ser >= 2.")

    gaps = list(gaps or [])
    if len(gaps) < k - 1:
        gaps += [0.0] * ((k - 1) - len(gaps))
    if len(gaps) > k - 1:
        gaps = gaps[: k - 1]
    gaps = [max(0.0, float(g)) for g in gaps]

    s = sum(gaps)
    if s > 1.0 + 1e-12:
        if cap_mode == "scale" and s > 0:
            gaps = [g / s for g in gaps]  # ahora suma 1
        else:
            excess = s - 1.0
            gaps[-1] = max(0.0, gaps[-1] - excess)

    s_eff = sum(gaps)
    remaining = max(0.0, 1.0 - s_eff)

    # x_k = 1 y vamos bajando
    xs = [0.0] * k
    xs[-1] = 1.0
    acc = 0.0
    for idx in range(k - 2, -1, -1):
        acc += gaps[idx]
        xs[idx] = clamp01(1.0 - acc)

    xs[-1] = 1.0
    return {"x_values": xs, "gaps_eff": gaps, "sum_gaps": s_eff, "remaining": remaining}


def build_model_from_taller_json(
    model_dict: dict,
) -> Tuple[Dict[str, float], Dict[str, Tuple[List[str], List[float]]]]:
    """
    Lee el JSON exportado del taller y construye:
      WEIGHTS: {variable: peso_pct}
      CONFIG : {variable: (labels, xs)} donde xs = lista x(j) peor->mejor basada en gaps del JSON.
    """
    weights: Dict[str, float] = {}
    config: Dict[str, Tuple[List[str], List[float]]] = {}

    settings = model_dict.get("settings") or {}
    cap_mode = settings.get("cap_mode", "clip")

    for v in model_dict.get("variables", []):
        name = str(v.get("name", "")).strip()
        if not name:
            continue

        peso = float(v.get("peso_pct", 0.0))
        k = int(v.get("k", 3))
        labels = list(v.get("labels") or [])
        gaps = list(v.get("gaps") or [])

        # normalizar labels a longitud k
        if len(labels) < k:
            labels = labels + [""] * (k - len(labels))
        if len(labels) > k:
            labels = labels[:k]

        labels = [
            lab.strip() if isinstance(lab, str) and lab.strip() else f"K={i+1}"
            for i, lab in enumerate(labels)
        ]

        conv = gaps_to_x(k=k, gaps=gaps, cap_mode=cap_mode)
        xs = conv["x_values"]  # peor -> mejor (K=1 .. K=k)

        weights[name] = peso
        config[name] = (labels, xs)

    return weights, config


# =========================================================
# 2) APP
# =========================================================

st.set_page_config(page_title="Calculadora Scoring Cliente", layout="wide")
st.title("Calculadora de Scoring de Cliente (con modelo del taller)")
st.caption("Carga el JSON exportado del taller. Score = Σ(Peso% · x). Modo 1 cliente o archivo masivo.")


# -------------------
# Sidebar
# -------------------
view = st.sidebar.radio("Pantalla", ["A. Scoring", "B. Resumen estratégico"])
st.sidebar.image("LOGOTIPO-AES-05.png", use_container_width=True)
st.sidebar.markdown("---")

# 1) Clasificación primero
st.sidebar.header("Clasificación A/B/C/D/E")
DEFAULT_A_MIN = 80.0
DEFAULT_B_MIN = 65.0
DEFAULT_C_MIN = 50.0
DEFAULT_D_MIN = 30.0

a_min = st.sidebar.slider("Tipo A si Score ≥", 0.0, 100.0, DEFAULT_A_MIN, 1.0)
b_min = st.sidebar.slider("Tipo B si Score ≥", 0.0, 100.0, DEFAULT_B_MIN, 1.0)
c_min = st.sidebar.slider("Tipo C si Score ≥", 0.0, 100.0, DEFAULT_C_MIN, 1.0)
d_min = st.sidebar.slider("Tipo D si Score ≥", 0.0, 100.0, DEFAULT_D_MIN, 1.0)
st.sidebar.caption("Tipo E automáticamente desde 0 hasta el umbral de D")

st.sidebar.markdown("---")

# 2) Modelo después
st.sidebar.header("Modelo (JSON del taller)")
uploaded_model = st.sidebar.file_uploader("Sube el JSON exportado del taller", type=["json"])

# (opcional) lo dejamos solo para mostrarlo en UI, ya no se usa para xs
st.sidebar.caption("Nota: en esta calculadora los x se leen del JSON (gaps). El slider xmin_floor ya no aplica.")


def classify(score: float) -> str:
    if score >= a_min:
        return "A"
    if score >= b_min:
        return "B"
    if score >= c_min:
        return "C"
    if score >= d_min:
        return "D"
    return "E"


# --- Cargar JSON y construir modelo ANTES de usar WEIGHTS/CONFIG ---
if uploaded_model is None:
    st.info("Sube el JSON del taller en la barra lateral para cargar pesos/k/etiquetas/gaps.")
    st.stop()

try:
    model = json.load(uploaded_model)
except Exception as e:
    st.error(f"No he podido leer el JSON: {e}")
    st.stop()

WEIGHTS, CONFIG = build_model_from_taller_json(model)
VAR_LIST = list(WEIGHTS.keys())

if not VAR_LIST:
    st.error("El JSON no contiene variables válidas.")
    st.stop()


# =========================================================
# Helpers scoring (batch + manual)
# =========================================================

def x_from_value(var: str, val) -> float:
    labels, xs = CONFIG[var]

    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0

    # si viene índice numérico
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        idx = int(val)
        if 0 <= idx < len(labels):
            return float(xs[idx])
        return 0.0

    # si viene etiqueta exacta o string numérico
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


# =========================================================
# Helpers generación A/B/C/D/E
# =========================================================

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
      'max' -> índice con x máximo
      'min' -> índice con x mínimo
      'mid' -> índice intermedio (n//2)
    """
    for var in VAR_LIST:
        _, xs = CONFIG[var]
        n = len(xs)
        if n <= 1:
            st.session_state[f"sel_{var}"] = 0
            continue

        if mode == "max":
            best_idx = max(range(n), key=lambda i: xs[i])
            st.session_state[f"sel_{var}"] = best_idx
        elif mode == "min":
            worst_idx = min(range(n), key=lambda i: xs[i])
            st.session_state[f"sel_{var}"] = worst_idx
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
    A: score >= a_min
    B: b_min <= score < a_min
    C: c_min <= score < b_min
    D: d_min <= score < c_min
    E: score < d_min
    """
    if not (a_min >= b_min >= c_min >= d_min):
        set_all_indices("mid")
        s = score_from_session()
        return True, s, "Umbrales incoherentes. Revisa que A ≥ B ≥ C ≥ D."

    min_s, max_s = min_max_score_possible()

    if tipo == "A":
        lo, hi = a_min, max_s
    elif tipo == "B":
        lo, hi = b_min, a_min - 1e-9
    elif tipo == "C":
        lo, hi = c_min, b_min - 1e-9
    elif tipo == "D":
        lo, hi = d_min, c_min - 1e-9
    else:  # E
        lo, hi = min_s, d_min - 1e-9

    if lo > max_s + 1e-9:
        set_all_indices("max")
        s = score_from_session()
        return False, s, f"No se puede alcanzar {lo:.2f}%. Máximo teórico {max_s:.2f}%."
    if hi < min_s - 1e-9:
        set_all_indices("min")
        s = score_from_session()
        return False, s, f"No se puede bajar de {hi:.2f}%. Mínimo teórico {min_s:.2f}%."

    # A desde arriba con variación
    if tipo == "A":
        set_all_indices("max")
        s = score_from_session()
        if s < a_min:
            return False, s, f"Máximo {s:.2f}% no llega al umbral A."

        vars_shuffled = VAR_LIST[:]
        random.shuffle(vars_shuffled)

        for var in vars_shuffled:
            key = f"sel_{var}"
            _, xs = CONFIG[var]
            idx = int(st.session_state[key])

            candidates = [i for i in range(len(xs)) if xs[i] < xs[idx]]
            if not candidates:
                continue
            new_idx = max(candidates, key=lambda i: xs[i])  # el peor más cercano

            st.session_state[key] = new_idx
            s2 = score_from_session()

            if s2 >= a_min:
                s = s2
            else:
                st.session_state[key] = idx

        return True, s, f"Cliente A generado (≥ {a_min:.2f}%)."

    # E desde abajo
    if tipo == "E":
        set_all_indices("min")
        s = score_from_session()
        if s >= d_min:
            return False, s, "El mínimo teórico no cae por debajo del umbral D. Ajusta d_min."
        return True, s, f"Cliente E generado (< {d_min:.2f}%)."

    # B/C/D: aleatorio controlado
    target_mid = (lo + hi) / 2.0
    best_state = None
    best_dist = None
    best_score = None

    for _ in range(tries):
        state = {}
        for var in VAR_LIST:
            _, xs = CONFIG[var]
            state[f"sel_{var}"] = random.randint(0, len(xs) - 1)

        s = score_from_state(state)

        if lo <= s <= hi:
            for k, v in state.items():
                st.session_state[k] = v
            return True, s, f"Cliente {tipo} generado correctamente."

        dist = abs(s - target_mid)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_state = state
            best_score = s

    if best_state is not None:
        for k, v in best_state.items():
            st.session_state[k] = v
        return False, float(best_score), f"No encontré un {tipo} perfecto. Te dejo el más cercano."

    set_all_indices("mid")
    s = score_from_session()
    return False, s, "Fallback a cliente intermedio."


def contributions_from_state(state: Dict[str, int]) -> Tuple[pd.DataFrame, float]:
    rows = []
    total = 0.0
    for var, w in WEIGHTS.items():
        labels, xs = CONFIG[var]
        idx = int(state.get(f"sel_{var}", 0))
        idx = max(0, min(idx, len(xs) - 1))
        x = float(xs[idx])
        contrib = w * x
        total += contrib
        rows.append({
            "Variable": var,
            "Selección": labels[idx] if 0 <= idx < len(labels) else f"idx={idx}",
            "Peso (%)": w,
            "x": x,
            "Contribución (%)": contrib,
        })

    df = pd.DataFrame(rows).sort_values("Contribución (%)", ascending=False)
    df["Contribución (%)"] = df["Contribución (%)"].round(4)
    return df, float(total)


def make_representative_state(tipo: str) -> Dict[str, int]:
    ensure_state()
    fill_random_client(tipo, tries=350)
    return current_state_dict()


# =========================================================
# VISTA A: SCORING
# =========================================================
if view == "A. Scoring":

    st.markdown("## Subir archivo para scoring masivo (varias filas)")
    uploaded = st.file_uploader(
        "Sube CSV o Excel con una fila por cliente (columnas = variables)",
        type=["csv", "xlsx"]
    )

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

                dist = (
                    df_res["Tipo"]
                    .value_counts(normalize=True)
                    .reindex(["A", "B", "C", "D", "E"])
                    .fillna(0) * 100
                )

                cA, cB, cC, cD, cE = st.columns(5)
                cA.metric("% Tipo A", f"{dist['A']:.1f}%")
                cB.metric("% Tipo B", f"{dist['B']:.1f}%")
                cC.metric("% Tipo C", f"{dist['C']:.1f}%")
                cD.metric("% Tipo D", f"{dist['D']:.1f}%")
                cE.metric("% Tipo E", f"{dist['E']:.1f}%")

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

                st.dataframe(
                    df_res[show_cols].sort_values("Score_total_%", ascending=False),
                    use_container_width=True
                )

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

    st.markdown("## Scoring manual de 1 cliente (inputs)")

    ensure_state()
    min_s, max_s = min_max_score_possible()
    st.caption(f"Rango teórico con este modelo: mínimo {min_s:.2f}% · máximo {max_s:.2f}%")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("🎲 Tipo A (muy alto)", key="btnA"):
            ok, s, msg = fill_random_client("A")
            (st.success if ok else st.warning)(f"{msg} Score: {s:.2f}%")

    with c2:
        if st.button("🎲 Tipo B (alto)", key="btnB"):
            ok, s, msg = fill_random_client("B")
            (st.success if ok else st.warning)(f"{msg} Score: {s:.2f}%")

    with c3:
        if st.button("🎲 Tipo C (medio)", key="btnC"):
            ok, s, msg = fill_random_client("C")
            (st.success if ok else st.warning)(f"{msg} Score: {s:.2f}%")

    with c4:
        if st.button("🎲 Tipo D (bajo)", key="btnD"):
            ok, s, msg = fill_random_client("D")
            (st.success if ok else st.warning)(f"{msg} Score: {s:.2f}%")

    with c5:
        if st.button("🎲 Tipo E (muy bajo)", key="btnE"):
            ok, s, msg = fill_random_client("E")
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
        st.caption("Tipo A/B/C/D/E se calcula con los umbrales configurados en la barra lateral.")

    st.dataframe(
        pd.DataFrame(rows).sort_values("Contribución (%)", ascending=False),
        use_container_width=True
    )

    st.markdown("## Fórmula")
    st.latex(r"Score=\sum_i (Peso_i \cdot x_i)")

# =========================================================
# VISTA B: RESUMEN ESTRATÉGICO
# =========================================================
else:
    st.markdown("## 👤 Resumen estratégico (buyer persona)")

    min_s, max_s = min_max_score_possible()
    st.info(
        f"Con el modelo actual, el score teórico va de **{min_s:.2f}%** a **{max_s:.2f}%**. "
        f"Umbrales: **A ≥ {a_min:.1f}%**, **B ≥ {b_min:.1f}%**, **C ≥ {c_min:.1f}%**, **D ≥ {d_min:.1f}%**, **E < {d_min:.1f}%**."
    )

    if not (a_min >= b_min >= c_min >= d_min):
        st.warning("⚠️ Umbrales incoherentes: asegúrate de que **A ≥ B ≥ C ≥ D** (E queda por debajo de D).")

    types = ["A", "B", "C", "D", "E"]
    icons = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "E": "🔴"}
    cols = st.columns(5)

    def drivers_md(df: pd.DataFrame, topn: int = 5) -> str:
        top = df.head(topn)
        lines = []
        for _, r in top.iterrows():
            lines.append(f"- **{r['Variable']}** → *{r['Selección']}* (**{r['Contribución (%)']:.2f}%**)")
        return "\n".join(lines)

    for i, t in enumerate(types):
        state = make_representative_state(t)
        dfT, scoreT = contributions_from_state(state)

        with cols[i]:
            st.markdown(f"### {icons[t]} Cliente Tipo {t}")
            st.metric("Ejemplo generado", f"{scoreT:.2f}%")

            st.markdown("**Drivers (top 5)**")
            st.markdown(drivers_md(dfT, topn=5))

    st.divider()
    st.markdown("### 🧾 Notas")
    st.write("Los ejemplos se recalculan con tu JSON y los umbrales A/B/C/D/E actuales.")
