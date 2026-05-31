"""
GAIE - Generative AI for Engineering
Script 5: Aplicação Web Streamlit — Design HELIOS

Execute: streamlit run 5_app_streamlit.py
"""

import os
import json
import base64
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="GAIE — HELIOS Geomagnetic AI Engine",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Equipe ───────────────────────────────────────────────────────────────────
EQUIPE = [
    ("Julia Azevedo Lins",           "RM99690"),
    ("Luis Gustavo Barreto Garrido", "RM99210"),
    ("Victor Hugo Aranda Forte",     "RM99667"),
    ("Guilherme Akio",               "RM98582"),
    ("Felipe Cortez",                "RM99750"),
]

# ── Logo em base64 (se existir) ──────────────────────────────────────────────
def carregar_logo_b64() -> str:
    caminho = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(caminho):
        with open(caminho, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

LOGO_B64 = carregar_logo_b64()

# ── CSS — Design System HELIOS ───────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

  /* ── Fundo e Base ── */
  .stApp { background-color: #0b0906; color: #F0EBE3; }
  * { font-family: 'Inter', sans-serif; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #110e0b 0%, #0e0b08 100%);
    border-right: 1px solid rgba(232,90,30,0.25);
  }
  section[data-testid="stSidebar"] * { color: #D4C8BC !important; }

  /* ── Cards de métricas ── */
  div[data-testid="stMetricValue"] {
    color: #FF7A3D !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    font-family: 'Rajdhani', sans-serif !important;
  }
  div[data-testid="stMetricLabel"] { color: #A89880 !important; font-size: 0.78rem !important; }
  div[data-testid="stMetricDelta"] svg { display: none; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #110e0b;
    border-bottom: 1px solid rgba(232,90,30,0.3);
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #A89880;
    border-radius: 4px 4px 0 0;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    padding: 8px 20px;
    border-bottom: 2px solid transparent;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(232,90,30,0.12) !important;
    color: #FF7A3D !important;
    border-bottom: 2px solid #E85A1E !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: #0e0b09;
    border: 1px solid rgba(232,90,30,0.15);
    border-top: none;
    padding: 20px;
    border-radius: 0 0 8px 8px;
  }

  /* ── Sliders ── */
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #E85A1E !important;
    border-color: #FF7A3D !important;
  }
  .stSlider [data-baseweb="slider"] div[data-baseweb="slider-track-fill"] {
    background: linear-gradient(90deg, #C04010, #E85A1E) !important;
  }

  /* ── Separadores ── */
  hr { border-color: rgba(232,90,30,0.2) !important; margin: 16px 0; }

  /* ── Dataframe ── */
  .stDataFrame { border: 1px solid rgba(232,90,30,0.2) !important; border-radius: 6px; }

  /* ── Selectbox / Checkbox ── */
  .stCheckbox label span { color: #D4C8BC !important; }
  div[data-baseweb="select"] > div { background: #1a1511 !important; border-color: rgba(232,90,30,0.3) !important; }

  /* ── Badges de nível G ── */
  .badge { padding: 8px 22px; border-radius: 4px; font-size: 1rem;
           font-weight: 700; display: inline-block; margin: 4px;
           letter-spacing: 0.06em; font-family: 'Rajdhani', sans-serif; }
  .g0 { background: rgba(30,100,40,0.3); color: #6FCF97; border: 1px solid #6FCF97; }
  .g1 { background: rgba(200,160,0,0.2); color: #F2C94C; border: 1px solid #F2C94C; }
  .g2 { background: rgba(220,100,0,0.2); color: #F2994A; border: 1px solid #F2994A; }
  .g3 { background: rgba(220,50,30,0.2); color: #EB5757; border: 1px solid #EB5757; }
  .g4 { background: rgba(140,30,140,0.2); color: #BB6BD9; border: 1px solid #BB6BD9; }
  .g5 { background: rgba(140,10,60,0.3); color: #FF79C6; border: 1px solid #FF79C6; }

  /* ── Cards personalizados ── */
  .helios-card {
    background: #141109;
    border: 1px solid rgba(232,90,30,0.2);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 12px;
  }
  .helios-card-title {
    color: #A89880;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .helios-card-value {
    color: #FF7A3D;
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Rajdhani', sans-serif;
  }

  /* ── Membro da equipe ── */
  .member-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(232,90,30,0.1);
  }
  .member-name { color: #D4C8BC; font-size: 0.88rem; }
  .member-rm { color: #E85A1E; font-size: 0.82rem; font-weight: 600;
               font-family: 'Rajdhani', sans-serif; letter-spacing: 0.05em; }

  /* ── Header principal ── */
  .main-header {
    border-bottom: 1px solid rgba(232,90,30,0.3);
    padding-bottom: 16px;
    margin-bottom: 20px;
  }
  .main-title {
    font-family: 'Rajdhani', sans-serif;
    color: #FF7A3D;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    line-height: 1.1;
  }
  .main-subtitle {
    color: #A89880;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .fiap-tag {
    color: #E85A1E;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def kp_para_nivel_g(kp: float) -> int:
    if kp < 5:   return 0
    elif kp < 6: return 1
    elif kp < 7: return 2
    elif kp < 8: return 3
    elif kp < 9: return 4
    else:        return 5

G_INFO = {
    0: {"label": "G0 — CALMO",    "classe": "g0", "emoji": "🟢", "cor": "#6FCF97",
        "desc": "Condições calmas. Sem impacto geomagnético significativo."},
    1: {"label": "G1 — MENOR",    "classe": "g1", "emoji": "🟡", "cor": "#F2C94C",
        "desc": "Tempestade menor. Aurora possível em latitudes altas (> 60°). Flutuações em redes elétricas."},
    2: {"label": "G2 — MODERADA", "classe": "g2", "emoji": "🟠", "cor": "#F2994A",
        "desc": "Tempestade moderada. Alarmes de tensão em sistemas elétricos. Aurora visível até ~55°."},
    3: {"label": "G3 — FORTE",    "classe": "g3", "emoji": "🔴", "cor": "#EB5757",
        "desc": "Tempestade forte. Irregularidades em satélites e GPS. Aurora visível até ~50° (Alemanha, Canadá)."},
    4: {"label": "G4 — SEVERA",   "classe": "g4", "emoji": "🟣", "cor": "#BB6BD9",
        "desc": "Tempestade severa. Possível perda de controle de satélites. Blackout de rádio HF. Aurora até ~45°."},
    5: {"label": "G5 — EXTREMA",  "classe": "g5", "emoji": "💜", "cor": "#FF79C6",
        "desc": "Tempestade extrema. Blackouts elétricos em grande escala (caso Quebec 1989: 9h sem energia para 6 mi pessoas)."},
}

@st.cache_resource
def carregar_modelos():
    base = os.path.dirname(__file__)
    def p(rel): return os.path.join(base, rel)
    if not os.path.exists(p("models/best_regressor.pkl")):
        return None, None, None, None, None
    modelo_reg   = joblib.load(p("models/best_regressor.pkl"))
    modelo_clf   = joblib.load(p("models/best_classifier.pkl"))
    scaler       = joblib.load(p("models/scaler.pkl"))
    feature_cols = joblib.load(p("data/feature_cols.pkl"))
    metricas = {}
    path_m = p("outputs/metricas.json")
    if os.path.exists(path_m):
        with open(path_m, encoding="utf-8") as f:
            metricas = json.load(f)
    return modelo_reg, modelo_clf, scaler, feature_cols, metricas

def engenharia_features_usuario(inputs: dict, feature_cols: list) -> np.ndarray:
    v   = inputs["velocidade_vento"]
    bz  = inputs["bz"]
    bz_neg = max(-bz, 0)
    newell = (v ** (4/3)) * (bz_neg ** (2/3)) if bz_neg > 0 else 0.0
    hora, mes = inputs["hora"], inputs["mes"]
    row = {
        "velocidade_vento":    v,
        "bz":                  bz,
        "bz_negativo":         bz_neg,
        "densidade_protons":   inputs["densidade_protons"],
        "campo_bt":            inputs["campo_bt"],
        "pressao_dinamica":    inputs["pressao_dinamica"],
        "temperatura":         inputs["temperatura"],
        "cme":                 float(inputs["cme"]),
        "velocidade_cme":      inputs["velocidade_cme"] if inputs["cme"] else 0.0,
        "flare_classe":        float(inputs["flare_classe"]),
        "newell_coupling":     newell,
        "cme_bz_interacao":    float(inputs["cme"]) * bz_neg,
        "hora_sin":            np.sin(2 * np.pi * hora / 24),
        "hora_cos":            np.cos(2 * np.pi * hora / 24),
        "mes_sin":             np.sin(2 * np.pi * mes / 12),
        "mes_cos":             np.cos(2 * np.pi * mes / 12),
        "bz_media_3h":         inputs.get("bz_media_3h", bz),
        "velocidade_media_3h": inputs.get("vel_media_3h", v),
        "bz_media_6h":         inputs.get("bz_media_3h", bz),
        "velocidade_media_6h": inputs.get("vel_media_3h", v),
    }
    return np.array([row[c] for c in feature_cols]).reshape(1, -1)

def gauge_kp(kp_value: float, nivel: int) -> go.Figure:
    cor = G_INFO[nivel]["cor"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=kp_value,
        delta={"reference": 5,
               "increasing": {"color": "#EB5757"},
               "decreasing": {"color": "#6FCF97"},
               "font": {"size": 14}},
        title={"text": "KP INDEX", "font": {"size": 14, "color": "#A89880",
               "family": "Rajdhani"}},
        number={"font": {"size": 48, "color": cor, "family": "Rajdhani"},
                "suffix": ""},
        gauge={
            "axis": {"range": [0, 9], "tickwidth": 1, "tickcolor": "#5A4A3A",
                     "tickfont": {"color": "#A89880", "size": 11}},
            "bar": {"color": cor, "thickness": 0.22},
            "bgcolor": "#0e0b09",
            "borderwidth": 1,
            "bordercolor": "rgba(232,90,30,0.3)",
            "steps": [
                {"range": [0, 5], "color": "rgba(30,80,40,0.25)"},
                {"range": [5, 6], "color": "rgba(200,160,0,0.15)"},
                {"range": [6, 7], "color": "rgba(220,100,0,0.15)"},
                {"range": [7, 8], "color": "rgba(220,50,30,0.2)"},
                {"range": [8, 9], "color": "rgba(140,10,60,0.25)"},
            ],
            "threshold": {
                "line": {"color": "#EB5757", "width": 2},
                "thickness": 0.75, "value": 5,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0b0906", plot_bgcolor="#0b0906",
        font_color="#F0EBE3", height=270,
        margin=dict(l=30, r=30, t=50, b=10),
    )
    return fig

def grafico_fatores(inputs: dict) -> go.Figure:
    bz_neg = max(-inputs["bz"], 0)
    fatores = {
        "Bz Sul (reconexão)": min(bz_neg / 30, 1),
        "Velocidade Vento":   min((inputs["velocidade_vento"] - 300) / 600, 1),
        "CME Ativo":          1.0 if inputs["cme"] else 0.0,
        "Classe Flare":       inputs["flare_classe"] / 4,
        "Pressão Dinâmica":   min(inputs["pressao_dinamica"] / 30, 1),
        "Densidade Prótons":  min(inputs["densidade_protons"] / 50, 1),
    }
    def cor(v):
        if v > 0.7: return "#EB5757"
        if v > 0.4: return "#F2994A"
        return "#6FCF97"
    cores = [cor(v) for v in fatores.values()]
    fig = go.Figure(go.Bar(
        x=list(fatores.values()), y=list(fatores.keys()),
        orientation="h", marker_color=cores,
        marker_line_color="rgba(0,0,0,0)", marker_line_width=0,
    ))
    fig.update_layout(
        title={"text": "FATORES DE RISCO", "font": {"size": 12, "color": "#A89880",
               "family": "Rajdhani"}, "x": 0},
        paper_bgcolor="#141109", plot_bgcolor="#141109",
        xaxis=dict(range=[0, 1], tickformat=".0%", color="#5A4A3A",
                   gridcolor="rgba(232,90,30,0.1)"),
        yaxis=dict(color="#A89880"),
        font_color="#D4C8BC", height=260,
        margin=dict(l=10, r=20, t=50, b=10),
    )
    return fig

def grafico_probabilidades(proba: np.ndarray) -> go.Figure:
    n = len(proba)
    labels = [f"G{i}" for i in range(n)]
    cores = ["#6FCF97", "#F2C94C", "#F2994A", "#EB5757", "#BB6BD9", "#FF79C6"][:n]
    fig = go.Figure(go.Bar(
        x=labels, y=proba,
        marker_color=cores,
        marker_line_color="rgba(0,0,0,0)",
        text=[f"{p:.1%}" for p in proba],
        textposition="outside",
        textfont={"color": "#D4C8BC", "size": 11, "family": "Rajdhani"},
    ))
    fig.update_layout(
        paper_bgcolor="#141109", plot_bgcolor="#141109",
        xaxis_color="#5A4A3A", yaxis_color="#5A4A3A",
        yaxis_tickformat=".0%", yaxis_range=[0, 1.2],
        yaxis_gridcolor="rgba(232,90,30,0.08)",
        font_color="#D4C8BC", height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig

def tabela_metricas_html(metricas: dict):
    if not metricas:
        st.warning("Execute o pipeline para gerar as métricas.")
        return
    if "regressao" in metricas:
        st.markdown('<p style="color:#A89880;letter-spacing:0.15em;font-size:0.8rem;text-transform:uppercase;">REGRESSÃO — KP Index</p>', unsafe_allow_html=True)
        df_reg = pd.DataFrame(metricas["regressao"])
        df_reg = df_reg.rename(columns={"modelo": "Modelo", "RMSE": "RMSE ↓", "MAE": "MAE ↓", "R2": "R² ↑"})
        st.dataframe(df_reg.style.format({"RMSE ↓": "{:.4f}", "MAE ↓": "{:.4f}", "R² ↑": "{:.4f}"}), use_container_width=True)
        st.markdown(f'<p style="color:#6FCF97;font-size:0.85rem;">★ Melhor: <b>{metricas.get("melhor_regressao","")}</b></p>', unsafe_allow_html=True)
    if "classificacao" in metricas:
        st.markdown('<p style="color:#A89880;letter-spacing:0.15em;font-size:0.8rem;text-transform:uppercase;margin-top:16px;">CLASSIFICAÇÃO — Nível G</p>', unsafe_allow_html=True)
        df_clf = pd.DataFrame(metricas["classificacao"])
        df_clf = df_clf.rename(columns={"modelo": "Modelo", "Accuracy": "Accuracy ↑", "F1_weighted": "F1-weighted ↑"})
        st.dataframe(df_clf.style.format({"Accuracy ↑": "{:.4f}", "F1-weighted ↑": "{:.4f}"}), use_container_width=True)
        st.markdown(f'<p style="color:#6FCF97;font-size:0.85rem;">★ Melhor: <b>{metricas.get("melhor_classificacao","")}</b></p>', unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    modelo_reg, modelo_clf, scaler, feature_cols, metricas = carregar_modelos()

    # ── HEADER PRINCIPAL ────────────────────────────────────────────
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        if LOGO_B64:
            st.markdown(f'<img src="data:image/png;base64,{LOGO_B64}" style="width:80px;margin-top:4px;">', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:3rem;margin-top:4px;">🌌</div>', unsafe_allow_html=True)
    with col_title:
        membros_inline = " · ".join([f"{n}" for n, _ in EQUIPE])
        st.markdown(f"""
        <div class="main-header">
          <div class="fiap-tag">FIAP · Global Solution 2026/1 · Generative AI for Engineering · Turma 4ESPY</div>
          <div class="main-title">GAIE — GEOMAGNETIC AI ENGINE</div>
          <div class="main-subtitle">Previsão de Tempestades Geomagnéticas por Machine Learning · Integrado ao
            <a href="https://helius-zeta.vercel.app/" style="color:#E85A1E;" target="_blank">HELIOS</a>
          </div>
          <div style="margin-top:8px;color:#5A4A3A;font-size:0.75rem;letter-spacing:0.05em;">{membros_inline}</div>
        </div>
        """, unsafe_allow_html=True)

    if modelo_reg is None:
        st.error("⚠️ Modelos não encontrados. Execute: `python 0_pipeline.py`")
        return

    # ── SIDEBAR ─────────────────────────────────────────────────────
    with st.sidebar:
        if LOGO_B64:
            st.markdown(f'<div style="text-align:center;padding:10px 0 16px 0;"><img src="data:image/png;base64,{LOGO_B64}" style="width:70px;"></div>', unsafe_allow_html=True)

        st.markdown('<p style="color:#E85A1E;letter-spacing:0.2em;font-size:0.8rem;font-weight:600;">⚡ PARÂMETROS DO VENTO SOLAR</p>', unsafe_allow_html=True)
        st.caption("Ajuste os valores para simular condições em tempo real")

        st.markdown('<p style="color:#A89880;letter-spacing:0.12em;font-size:0.75rem;text-transform:uppercase;margin-top:8px;">Campo Magnético</p>', unsafe_allow_html=True)
        bz  = st.slider("Campo Bz (nT)", -60.0, 25.0, 0.0, 0.5, help="Componente sul do IMF. Valores negativos causam reconexão magnética.")
        bt  = st.slider("Campo Total Bt (nT)", 1.0, 80.0, 8.0, 0.5)

        st.markdown('<p style="color:#A89880;letter-spacing:0.12em;font-size:0.75rem;text-transform:uppercase;margin-top:8px;">Vento Solar</p>', unsafe_allow_html=True)
        vel = st.slider("Velocidade (km/s)", 200, 1100, 450, 10)
        den = st.slider("Densidade de Prótons (p/cc)", 1.0, 100.0, 8.0, 0.5)
        pre = st.slider("Pressão Dinâmica (nPa)", 0.5, 40.0, 3.0, 0.5)
        tmp = st.slider("Temperatura (eV)", 5.0, 300.0, 70.0, 1.0)

        st.markdown('<p style="color:#A89880;letter-spacing:0.12em;font-size:0.75rem;text-transform:uppercase;margin-top:8px;">Eventos Solares</p>', unsafe_allow_html=True)
        cme = st.checkbox("CME Ativo (Ejeção de Massa Coronal)", value=False)
        vel_cme = 0.0
        if cme:
            vel_cme = st.slider("Velocidade da CME (km/s)", 300, 3500, 900, 50)
            st.slider("Ângulo da CME (°)", 0, 360, 0, 5)

        flare_map = {"Sem Flare": 0, "Classe B": 1, "Classe C": 2, "Classe M": 3, "Classe X": 4}
        flare_sel = st.selectbox("Classe de Flare Solar", list(flare_map.keys()))
        flare_classe = flare_map[flare_sel]

        st.markdown('<p style="color:#A89880;letter-spacing:0.12em;font-size:0.75rem;text-transform:uppercase;margin-top:8px;">Contexto Temporal</p>', unsafe_allow_html=True)
        hora = st.slider("Hora do Dia (UTC)", 0, 23, 12, 1)
        mes  = st.slider("Mês do Ano", 1, 12, 6, 1)

        st.markdown('<p style="color:#A89880;letter-spacing:0.12em;font-size:0.75rem;text-transform:uppercase;margin-top:8px;">Histórico Recente (3–6h)</p>', unsafe_allow_html=True)
        bz_hist  = st.slider("Média Bz anterior (nT)", -50.0, 20.0, 0.0, 0.5)
        vel_hist = st.slider("Média Velocidade anterior (km/s)", 200, 1000, 450, 10)

        # Equipe na sidebar
        st.markdown("---")
        st.markdown('<p style="color:#E85A1E;letter-spacing:0.2em;font-size:0.75rem;font-weight:600;">/ MISSION CREW</p>', unsafe_allow_html=True)
        for nome, rm in EQUIPE:
            st.markdown(f"""
            <div class="member-row">
              <span class="member-name">{nome}</span>
              <span class="member-rm">{rm}</span>
            </div>""", unsafe_allow_html=True)

    # ── PREDIÇÃO ────────────────────────────────────────────────────
    inputs = {
        "velocidade_vento": float(vel), "bz": float(bz),
        "densidade_protons": float(den), "campo_bt": float(bt),
        "pressao_dinamica": float(pre), "temperatura": float(tmp),
        "cme": int(cme), "velocidade_cme": float(vel_cme),
        "flare_classe": int(flare_classe), "hora": hora, "mes": mes,
        "bz_media_3h": float(bz_hist), "vel_media_3h": float(vel_hist),
    }
    X_raw    = engenharia_features_usuario(inputs, feature_cols)
    X_scaled = scaler.transform(X_raw)
    kp_pred  = float(np.clip(modelo_reg.predict(X_scaled)[0], 0, 9))
    g_pred   = int(modelo_clf.predict(X_scaled)[0])
    g_info   = G_INFO.get(g_pred, G_INFO[0])
    proba    = modelo_clf.predict_proba(X_scaled)[0] if hasattr(modelo_clf, "predict_proba") else np.zeros(6)

    # ── ABAS ────────────────────────────────────────────────────────
    tab_prev, tab_shap, tab_met, tab_sobre = st.tabs([
        "⚡  Previsão em Tempo Real",
        "🔍  Interpretabilidade SHAP",
        "📊  Métricas dos Modelos",
        "🛸  Sobre o GAIE",
    ])

    # ════════════ ABA 1 — PREVISÃO ════════════
    with tab_prev:
        col_g, col_r = st.columns([1, 1.15])

        with col_g:
            st.plotly_chart(gauge_kp(kp_pred, g_pred), use_container_width=True)
            nivel_classe = g_info["classe"]
            st.markdown(f"""
            <div style="text-align:center;margin-top:-8px;">
              <span class="badge {nivel_classe}">{g_info['emoji']} {g_info['label']}</span>
            </div>
            <div style="text-align:center;color:#A89880;font-size:0.82rem;margin-top:10px;padding:0 10px;">
              {g_info['desc']}
            </div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown('<p style="color:#E85A1E;letter-spacing:0.15em;font-size:0.8rem;font-weight:600;">RESUMO DA PREVISÃO</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            bz_neg = max(-bz, 0)
            newell = (vel**(4/3)) * (bz_neg**(2/3)) if bz_neg > 0 else 0
            with c1:
                st.metric("KP Index Previsto", f"{kp_pred:.2f}", f"{kp_pred-5:.2f} vs G1")
                st.metric("Nível da Tempestade", g_info["label"].split("—")[0].strip())
            with c2:
                st.metric("Bz Sul (driver)", f"{bz_neg:.1f} nT")
                st.metric("Acoplamento Newell", f"{newell/1e4:.2f} ×10⁴")

            st.markdown('<p style="color:#A89880;letter-spacing:0.12em;font-size:0.75rem;text-transform:uppercase;margin-top:10px;">Probabilidade por Nível G</p>', unsafe_allow_html=True)
            st.plotly_chart(grafico_probabilidades(proba), use_container_width=True)

        st.markdown("---")
        col_f, col_c = st.columns(2)

        with col_f:
            st.plotly_chart(grafico_fatores(inputs), use_container_width=True)

        with col_c:
            st.markdown('<p style="color:#A89880;letter-spacing:0.12em;font-size:0.75rem;text-transform:uppercase;">CONDIÇÕES SIMULADAS</p>', unsafe_allow_html=True)
            flare_nome = ["Nenhum", "B", "C", "M", "X"][flare_classe]
            dados_tabela = {
                "Parâmetro": ["Bz", "Velocidade", "Densidade", "Bt", "Pressão", "CME", "Flare", "Hora UTC"],
                "Valor": [f"{bz:.1f} nT", f"{vel} km/s", f"{den:.1f} p/cc",
                          f"{bt:.1f} nT", f"{pre:.1f} nPa",
                          "Ativo" if cme else "Inativo", flare_nome, f"{hora:02d}:00"],
                "Status": [
                    "⚠️ Crítico" if bz < -15 else "⚠️ Atenção" if bz < -5 else "✅ Normal",
                    "⚠️ Alto" if vel > 700 else "⚠️ Elevado" if vel > 500 else "✅ Normal",
                    "⚠️ Elevada" if den > 20 else "✅ Normal",
                    "⚠️ Forte" if bt > 25 else "⚠️ Moderado" if bt > 10 else "✅ Normal",
                    "⚠️ Alta" if pre > 15 else "⚠️ Elevada" if pre > 6 else "✅ Normal",
                    "⚠️ CME Ativo" if cme else "✅ Inativo",
                    "⚠️ Flare X" if flare_classe == 4 else "⚠️ Flare M" if flare_classe == 3 else "✅ Baixo",
                    "✅ OK",
                ],
            }
            st.dataframe(pd.DataFrame(dados_tabela), use_container_width=True, hide_index=True)

    # ════════════ ABA 2 — SHAP ════════════
    with tab_shap:
        st.markdown('<p style="color:#E85A1E;letter-spacing:0.15em;font-size:0.8rem;font-weight:600;">INTERPRETABILIDADE SHAP — COMO O MODELO DECIDE</p>', unsafe_allow_html=True)
        st.caption("SHAP (SHapley Additive exPlanations) quantifica a contribuição de cada variável para cada previsão individual.")
        base = os.path.dirname(__file__)
        plots = {
            "Summary Plot — Regressão (KP Index)":    "shap_plots/summary_regressao.png",
            "Summary Plot — Classificação (Nível G)": "shap_plots/summary_classificacao.png",
            "Importância Média — Regressão":           "shap_plots/bar_regressao.png",
            "Importância Média — Classificação":       "shap_plots/bar_classificacao.png",
            "Dependence Plot — Campo Bz":              "shap_plots/dependence_bz_regressao.png",
            "Waterfall — Caso Extremo":                "shap_plots/waterfall_extremo_regressao.png",
        }
        c1, c2 = st.columns(2)
        cols = [c1, c2]
        for i, (titulo_plot, path_rel) in enumerate(plots.items()):
            path_abs = os.path.join(base, path_rel)
            with cols[i % 2]:
                st.markdown(f'<p style="color:#A89880;font-size:0.8rem;letter-spacing:0.08em;">{titulo_plot.upper()}</p>', unsafe_allow_html=True)
                if os.path.exists(path_abs):
                    st.image(path_abs, use_container_width=True)
                else:
                    st.info("Execute `python 4_shap_interpretabilidade.py`")
        st.markdown("---")
        path_rel = os.path.join(base, "outputs/interpretacao_shap.txt")
        if os.path.exists(path_rel):
            st.markdown('<p style="color:#E85A1E;letter-spacing:0.15em;font-size:0.8rem;font-weight:600;">INTERPRETAÇÃO FÍSICA DAS VARIÁVEIS</p>', unsafe_allow_html=True)
            with open(path_rel, encoding="utf-8") as f:
                st.code(f.read(), language="text")

    # ════════════ ABA 3 — MÉTRICAS ════════════
    with tab_met:
        st.markdown('<p style="color:#E85A1E;letter-spacing:0.15em;font-size:0.8rem;font-weight:600;">COMPARAÇÃO DE MODELOS</p>', unsafe_allow_html=True)
        tabela_metricas_html(metricas)
        st.markdown("---")
        base = os.path.dirname(__file__)
        cr, cc = st.columns(2)
        with cr:
            p = os.path.join(base, "outputs/comparacao_regressao.png")
            if os.path.exists(p):
                st.image(p, use_container_width=True)
        with cc:
            p = os.path.join(base, "outputs/comparacao_classificacao.png")
            if os.path.exists(p):
                st.image(p, use_container_width=True)
        p_eda = os.path.join(base, "outputs/eda_solar.png")
        if os.path.exists(p_eda):
            st.markdown("---")
            st.markdown('<p style="color:#A89880;letter-spacing:0.12em;font-size:0.75rem;text-transform:uppercase;">Análise Exploratória de Dados (EDA)</p>', unsafe_allow_html=True)
            st.image(p_eda, use_container_width=True)

    # ════════════ ABA 4 — SOBRE ════════════
    with tab_sobre:
        col_info, col_team = st.columns([1.6, 1])

        with col_info:
            st.markdown("""
<p style="color:#E85A1E;letter-spacing:0.15em;font-size:0.8rem;font-weight:600;">SOBRE O GAIE</p>

**GAIE** (Geomagnetic AI Engine) é a camada preditiva de Machine Learning do
[**HELIOS Space Intelligence Platform**](https://helius-zeta.vercel.app/).

---
#### 🎯 Problema
Tempestades geomagnéticas causam danos bilionários em infraestrutura crítica:
- **Quebec, 1989**: colapso total da rede elétrica — 9h sem energia para 6 milhões de pessoas
- **Halloween Storms, 2003**: 30 satélites danificados, blackout HF global
- **Custo estimado** de uma tempestade G5 hoje: até **US$ 2,6 trilhões** (Lloyd's of London, 2013)

#### 🔬 Metodologia
| Etapa | Descrição |
|-------|-----------|
| **Dados** | 9.749 linhas reais (NOAA SWPC) + 1.500 sintéticas = 11.249 total |
| **Features** | 20 atributos com justificativa física (Newell, Borovsky) |
| **Regressão** | Random Forest vs XGBoost vs Ridge — prevê KP Index |
| **Classificação** | Random Forest vs XGBoost vs Logistic — prevê nível G |
| **SHAP** | TreeExplainer — importância física de cada variável |
| **Deploy** | Streamlit Cloud com interface espacial em tempo real |

#### 📡 Fontes de Dados
- **NOAA SWPC MAG**: campo Bz, Bt — 1 min de resolução
- **NOAA SWPC Plasma**: velocidade, densidade, temperatura — 1 min
- **NOAA SWPC KP Index**: índice geomagnético 3-horário
- **NASA DONKI API**: flares solares e tempestades GST

#### 📚 Referências
1. Newell et al. (2008). *A solar wind magnetosphere coupling function.* JGR. doi:10.1029/2007JA012825
2. Borovsky & Denton (2006). *Differences between CME-driven and CIR-driven storms.* JGR.
3. Richardson & Cane (2012). *Near-Earth solar wind magnetic fields.* JGR.

#### ⚙️ Execução
```bash
pip install -r requirements.txt
python 0_pipeline.py
streamlit run 5_app_streamlit.py
```
""", unsafe_allow_html=True)

        with col_team:
            logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="width:100px;display:block;margin:0 auto 20px auto;">' if LOGO_B64 else '<div style="font-size:4rem;text-align:center;margin-bottom:20px;">🌌</div>'
            membros_html = "".join([f"""
            <div class="member-row">
              <span class="member-name">{nome}</span>
              <span class="member-rm">{rm}</span>
            </div>""" for nome, rm in EQUIPE])

            st.markdown(f"""
<div style="background:#0e0b09;border:1px solid rgba(232,90,30,0.25);border-radius:8px;padding:24px;">
  {logo_html}
  <p style="color:#E85A1E;letter-spacing:0.2em;font-size:0.75rem;font-weight:600;text-align:center;margin-bottom:16px;">/ MISSION CREW · INTEGRANTES</p>
  {membros_html}
  <div style="margin-top:16px;border-top:1px solid rgba(232,90,30,0.15);padding-top:14px;">
    <p style="color:#5A4A3A;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;text-align:center;">FIAP · Global Solution 2026/1</p>
    <p style="color:#5A4A3A;font-size:0.72rem;letter-spacing:0.12em;text-align:center;">Generative AI for Engineering · Turma 4ESPY</p>
  </div>
  <div style="margin-top:14px;text-align:center;">
    <a href="https://github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI"
       style="color:#E85A1E;font-size:0.75rem;letter-spacing:0.08em;" target="_blank">
       GitHub do Projeto
    </a>
  </div>
</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
