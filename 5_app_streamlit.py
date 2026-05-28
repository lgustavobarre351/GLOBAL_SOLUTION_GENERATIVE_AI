"""
GAIE - Generative AI for Engineering
Script 5: Aplicação Web Streamlit

Interface interativa para previsão em tempo real da intensidade de
tempestades geomagnéticas com base nos dados do vento solar.

Execute: streamlit run 5_app_streamlit.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="GAIE — Previsão de Tempestades Geomagnéticas",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado (tema espacial escuro) ───────────────────────────────────
st.markdown("""
<style>
  /* Fundo geral */
  .stApp { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

  /* Títulos */
  h1, h2, h3 { color: #e6edf3 !important; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 20px; text-align: center; margin-bottom: 8px;
  }
  .metric-card .label { color: #8b949e; font-size: 12px; margin-bottom: 4px; }
  .metric-card .value { color: #4fc3f7; font-size: 26px; font-weight: bold; }

  /* Badge de nível G */
  .badge { padding: 6px 18px; border-radius: 20px; font-size: 18px;
           font-weight: bold; display: inline-block; margin: 4px; }
  .g0 { background: #1b5e20; color: #a5d6a7; }
  .g1 { background: #f9a825; color: #1a1a1a; }
  .g2 { background: #e65100; color: #fff3e0; }
  .g3 { background: #b71c1c; color: #ffcdd2; }
  .g4 { background: #6a1b9a; color: #e1bee7; }
  .g5 { background: #880e4f; color: #fce4ec; }

  /* Sliders */
  .stSlider > div { color: #8b949e; }
  div[data-testid="stMetricValue"] { color: #4fc3f7 !important; font-size: 28px !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { background: #161b22; color: #8b949e; border-radius: 6px 6px 0 0; }
  .stTabs [aria-selected="true"] { background: #1f6feb !important; color: white !important; }
  .stTabs [data-baseweb="tab-panel"] { background: #0d1117; border: 1px solid #30363d; border-radius: 0 0 8px 8px; padding: 16px; }
</style>
""", unsafe_allow_html=True)


# ── Funções auxiliares ───────────────────────────────────────────────────────

def kp_para_nivel_g(kp: float) -> int:
    """Converte KP Index para escala de tempestade NOAA G0–G5."""
    if kp < 5:   return 0
    elif kp < 6: return 1
    elif kp < 7: return 2
    elif kp < 8: return 3
    elif kp < 9: return 4
    else:        return 5


G_INFO = {
    0: {"label": "G0 — Calmo",     "classe": "g0", "emoji": "🟢", "cor": "#4caf50",
        "desc": "Condições calmas. Sem impacto geomagnético significativo."},
    1: {"label": "G1 — Menor",     "classe": "g1", "emoji": "🟡", "cor": "#fdd835",
        "desc": "Tempestade menor. Flutuações fracas em redes elétricas. Aurora possível em latitudes altas (>60°)."},
    2: {"label": "G2 — Moderada",  "classe": "g2", "emoji": "🟠", "cor": "#ff9800",
        "desc": "Tempestade moderada. Alarmes de tensão em sistemas elétricos. Aurora visível até ~55° de latitude."},
    3: {"label": "G3 — Forte",     "classe": "g3", "emoji": "🔴", "cor": "#f44336",
        "desc": "Tempestade forte. Irregularidades em satélites e GPS. Aurora visível até ~50° (ex: Alemanha, Canadá)."},
    4: {"label": "G4 — Severa",    "classe": "g4", "emoji": "🟣", "cor": "#9c27b0",
        "desc": "Tempestade severa. Possível perda de controle de satélites. Blackout de rádio HF. Aurora até ~45°."},
    5: {"label": "G5 — Extrema",   "classe": "g5", "emoji": "💜", "cor": "#880e4f",
        "desc": "Tempestade extrema. Blackouts elétricos em grande escala. Caso histórico: Quebec 1989 (9 horas sem energia)."},
}


@st.cache_resource
def carregar_modelos():
    """Carrega modelos com cache para evitar recarregamento desnecessário."""
    base = os.path.dirname(__file__)

    def p(rel): return os.path.join(base, rel)

    if not os.path.exists(p("models/best_regressor.pkl")):
        return None, None, None, None, None

    modelo_reg   = joblib.load(p("models/best_regressor.pkl"))
    modelo_clf   = joblib.load(p("models/best_classifier.pkl"))
    scaler       = joblib.load(p("models/scaler.pkl"))
    feature_cols = joblib.load(p("data/feature_cols.pkl"))

    metricas = {}
    path_metricas = p("outputs/metricas.json")
    if os.path.exists(path_metricas):
        with open(path_metricas, encoding="utf-8") as f:
            metricas = json.load(f)

    return modelo_reg, modelo_clf, scaler, feature_cols, metricas


def engenharia_features_usuario(inputs: dict, feature_cols: list) -> np.ndarray:
    """
    Calcula features derivadas a partir dos inputs do usuário.
    Segue a mesma lógica de 2_preprocessamento.py para consistência.
    """
    v   = inputs["velocidade_vento"]
    bz  = inputs["bz"]
    dens = inputs["densidade_protons"]
    bt  = inputs["campo_bt"]
    p   = inputs["pressao_dinamica"]
    T   = inputs["temperatura"]
    cme = inputs["cme"]
    v_cme = inputs["velocidade_cme"]
    flare = inputs["flare_classe"]
    hora  = inputs["hora"]
    mes   = inputs["mes"]
    bz_media_3h = inputs.get("bz_media_3h", bz)
    vel_media_3h = inputs.get("vel_media_3h", v)

    bz_neg    = max(-bz, 0)
    newell    = (v ** (4 / 3)) * (bz_neg ** (2 / 3)) if bz_neg > 0 else 0.0
    cme_bz    = cme * bz_neg
    hora_sin  = np.sin(2 * np.pi * hora / 24)
    hora_cos  = np.cos(2 * np.pi * hora / 24)
    mes_sin   = np.sin(2 * np.pi * mes / 12)
    mes_cos   = np.cos(2 * np.pi * mes / 12)

    row = {
        "velocidade_vento":    v,
        "bz":                  bz,
        "bz_negativo":         bz_neg,
        "densidade_protons":   dens,
        "campo_bt":            bt,
        "pressao_dinamica":    p,
        "temperatura":         T,
        "cme":                 float(cme),
        "velocidade_cme":      v_cme if cme else 0.0,
        "flare_classe":        float(flare),
        "newell_coupling":     newell,
        "cme_bz_interacao":    cme_bz,
        "hora_sin":            hora_sin,
        "hora_cos":            hora_cos,
        "mes_sin":             mes_sin,
        "mes_cos":             mes_cos,
        "bz_media_3h":         bz_media_3h,
        "velocidade_media_3h": vel_media_3h,
        "bz_media_6h":         bz_media_3h,   # proxy: usa mesma janela
        "velocidade_media_6h": vel_media_3h,
    }

    x = np.array([row[col] for col in feature_cols]).reshape(1, -1)
    return x


def gauge_kp(kp_value: float) -> go.Figure:
    """Cria gráfico de gauge para o KP Index."""
    cor = G_INFO[kp_para_nivel_g(kp_value)]["cor"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=kp_value,
        delta={"reference": 5, "increasing": {"color": "#f44336"},
               "decreasing": {"color": "#4caf50"}},
        title={"text": "KP Index Previsto", "font": {"size": 18, "color": "#e6edf3"}},
        number={"font": {"size": 38, "color": cor}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 9], "tickwidth": 1, "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e"}},
            "bar": {"color": cor, "thickness": 0.25},
            "bgcolor": "#161b22",
            "borderwidth": 1,
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 5], "color": "#0d3b1a"},
                {"range": [5, 6], "color": "#5c3d00"},
                {"range": [6, 7], "color": "#5c2a00"},
                {"range": [7, 8], "color": "#5c0000"},
                {"range": [8, 9], "color": "#3d0033"},
            ],
            "threshold": {
                "line": {"color": "#f44336", "width": 3},
                "thickness": 0.8,
                "value": 5,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        height=280,
        margin=dict(l=30, r=30, t=60, b=10),
    )
    return fig


def grafico_fatores_risco(inputs: dict, feature_cols: list) -> go.Figure:
    """Gráfico de barras mostrando os fatores de risco com base nos inputs."""
    fatores = {
        "Bz Sul (reconexão)":     min(max(-inputs["bz"], 0) / 30, 1),
        "Velocidade Vento":       min((inputs["velocidade_vento"] - 300) / 600, 1),
        "Evento CME":             1.0 if inputs["cme"] else 0.0,
        "Classe Flare":           inputs["flare_classe"] / 4,
        "Pressão Dinâmica":       min(inputs["pressao_dinamica"] / 30, 1),
        "Densidade de Prótons":   min(inputs["densidade_protons"] / 50, 1),
    }

    cores = [
        "#f44336" if v > 0.7 else "#ff9800" if v > 0.4 else "#4caf50"
        for v in fatores.values()
    ]

    fig = go.Figure(go.Bar(
        x=list(fatores.values()),
        y=list(fatores.keys()),
        orientation="h",
        marker_color=cores,
        marker_line_color="#30363d",
        marker_line_width=1,
    ))
    fig.update_layout(
        title="Fatores de Risco Detectados",
        title_font_color="#e6edf3",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        xaxis=dict(range=[0, 1], tickformat=".0%", color="#8b949e",
                   gridcolor="#30363d"),
        yaxis=dict(color="#8b949e"),
        font_color="#e6edf3",
        height=280,
        margin=dict(l=10, r=20, t=50, b=10),
    )
    return fig


def tabela_metricas(metricas: dict) -> None:
    """Exibe tabelas comparativas de métricas dos modelos."""
    if not metricas:
        st.warning("Arquivo outputs/metricas.json não encontrado. Execute o pipeline completo.")
        return

    # Regressão
    if "regressao" in metricas:
        st.subheader("📈 Regressão — KP Index")
        df_reg = pd.DataFrame(metricas["regressao"])
        df_reg = df_reg.rename(columns={"modelo": "Modelo", "RMSE": "RMSE ↓",
                                         "MAE": "MAE ↓", "R2": "R² ↑"})
        melhor = metricas.get("melhor_regressao", "")
        st.dataframe(
            df_reg.style
            .highlight_min(subset=["RMSE ↓", "MAE ↓"], color="#1b5e20")
            .highlight_max(subset=["R² ↑"], color="#1b5e20")
            .format({"RMSE ↓": "{:.4f}", "MAE ↓": "{:.4f}", "R² ↑": "{:.4f}"}),
            use_container_width=True,
        )
        st.info(f"✓ Melhor modelo de regressão: **{melhor}**")

    # Classificação
    if "classificacao" in metricas:
        st.subheader("🏷️ Classificação — Nível G")
        df_clf = pd.DataFrame(metricas["classificacao"])
        df_clf = df_clf.rename(columns={"modelo": "Modelo",
                                         "Accuracy": "Accuracy ↑",
                                         "F1_weighted": "F1-weighted ↑"})
        melhor = metricas.get("melhor_classificacao", "")
        st.dataframe(
            df_clf.style
            .highlight_max(subset=["Accuracy ↑", "F1-weighted ↑"], color="#1b5e20")
            .format({"Accuracy ↑": "{:.4f}", "F1-weighted ↑": "{:.4f}"}),
            use_container_width=True,
        )
        st.info(f"✓ Melhor modelo de classificação: **{melhor}**")


# ── Layout principal ─────────────────────────────────────────────────────────

def main():
    # Cabeçalho
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 5px 0;">
      <h1 style="color:#4fc3f7; font-size:2.4rem; margin:0;">🌌 GAIE — Geomagnetic AI Engine</h1>
      <p style="color:#8b949e; margin:4px 0 0 0; font-size:1.05rem;">
        Previsão em Tempo Real de Tempestades Geomagnéticas por Machine Learning<br>
        Integrado ao <a href="https://helius-zeta.vercel.app/" style="color:#4fc3f7;" target="_blank">HELIOS Space Intelligence Platform</a>
      </p>
    </div>
    <hr style="border-color:#30363d; margin: 12px 0 20px 0;">
    """, unsafe_allow_html=True)

    # Carrega modelos
    modelo_reg, modelo_clf, scaler, feature_cols, metricas = carregar_modelos()

    if modelo_reg is None:
        st.error("⚠️ Modelos não encontrados. Execute primeiro: `python 0_pipeline.py`")
        st.code("python 0_pipeline.py", language="bash")
        return

    # ── SIDEBAR — Inputs do Vento Solar ─────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛸 Parâmetros do Vento Solar")
        st.caption("Ajuste os valores para simular condições em tempo real")

        st.markdown("### 🌐 Campo Magnético")
        bz  = st.slider("Campo Bz (nT)", -60.0, 25.0, 0.0, 0.5,
                         help="Componente sul do IMF. Valores negativos causam reconexão magnética.")
        bt  = st.slider("Campo Total Bt (nT)", 1.0, 80.0, 8.0, 0.5)

        st.markdown("### 💨 Vento Solar")
        vel = st.slider("Velocidade (km/s)", 200, 1100, 450, 10,
                         help="Velocidade do vento solar. Típica: 300–600 km/s.")
        den = st.slider("Densidade de Prótons (p/cc)", 1.0, 100.0, 8.0, 0.5)
        pre = st.slider("Pressão Dinâmica (nPa)", 0.5, 40.0, 3.0, 0.5)
        tmp = st.slider("Temperatura (eV)", 5.0, 300.0, 70.0, 1.0)

        st.markdown("### ☀️ Eventos Solares")
        cme = st.checkbox("Ejeção de Massa Coronal (CME)", value=False)
        vel_cme = 0.0
        angulo_cme = 0.0
        if cme:
            vel_cme   = st.slider("Velocidade da CME (km/s)", 300, 3500, 900, 50)
            angulo_cme = st.slider("Ângulo da CME (°)", 0, 360, 0, 5,
                                    help="Ângulo de propagação em relação à eclíptica.")

        flare_map = {"Sem Flare": 0, "Classe B": 1, "Classe C": 2, "Classe M": 3, "Classe X": 4}
        flare_sel = st.selectbox("Classe de Flare Solar", list(flare_map.keys()))
        flare_classe = flare_map[flare_sel]

        st.markdown("### 🕐 Contexto Temporal")
        hora = st.slider("Hora do Dia (UTC)", 0, 23, 12, 1)
        mes  = st.slider("Mês do Ano", 1, 12, 6, 1)

        st.markdown("### 📊 Histórico Recente (3–6h)")
        bz_hist  = st.slider("Média Bz anterior (nT)", -50.0, 20.0, 0.0, 0.5,
                              help="Média do campo Bz nas últimas 3–6 horas.")
        vel_hist = st.slider("Média Velocidade anterior (km/s)", 200, 1000, 450, 10)

    # Monta inputs
    inputs = {
        "velocidade_vento":  float(vel),
        "bz":                float(bz),
        "densidade_protons": float(den),
        "campo_bt":          float(bt),
        "pressao_dinamica":  float(pre),
        "temperatura":       float(tmp),
        "cme":               int(cme),
        "velocidade_cme":    float(vel_cme),
        "flare_classe":      int(flare_classe),
        "hora":              hora,
        "mes":               mes,
        "bz_media_3h":       float(bz_hist),
        "vel_media_3h":      float(vel_hist),
    }

    # Calcula features e faz predição
    X_raw = engenharia_features_usuario(inputs, feature_cols)
    X_scaled = scaler.transform(X_raw)

    kp_pred    = float(np.clip(modelo_reg.predict(X_scaled)[0], 0, 9))
    g_pred     = int(modelo_clf.predict(X_scaled)[0])
    g_info     = G_INFO.get(g_pred, G_INFO[0])

    # Probabilidades do classificador (se disponível)
    if hasattr(modelo_clf, "predict_proba"):
        proba = modelo_clf.predict_proba(X_scaled)[0]
    else:
        proba = np.zeros(6)
        proba[g_pred] = 1.0

    # ── ABAS PRINCIPAIS ──────────────────────────────────────────────────────
    aba_pred, aba_shap, aba_metricas, aba_sobre = st.tabs([
        "🎯 Previsão em Tempo Real",
        "🔍 Interpretabilidade SHAP",
        "📊 Métricas dos Modelos",
        "ℹ️ Sobre o GAIE",
    ])

    # ════════════════════════════════════════════════════════════
    # ABA 1 — Previsão
    # ════════════════════════════════════════════════════════════
    with aba_pred:
        col_gauge, col_info = st.columns([1, 1.2])

        with col_gauge:
            st.plotly_chart(gauge_kp(kp_pred), use_container_width=True)

            nivel_classe = g_info["classe"]
            st.markdown(
                f"""<div style="text-align:center;">
                  <span class="badge {nivel_classe}">{g_info['emoji']} {g_info['label']}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(f"""
            <div style="text-align:center; color:#8b949e; font-size:13px; margin-top:8px;">
              {g_info['desc']}
            </div>""", unsafe_allow_html=True)

        with col_info:
            st.markdown("#### 📋 Resumo da Previsão")

            cols = st.columns(2)
            with cols[0]:
                st.metric("KP Index Previsto", f"{kp_pred:.2f}", delta=f"{kp_pred - 5:.2f} vs G1")
                st.metric("Nível da Tempestade", g_info["label"])
            with cols[1]:
                bz_neg = max(-bz, 0)
                newell = (vel ** (4/3)) * (bz_neg ** (2/3)) if bz_neg > 0 else 0
                st.metric("Bz Sul (driver)", f"{bz_neg:.1f} nT")
                st.metric("Acoplamento Newell", f"{newell/1e4:.2f} ×10⁴")

            # Probabilidades por classe G
            st.markdown("#### 📈 Probabilidade por Nível G")
            n_classes_modelo = len(proba)
            prob_labels = [f"G{i}" for i in range(n_classes_modelo)]
            prob_cores  = ["#4caf50", "#fdd835", "#ff9800", "#f44336", "#9c27b0", "#880e4f"][:n_classes_modelo]
            fig_prob = go.Figure(go.Bar(
                x=prob_labels,
                y=proba,
                marker_color=prob_cores[:n_classes_modelo],
                marker_line_color="#30363d",
                text=[f"{p:.1%}" for p in proba],
                textposition="outside",
                textfont_color="#e6edf3",
            ))
            fig_prob.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                xaxis_color="#8b949e", yaxis_color="#8b949e",
                yaxis_tickformat=".0%", yaxis_range=[0, 1.15],
                font_color="#e6edf3", height=200,
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")
        col_risco, col_cond = st.columns(2)

        with col_risco:
            st.plotly_chart(grafico_fatores_risco(inputs, feature_cols), use_container_width=True)

        with col_cond:
            st.markdown("#### 🛰️ Condições Simuladas")
            dados_tabela = {
                "Parâmetro": ["Campo Bz", "Velocidade", "Densidade", "Campo Bt",
                               "Pressão Din.", "CME", "Flare", "Hora UTC"],
                "Valor": [f"{bz:.1f} nT", f"{vel} km/s", f"{den:.1f} p/cc",
                          f"{bt:.1f} nT", f"{pre:.1f} nPa",
                          "Sim" if cme else "Não",
                          ["Nenhum", "B", "C", "M", "X"][flare_classe],
                          f"{hora:02d}:00"],
                "Status": [
                    "⚠️ Crítico" if bz < -15 else "🟡 Atenção" if bz < -5 else "✅ Normal",
                    "⚠️ Alto"    if vel > 700  else "🟡 Elevado" if vel > 500 else "✅ Normal",
                    "🟡 Elevada" if den > 20   else "✅ Normal",
                    "⚠️ Forte"   if bt > 25    else "🟡 Moderado" if bt > 10 else "✅ Normal",
                    "⚠️ Alta"    if pre > 15   else "🟡 Elevada"  if pre > 6  else "✅ Normal",
                    "⚠️ CME Ativo" if cme else "✅ Sem CME",
                    "⚠️ X" if flare_classe == 4 else "🟡 M" if flare_classe == 3 else "✅ Baixo",
                    "✅ OK",
                ],
            }
            st.dataframe(pd.DataFrame(dados_tabela), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════
    # ABA 2 — SHAP
    # ════════════════════════════════════════════════════════════
    with aba_shap:
        st.markdown("### 🔍 Interpretabilidade SHAP — Como o Modelo Toma Decisões")
        st.caption("SHAP (SHapley Additive exPlanations) quantifica a contribuição de cada variável para cada previsão.")

        base = os.path.dirname(__file__)
        plots_shap = {
            "📊 Summary Plot — Regressão (KP Index)":   "shap_plots/summary_regressao.png",
            "📊 Summary Plot — Classificação (Nível G)": "shap_plots/summary_classificacao.png",
            "📈 Importância Média — Regressão":          "shap_plots/bar_regressao.png",
            "📈 Importância Média — Classificação":      "shap_plots/bar_classificacao.png",
            "🔗 Dependence Plot — Campo Bz":             "shap_plots/dependence_bz_regressao.png",
            "🌊 Waterfall — Caso Extremo":               "shap_plots/waterfall_extremo_regressao.png",
        }

        col1, col2 = st.columns(2)
        cols_shap = [col1, col2]
        for i, (titulo, path_rel) in enumerate(plots_shap.items()):
            path_abs = os.path.join(base, path_rel)
            with cols_shap[i % 2]:
                st.markdown(f"**{titulo}**")
                if os.path.exists(path_abs):
                    st.image(path_abs, use_container_width=True)
                else:
                    st.warning(f"Execute `python 4_shap_interpretabilidade.py` para gerar este plot.")

        # Relatório textual
        st.markdown("---")
        st.markdown("### 📝 Interpretação Física das Variáveis")
        path_relatorio = os.path.join(base, "outputs/interpretacao_shap.txt")
        if os.path.exists(path_relatorio):
            with open(path_relatorio, encoding="utf-8") as f:
                st.code(f.read(), language="text")
        else:
            st.info("Relatório gerado após executar `python 4_shap_interpretabilidade.py`")

    # ════════════════════════════════════════════════════════════
    # ABA 3 — Métricas
    # ════════════════════════════════════════════════════════════
    with aba_metricas:
        st.markdown("### 📊 Comparação de Modelos")

        tabela_metricas(metricas)

        st.markdown("---")
        st.markdown("#### 🖼️ Gráficos Comparativos")
        base = os.path.dirname(__file__)
        col_r, col_c = st.columns(2)

        path_reg = os.path.join(base, "outputs/comparacao_regressao.png")
        path_clf = os.path.join(base, "outputs/comparacao_classificacao.png")

        with col_r:
            if os.path.exists(path_reg):
                st.image(path_reg, use_container_width=True)
            else:
                st.warning("Execute o pipeline para gerar o gráfico de regressão.")
        with col_c:
            if os.path.exists(path_clf):
                st.image(path_clf, use_container_width=True)
            else:
                st.warning("Execute o pipeline para gerar o gráfico de classificação.")

        # EDA
        path_eda = os.path.join(base, "outputs/eda_solar.png")
        if os.path.exists(path_eda):
            st.markdown("#### 🔭 Análise Exploratória de Dados (EDA)")
            st.image(path_eda, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # ABA 4 — Sobre
    # ════════════════════════════════════════════════════════════
    with aba_sobre:
        st.markdown("""
### 🌌 Sobre o GAIE — Geomagnetic AI Engine

**GAIE** é a camada preditiva de Machine Learning integrada ao
[**HELIOS**](https://helius-zeta.vercel.app/) — plataforma de inteligência de clima espacial
que monitora dados em tempo real da **NASA DONKI** e **NOAA SWPC**.

---
#### 🎯 Problema
Tempestades geomagnéticas causam danos bilionários:
- **Quebec, 1989**: colapso total da rede elétrica (9h sem energia para 6 milhões de pessoas)
- **Halloween Storms, 2003**: danos a satélites, apagões na Suécia, blackout de rádio
- **Custo estimado** de uma tempestade G5 hoje: US$ 0.6–2.6 trilhões (Lloyd's of London, 2013)

#### 🔬 Metodologia
| Etapa | Descrição |
|-------|-----------|
| **Dados** | Dataset sintético (1300 linhas × 13+ colunas) com distribuições físicas reais |
| **Feature Engineering** | Bz sul, acoplamento de Newell, interações CME×Bz, janelas temporais |
| **Regressão** | Random Forest vs XGBoost vs Ridge — prevê KP Index (0–9) |
| **Classificação** | Random Forest vs XGBoost vs Logistic Regression — prevê nível G |
| **Interpretação** | SHAP TreeExplainer — importância física de cada variável |
| **Deploy** | Streamlit com interface espacial em tempo real |

#### 📡 Fontes de Dados
- **NASA DONKI API**: `https://api.nasa.gov/DONKI` — Eventos CME e tempestades GST
- **NOAA SWPC**: `https://services.swpc.noaa.gov` — KP Index histórico
- **Dados sintéticos**: baseados em Borovsky & Denton (2006) e Newell et al. (2008)

#### 📚 Referências Científicas
1. Newell, P.T. et al. (2008). *A solar wind magnetosphere coupling function.*
   JGR, doi:10.1029/2007JA012825
2. Borovsky, J.E. & Denton, M.H. (2006). *Differences between CME-driven storms and CIR-driven storms.*
   JGR, doi:10.1029/2005JA011447
3. Richardson, I.G. & Cane, H.V. (2012). *Near-Earth solar wind magnetic fields.*
   JGR, doi:10.1029/2011JA017364

#### ⚙️ Como Executar
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar pipeline completo
python 0_pipeline.py

# Iniciar aplicação
streamlit run 5_app_streamlit.py
```

#### 🏗️ Estrutura do Projeto
```
gaie_helios/
├── 0_pipeline.py                  # Orquestrador do pipeline
├── 1_coleta_dados.py              # Coleta NASA DONKI + NOAA + sintético
├── 2_preprocessamento.py          # Limpeza + feature engineering + EDA
├── 3_modelos.py                   # Treino e comparação de modelos
├── 4_shap_interpretabilidade.py   # Análise SHAP
├── 5_app_streamlit.py             # Aplicação web
├── data/                          # Datasets e arrays processados
├── models/                        # Modelos treinados (.pkl)
├── outputs/                       # Gráficos e métricas
└── shap_plots/                    # Plots SHAP
```
        """)


if __name__ == "__main__":
    main()
