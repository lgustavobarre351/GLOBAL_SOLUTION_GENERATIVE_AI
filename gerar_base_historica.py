"""
GAIE - Generative AI for Engineering
Script: Geração da Base de Dados para API

Gera dois blocos de dados:
  1. HISTÓRICO REAL  — dados reais do satélite DSCOVR (NOAA SWPC, últimos 7 dias)
                       com as previsões do modelo aplicadas a cada registro.
  2. FORECAST FUTURO — projeção das próximas 48 horas com base nas condições
                       atuais do vento solar, assumindo evolução gradual.

Saídas:
  data/base_api.csv             — histórico + forecast em um único arquivo
  data/historico_real.json      — apenas dados reais com previsões
  data/forecast_48h.json        — apenas previsões futuras (próximas 48h)
  data/resumo_estatistico.json  — estatísticas e metadados para a API

Execute: python gerar_base_historica.py
"""

import os
import json
import warnings
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Configuração ──────────────────────────────────────────────────────────────
NASA_API_KEY = os.getenv("NASA_API_KEY", "jwgRl7xlQuKDImPOikXJL9zTw9hnQxCKxuohl25e")

FEATURE_COLS = [
    "velocidade_vento", "bz", "bz_negativo", "densidade_protons",
    "campo_bt", "pressao_dinamica", "temperatura", "cme", "velocidade_cme",
    "flare_classe", "newell_coupling", "cme_bz_interacao",
    "hora_sin", "hora_cos", "mes_sin", "mes_cos",
    "bz_media_3h", "velocidade_media_3h", "bz_media_6h", "velocidade_media_6h",
]

NIVEL_LABELS = {
    0: "G0 - Calmo",
    1: "G1 - Tempestade Menor",
    2: "G2 - Tempestade Moderada",
    3: "G3 - Tempestade Forte",
    4: "G4 - Tempestade Severa",
    5: "G5 - Tempestade Extrema",
}


# ── 1. Coleta dados reais NOAA ────────────────────────────────────────────────
def coletar_noaa_mag():
    url = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df = df.rename(columns={"bz_gsm": "bz", "bt": "campo_bt"})
        df["timestamp"] = pd.to_datetime(df["time_tag"])
        for c in ["bz", "campo_bt"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["timestamp", "bz", "campo_bt"]].dropna()
    except Exception as e:
        print(f"  ✗ NOAA MAG: {e}")
        return pd.DataFrame()


def coletar_noaa_plasma():
    url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df = df.rename(columns={"density": "densidade_protons",
                                 "speed": "velocidade_vento",
                                 "temperature": "temperatura"})
        df["timestamp"] = pd.to_datetime(df["time_tag"])
        for c in ["densidade_protons", "velocidade_vento", "temperatura"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["timestamp", "velocidade_vento", "densidade_protons", "temperatura"]].dropna()
    except Exception as e:
        print(f"  ✗ NOAA Plasma: {e}")
        return pd.DataFrame()


def coletar_noaa_kp():
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data[0], dict):
            df = pd.DataFrame(data)
            df = df.rename(columns={"time_tag": "timestamp_kp", "Kp": "kp_real"})
        else:
            df = pd.DataFrame(data[1:], columns=data[0])
            df = df.rename(columns={"time_tag": "timestamp_kp"})
            df = df.rename(columns={df.columns[1]: "kp_real"})
        df["timestamp_kp"] = pd.to_datetime(df["timestamp_kp"])
        df["kp_real"] = pd.to_numeric(df["kp_real"], errors="coerce")
        return df[["timestamp_kp", "kp_real"]].dropna()
    except Exception as e:
        print(f"  ✗ NOAA KP: {e}")
        return pd.DataFrame()


def montar_dataset_real():
    print("  [NOAA] Coletando MAG (Bz)...")
    df_mag = coletar_noaa_mag()
    print(f"         → {len(df_mag):,} registros")

    print("  [NOAA] Coletando Plasma (velocidade, densidade)...")
    df_plasma = coletar_noaa_plasma()
    print(f"         → {len(df_plasma):,} registros")

    print("  [NOAA] Coletando KP Index...")
    df_kp = coletar_noaa_kp()
    print(f"         → {len(df_kp)} registros (3-horários)")

    if df_mag.empty or df_plasma.empty:
        raise RuntimeError("Falha ao coletar dados reais da NOAA.")

    # Merge MAG + Plasma
    df = pd.merge_asof(
        df_mag.sort_values("timestamp"),
        df_plasma.sort_values("timestamp"),
        on="timestamp", tolerance=pd.Timedelta("2min"), direction="nearest"
    )

    # Adiciona KP real (forward-fill de 3h para 1min)
    if not df_kp.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            df_kp.rename(columns={"timestamp_kp": "timestamp"}).sort_values("timestamp"),
            on="timestamp", direction="backward"
        )
        df["kp_real"] = df["kp_real"].ffill()
    else:
        df["kp_real"] = None

    df = df.dropna(subset=["bz", "velocidade_vento", "densidade_protons"])

    # Colunas que o modelo precisa mas não temos da NOAA
    df["cme"]          = 0
    df["velocidade_cme"] = 0.0
    df["flare_classe"] = 0

    # Pressão dinâmica
    mp = 1.6726e-27
    df["pressao_dinamica"] = np.clip(
        0.5 * mp * df["densidade_protons"] * 1e6 * (df["velocidade_vento"] * 1e3) ** 2 * 1e9,
        0.4, 45
    )

    df["tipo"]   = "historico_real"
    df["fonte"]  = "NOAA_SWPC_DSCOVR"
    return df.reset_index(drop=True)


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def aplicar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bz_negativo"]    = np.maximum(-df["bz"], 0)
    df["newell_coupling"] = np.where(
        df["bz_negativo"] > 0,
        (df["velocidade_vento"] ** (4/3)) * (df["bz_negativo"] ** (2/3)),
        0.0
    )
    df["cme_bz_interacao"] = df["cme"] * df["bz_negativo"]

    ts = pd.to_datetime(df["timestamp"])
    df["hora_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hora_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["mes_sin"]  = np.sin(2 * np.pi * ts.dt.month / 12)
    df["mes_cos"]  = np.cos(2 * np.pi * ts.dt.month / 12)

    df["bz_media_3h"]         = df["bz"].rolling(3, min_periods=1).mean()
    df["velocidade_media_3h"] = df["velocidade_vento"].rolling(3, min_periods=1).mean()
    df["bz_media_6h"]         = df["bz"].rolling(6, min_periods=1).mean()
    df["velocidade_media_6h"] = df["velocidade_vento"].rolling(6, min_periods=1).mean()
    return df


def kp_para_nivel_g(kp):
    if kp < 5:   return 0
    elif kp < 6: return 1
    elif kp < 7: return 2
    elif kp < 8: return 3
    elif kp < 9: return 4
    else:        return 5


# ── 3. Rodar modelo ───────────────────────────────────────────────────────────
def rodar_modelo(df, modelo_reg, modelo_clf, scaler):
    X = df[FEATURE_COLS].values
    X_s = scaler.transform(X)

    df["kp_previsto"]      = np.clip(modelo_reg.predict(X_s), 0, 9).round(3)
    df["nivel_g_previsto"] = modelo_clf.predict(X_s).astype(int)
    df["nivel_g_label"]    = df["nivel_g_previsto"].map(NIVEL_LABELS)

    if hasattr(modelo_clf, "predict_proba"):
        proba = modelo_clf.predict_proba(X_s)
        for i in range(proba.shape[1]):
            df[f"prob_G{i}"] = (proba[:, i] * 100).round(1)

    return df


# ── 4. Gerar Forecast Futuro (48h) ────────────────────────────────────────────
def gerar_forecast_futuro(df_real, modelo_reg, modelo_clf, scaler, horas=48):
    """
    Gera previsões para as próximas `horas` horas com base nas condições
    mais recentes do vento solar.

    Modelo de evolução:
      - Bz:        decai suavemente em direção a 0 (condição de equilíbrio)
      - Velocidade: decai em direção à média histórica (~450 km/s)
      - Demais:     mantidos próximos às condições recentes
    """
    # Média das últimas 6 horas como condição inicial
    ultimas = df_real.tail(360).mean(numeric_only=True)  # ~6h em minutos

    ts_inicio = pd.to_datetime(df_real["timestamp"].max()) + pd.Timedelta("1h")
    timestamps_futuros = pd.date_range(start=ts_inicio, periods=horas, freq="1h")

    linhas = []
    for i, ts in enumerate(timestamps_futuros):
        # Decaimento exponencial em direção ao estado de equilíbrio
        # Meia-vida de ~24h (0.5 = exp(-k*24) → k = ln(2)/24)
        decaimento = np.exp(-i * np.log(2) / 24)

        bz_fut  = float(ultimas.get("bz", 0)) * decaimento
        vel_fut = float(ultimas.get("velocidade_vento", 450)) * decaimento + 450 * (1 - decaimento)
        den_fut = float(ultimas.get("densidade_protons", 8)) * decaimento + 8 * (1 - decaimento)
        bt_fut  = abs(bz_fut) + 4.0

        mp = 1.6726e-27
        pre_fut = np.clip(
            0.5 * mp * den_fut * 1e6 * (vel_fut * 1e3) ** 2 * 1e9,
            0.4, 45
        )

        linhas.append({
            "timestamp":        ts,
            "tipo":             "forecast_futuro",
            "fonte":            "GAIE_ML_Forecast",
            "horas_a_frente":   i + 1,
            "bz":               round(bz_fut, 2),
            "campo_bt":         round(bt_fut, 2),
            "velocidade_vento": round(vel_fut, 1),
            "densidade_protons": round(den_fut, 2),
            "pressao_dinamica": round(pre_fut, 3),
            "temperatura":      float(ultimas.get("temperatura", 70)),
            "cme":              0,
            "velocidade_cme":   0.0,
            "flare_classe":     0,
            "kp_real":          None,
        })

    df_fut = pd.DataFrame(linhas)
    df_fut = aplicar_features(df_fut)
    df_fut = rodar_modelo(df_fut, modelo_reg, modelo_clf, scaler)

    # Margem de incerteza cresce com o tempo (±0.1 KP por hora)
    df_fut["incerteza_kp"] = (df_fut["horas_a_frente"] * 0.08).clip(upper=2.0).round(2)
    df_fut["kp_minimo"]    = np.clip(df_fut["kp_previsto"] - df_fut["incerteza_kp"], 0, 9).round(3)
    df_fut["kp_maximo"]    = np.clip(df_fut["kp_previsto"] + df_fut["incerteza_kp"], 0, 9).round(3)

    return df_fut


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)

    print("\n" + "=" * 62)
    print("  GAIE — Geração da Base de Dados para API")
    print("=" * 62)

    # Carrega modelos
    print("\n[1/5] Carregando modelos treinados...")
    modelo_reg  = joblib.load(os.path.join("models", "best_regressor.pkl"))
    modelo_clf  = joblib.load(os.path.join("models", "best_classifier.pkl"))
    scaler      = joblib.load(os.path.join("models", "scaler.pkl"))
    nome_reg    = joblib.load(os.path.join("models", "best_regressor_name.pkl"))
    nome_clf    = joblib.load(os.path.join("models", "best_classifier_name.pkl"))
    print(f"  Modelos: {nome_reg} (regressão) | {nome_clf} (classificação)")

    # Coleta dados reais NOAA
    print("\n[2/5] Coletando dados reais do satélite DSCOVR (NOAA SWPC)...")
    df_real = montar_dataset_real()
    print(f"  ✓ {len(df_real):,} registros reais coletados")
    print(f"  Período: {df_real['timestamp'].min()} → {df_real['timestamp'].max()}")

    # Feature engineering
    print("\n[3/5] Aplicando feature engineering...")
    df_real = aplicar_features(df_real)

    # Previsões no histórico real
    print("\n[4/5] Rodando modelo no histórico real...")
    df_real = rodar_modelo(df_real, modelo_reg, modelo_clf, scaler)
    df_real["horas_a_frente"] = 0   # 0 = dados do passado/presente
    df_real["incerteza_kp"]   = 0.0
    df_real["kp_minimo"]      = df_real["kp_previsto"]
    df_real["kp_maximo"]      = df_real["kp_previsto"]

    # Gera forecast das próximas 48 horas
    print("\n[5/5] Gerando forecast das próximas 48 horas...")
    df_forecast = gerar_forecast_futuro(df_real, modelo_reg, modelo_clf, scaler, horas=48)
    print(f"  ✓ {len(df_forecast)} previsões futuras geradas")
    print(f"  Período: {df_forecast['timestamp'].min()} → {df_forecast['timestamp'].max()}")

    # Consolida
    colunas_finais = [
        "timestamp", "tipo", "fonte", "horas_a_frente",
        # Parâmetros do vento solar
        "velocidade_vento", "bz", "densidade_protons", "campo_bt",
        "pressao_dinamica", "temperatura", "cme", "flare_classe",
        # Features derivadas principais
        "bz_negativo", "newell_coupling",
        # Previsões do modelo
        "kp_previsto", "kp_minimo", "kp_maximo", "incerteza_kp",
        "nivel_g_previsto", "nivel_g_label",
        # Probabilidades por nível G
        *[f"prob_G{i}" for i in range(6) if f"prob_G{i}" in df_real.columns],
        # Valor real (histórico tem, forecast não tem)
        "kp_real",
    ]
    colunas_finais = [c for c in colunas_finais if c in df_real.columns or c in df_forecast.columns]

    df_total = pd.concat(
        [df_real[colunas_finais], df_forecast[colunas_finais]],
        ignore_index=True
    )
    df_total["timestamp"] = df_total["timestamp"].astype(str)

    # ── Salva arquivos ────────────────────────────────────────────────
    print("\n  Salvando arquivos...")

    # CSV completo
    path_csv = os.path.join("data", "base_api.csv")
    df_total.to_csv(path_csv, index=False)
    print(f"  ✓ base_api.csv ({len(df_total):,} linhas | {os.path.getsize(path_csv)//1024} KB)")

    # JSON histórico real
    path_hist = os.path.join("data", "historico_real.json")
    df_hist = df_total[df_total["tipo"] == "historico_real"]
    with open(path_hist, "w", encoding="utf-8") as f:
        json.dump({
            "gerado_em":      datetime.now().isoformat(),
            "tipo":           "historico_real",
            "modelo":         nome_reg,
            "total":          int(len(df_hist)),
            "periodo_inicio": str(df_hist["timestamp"].min()),
            "periodo_fim":    str(df_hist["timestamp"].max()),
            "fonte":          "NOAA SWPC — Satélite DSCOVR L1",
            "registros":      df_hist.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    print(f"  ✓ historico_real.json ({len(df_hist):,} registros)")

    # JSON forecast futuro
    path_fc = os.path.join("data", "forecast_48h.json")
    df_fc = df_total[df_total["tipo"] == "forecast_futuro"]
    with open(path_fc, "w", encoding="utf-8") as f:
        json.dump({
            "gerado_em":         datetime.now().isoformat(),
            "tipo":              "forecast_futuro",
            "modelo":            nome_reg,
            "total":             int(len(df_fc)),
            "horizonte_horas":   48,
            "periodo_inicio":    str(df_fc["timestamp"].min()),
            "periodo_fim":       str(df_fc["timestamp"].max()),
            "nota":              "Forecast baseado na evolução esperada das condições atuais do vento solar. Incerteza cresce com o tempo.",
            "registros":         df_fc.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    print(f"  ✓ forecast_48h.json ({len(df_fc)} previsões futuras)")

    # Resumo estatístico — separando real de sintético (fix #4)
    dist_real = df_hist["nivel_g_previsto"].value_counts().sort_index()
    path_res = os.path.join("data", "resumo_estatistico.json")

    # aviso explícito se todos os eventos G1+ são sintéticos
    kp_max_real = float(df_hist["kp_previsto"].max())
    aviso_sintetico = (
        "AVISO: período de coleta foi quieto (KP máx real = {:.2f}). "
        "Eventos G1+ neste resumo são provenientes de dados sintéticos.".format(kp_max_real)
        if kp_max_real < 5 else None
    )

    with open(path_res, "w", encoding="utf-8") as f:
        json.dump({
            "gerado_em":              datetime.now().isoformat(),
            "modelo_regressao":       nome_reg,
            "modelo_classificacao":   nome_clf,
            "aviso":                  aviso_sintetico,
            "historico": {
                "total_registros":    int(len(df_hist)),
                "periodo_inicio":     str(df_hist["timestamp"].min()),
                "periodo_fim":        str(df_hist["timestamp"].max()),
                "fonte":              "NOAA SWPC — Satélite DSCOVR (100% dados reais)",
                "kp_medio":           round(float(df_hist["kp_previsto"].mean()), 3),
                "kp_maximo":          round(float(df_hist["kp_previsto"].max()), 3),
                "nota_storms":        "Todos os registros neste arquivo são dados reais. Ausência de G1+ indica período quieto — não frequência real de tempestades.",
                "distribuicao_G_real": {
                    NIVEL_LABELS[int(k)]: {"count": int(v), "percentual": round(v/len(df_hist)*100, 1)}
                    for k, v in dist_real.items()
                },
            },
            "forecast_48h": {
                "total_registros":    int(len(df_fc)),
                "periodo_inicio":     str(df_fc["timestamp"].min()),
                "periodo_fim":        str(df_fc["timestamp"].max()),
                "kp_previsto_agora":  round(float(df_fc.iloc[0]["kp_previsto"]), 3),
                "nivel_agora":        NIVEL_LABELS[int(df_fc.iloc[0]["nivel_g_previsto"])],
                "pior_previsao_48h":  NIVEL_LABELS[int(df_fc["nivel_g_previsto"].max())],
                "kp_max_48h":        round(float(df_fc["kp_previsto"].max()), 3),
            },
            "endpoints_api": {
                "GET /historico":               "Histórico real dos últimos 7 dias com previsões",
                "GET /forecast":                "Previsão das próximas 48 horas",
                "GET /forecast/{hora}":         "Previsão para uma hora específica (1-48)",
                "GET /resumo":                  "Estatísticas e situação atual",
                "GET /alertas":                 "Apenas registros com nivel_g_previsto >= 1",
                "POST /previsao":               "Previsão on-demand com parâmetros customizados",
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"  ✓ resumo_estatistico.json")

    # ── Sumário ────────────────────────────────────────────────────────
    kp_agora  = float(df_fc.iloc[0]["kp_previsto"])
    g_agora   = NIVEL_LABELS[int(df_fc.iloc[0]["nivel_g_previsto"])]
    kp_max48  = float(df_fc["kp_previsto"].max())
    g_max48   = NIVEL_LABELS[int(df_fc["nivel_g_previsto"].max())]

    print("\n" + "=" * 62)
    print("  CONCLUÍDO!")
    print(f"\n  HISTÓRICO REAL    : {len(df_hist):,} registros (NOAA DSCOVR)")
    print(f"  FORECAST FUTURO   : {len(df_fc)} previsões (próximas 48h)")
    print(f"\n  Condição AGORA    : {g_agora} (KP={kp_agora:.2f})")
    print(f"  Pior nas 48h      : {g_max48} (KP={kp_max48:.2f})")
    print("\n  Arquivos gerados:")
    print("    data/base_api.csv             ← tudo em um CSV")
    print("    data/historico_real.json      ← GET /historico")
    print("    data/forecast_48h.json        ← GET /forecast")
    print("    data/resumo_estatistico.json  ← GET /resumo")
    print("=" * 62)


if __name__ == "__main__":
    main()
