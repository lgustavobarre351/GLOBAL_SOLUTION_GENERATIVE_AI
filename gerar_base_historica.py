"""
GAIE - Generative AI for Engineering
Script: Geração da Base Histórica de Previsões

Carrega todo o dataset histórico, roda os modelos treinados em cada linha
e salva uma base enriquecida com as previsões do GAIE, pronta para consumo
por uma API REST.

Saídas geradas:
  data/historico_previsoes.csv    — base completa (11k+ linhas)
  data/historico_previsoes.json   — formato JSON para API
  data/resumo_estatistico.json    — estatísticas agregadas por período
  data/ultimas_24h.json           — previsões das últimas 24h (endpoint rápido)

Execute: python gerar_base_historica.py
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Engenharia de features (idêntica ao 2_preprocessamento.py) ────────────────
def aplicar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bz_negativo"]      = np.maximum(-df["bz"], 0)
    df["newell_coupling"]  = np.where(
        df["bz_negativo"] > 0,
        (df["velocidade_vento"] ** (4/3)) * (df["bz_negativo"] ** (2/3)),
        0.0
    )
    df["cme_bz_interacao"] = df["cme"] * df["bz_negativo"]

    ts = pd.to_datetime(df["timestamp"], format="mixed")
    df["hora_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hora_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["mes_sin"]  = np.sin(2 * np.pi * ts.dt.month / 12)
    df["mes_cos"]  = np.cos(2 * np.pi * ts.dt.month / 12)

    df["bz_media_3h"]           = df["bz"].rolling(3, min_periods=1).mean()
    df["velocidade_media_3h"]   = df["velocidade_vento"].rolling(3, min_periods=1).mean()
    df["bz_media_6h"]           = df["bz"].rolling(6, min_periods=1).mean()
    df["velocidade_media_6h"]   = df["velocidade_vento"].rolling(6, min_periods=1).mean()
    return df


def kp_para_nivel_g(kp: float) -> int:
    if kp < 5:   return 0
    elif kp < 6: return 1
    elif kp < 7: return 2
    elif kp < 8: return 3
    elif kp < 9: return 4
    else:        return 5


NIVEL_LABELS = {
    0: "G0 - Calmo",
    1: "G1 - Tempestade Menor",
    2: "G2 - Tempestade Moderada",
    3: "G3 - Tempestade Forte",
    4: "G4 - Tempestade Severa",
    5: "G5 - Tempestade Extrema",
}

FEATURE_COLS = [
    "velocidade_vento", "bz", "bz_negativo", "densidade_protons",
    "campo_bt", "pressao_dinamica", "temperatura", "cme", "velocidade_cme",
    "flare_classe", "newell_coupling", "cme_bz_interacao",
    "hora_sin", "hora_cos", "mes_sin", "mes_cos",
    "bz_media_3h", "velocidade_media_3h", "bz_media_6h", "velocidade_media_6h",
]


def main():
    print("\n" + "=" * 62)
    print("  GAIE — Geração da Base Histórica de Previsões")
    print("=" * 62)

    # ── 1. Carrega artefatos ───────────────────────────────────────────
    print("\n[1/5] Carregando modelos e dataset...")

    modelo_reg  = joblib.load(os.path.join("models", "best_regressor.pkl"))
    modelo_clf  = joblib.load(os.path.join("models", "best_classifier.pkl"))
    scaler      = joblib.load(os.path.join("models", "scaler.pkl"))
    nome_reg    = joblib.load(os.path.join("models", "best_regressor_name.pkl"))
    nome_clf    = joblib.load(os.path.join("models", "best_classifier_name.pkl"))

    df = pd.read_csv(os.path.join("data", "solar_wind_dataset.csv"))
    print(f"  Dataset carregado: {len(df):,} registros")
    print(f"  Modelo regressão : {nome_reg}")
    print(f"  Modelo classificação: {nome_clf}")

    # ── 2. Feature engineering ────────────────────────────────────────
    print("\n[2/5] Aplicando engenharia de features...")
    df = aplicar_features(df)

    # ── 3. Predições ──────────────────────────────────────────────────
    print("\n[3/5] Gerando previsões para todo o histórico...")

    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    # Regressão: KP Index previsto
    kp_previsto = np.clip(modelo_reg.predict(X_scaled), 0, 9)

    # Classificação: nível G previsto
    nivel_previsto = modelo_clf.predict(X_scaled)

    # Probabilidades por classe G
    if hasattr(modelo_clf, "predict_proba"):
        proba = modelo_clf.predict_proba(X_scaled)   # shape (n, n_classes)
        n_classes = proba.shape[1]
        for i in range(n_classes):
            df[f"prob_G{i}"] = (proba[:, i] * 100).round(1)
    else:
        n_classes = 6
        for i in range(n_classes):
            df[f"prob_G{i}"] = 0.0
        for idx, g in enumerate(nivel_previsto):
            df.loc[idx, f"prob_G{int(g)}"] = 100.0

    # Adiciona colunas de previsão
    df["kp_previsto"]        = kp_previsto.round(3)
    df["nivel_g_previsto"]   = nivel_previsto.astype(int)
    df["nivel_g_label"]      = df["nivel_g_previsto"].map(NIVEL_LABELS)
    df["nivel_g_real"]       = df["nivel_g"].astype(int)        # valor real (se disponível)
    df["kp_real"]            = df["kp_index"].round(3)
    df["erro_kp"]            = (df["kp_previsto"] - df["kp_real"]).round(3)
    df["acerto_nivel"]       = (df["nivel_g_previsto"] == df["nivel_g_real"]).astype(int)
    df["timestamp_previsao"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["fonte"]              = df["fonte"] if "fonte" in df.columns else "desconhecido"

    print(f"  Previsões geradas: {len(df):,} linhas")
    print(f"  Acurácia no histórico: {df['acerto_nivel'].mean()*100:.1f}%")

    # ── 4. Salva base completa ────────────────────────────────────────
    print("\n[4/5] Salvando arquivos...")

    # Colunas para exportar (ordem organizada para API)
    cols_api = [
        "timestamp", "fonte",
        # Inputs do vento solar
        "velocidade_vento", "bz", "densidade_protons", "campo_bt",
        "pressao_dinamica", "temperatura", "cme", "velocidade_cme", "flare_classe",
        # Features derivadas principais
        "bz_negativo", "newell_coupling", "cme_bz_interacao",
        # Previsões
        "kp_previsto", "nivel_g_previsto", "nivel_g_label",
        # Probabilidades por nível
        *[f"prob_G{i}" for i in range(n_classes)],
        # Valores reais (para comparação)
        "kp_real", "nivel_g_real", "erro_kp", "acerto_nivel",
        # Metadados
        "timestamp_previsao",
    ]
    cols_api = [c for c in cols_api if c in df.columns]
    df_export = df[cols_api].copy()

    # CSV completo
    path_csv = os.path.join("data", "historico_previsoes.csv")
    df_export.to_csv(path_csv, index=False)
    print(f"  ✓ CSV salvo: {path_csv} ({os.path.getsize(path_csv)/1024:.0f} KB)")

    # JSON completo (para API)
    path_json = os.path.join("data", "historico_previsoes.json")
    df_export["timestamp"] = df_export["timestamp"].astype(str)
    registros = df_export.to_dict(orient="records")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({
            "gerado_em": datetime.now().isoformat(),
            "modelo_regressao": nome_reg,
            "modelo_classificacao": nome_clf,
            "total_registros": len(registros),
            "periodo_inicio": str(df_export["timestamp"].min()),
            "periodo_fim":    str(df_export["timestamp"].max()),
            "registros": registros,
        }, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON salvo: {path_json} ({os.path.getsize(path_json)/1024:.0f} KB)")

    # Últimas 24h (endpoint rápido da API)
    df_export["timestamp_dt"] = pd.to_datetime(df_export["timestamp"], format="mixed")
    cutoff = df_export["timestamp_dt"].max() - pd.Timedelta("24h")
    df_24h = df_export[df_export["timestamp_dt"] >= cutoff].drop(columns=["timestamp_dt"])
    path_24h = os.path.join("data", "ultimas_24h.json")
    with open(path_24h, "w", encoding="utf-8") as f:
        json.dump({
            "gerado_em": datetime.now().isoformat(),
            "periodo": "ultimas_24h",
            "total_registros": len(df_24h),
            "registros": df_24h.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Últimas 24h: {path_24h} ({len(df_24h)} registros)")

    # ── 5. Resumo estatístico (agregações para dashboard/API) ─────────
    print("\n[5/5] Gerando resumo estatístico...")

    # Distribuição por nível G
    dist_g = df["nivel_g_previsto"].value_counts().sort_index()

    # Métricas agregadas por mês
    df["mes_ano"] = pd.to_datetime(df["timestamp"], format="mixed").dt.strftime("%Y-%m")
    por_mes = df.groupby("mes_ano").agg(
        registros=("kp_previsto", "count"),
        kp_medio=("kp_previsto", "mean"),
        kp_max=("kp_previsto", "max"),
        tempestades_g1plus=("nivel_g_previsto", lambda x: (x >= 1).sum()),
        tempestades_g3plus=("nivel_g_previsto", lambda x: (x >= 3).sum()),
    ).round(3).reset_index()

    resumo = {
        "gerado_em": datetime.now().isoformat(),
        "modelo_regressao": nome_reg,
        "modelo_classificacao": nome_clf,
        "total_registros": int(len(df)),
        "periodo_inicio": str(df["timestamp"].min()),
        "periodo_fim":    str(df["timestamp"].max()),
        "acuracia_geral": round(float(df["acerto_nivel"].mean()), 4),
        "rmse_historico": round(float(np.sqrt(((df["kp_previsto"] - df["kp_real"])**2).mean())), 4),
        "kp_medio_previsto": round(float(df["kp_previsto"].mean()), 3),
        "kp_maximo_previsto": round(float(df["kp_previsto"].max()), 3),
        "distribuicao_nivel_g": {
            NIVEL_LABELS[int(k)]: {"count": int(v), "percentual": round(v/len(df)*100, 1)}
            for k, v in dist_g.items()
        },
        "por_mes": por_mes.to_dict(orient="records"),
        "endpoints_disponiveis": {
            "/historico": "Todos os registros com previsões",
            "/historico/ultimas-24h": "Apenas últimas 24 horas",
            "/historico/tempestades": "Apenas eventos G1+ (nivel_g_previsto >= 1)",
            "/historico/resumo": "Estatísticas agregadas",
            "/previsao": "Previsão em tempo real (recebe parâmetros)",
        }
    }

    path_resumo = os.path.join("data", "resumo_estatistico.json")
    with open(path_resumo, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Resumo salvo: {path_resumo}")

    # ── Sumário final ─────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Base histórica gerada com sucesso!")
    print(f"  Total de previsões : {len(df):,}")
    print(f"  Acurácia histórica : {df['acerto_nivel'].mean()*100:.1f}%")
    print(f"  RMSE histórico     : {np.sqrt(((df['kp_previsto']-df['kp_real'])**2).mean()):.4f}")
    print(f"  KP máximo previsto : {df['kp_previsto'].max():.2f}")
    print("\n  Distribuição das previsões:")
    for g, cnt in dist_g.items():
        label = NIVEL_LABELS[int(g)]
        print(f"    {label}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
    print("\n  Arquivos prontos para a API:")
    print("    data/historico_previsoes.csv    ← base completa")
    print("    data/historico_previsoes.json   ← JSON para API")
    print("    data/ultimas_24h.json           ← endpoint /ultimas-24h")
    print("    data/resumo_estatistico.json    ← endpoint /resumo")
    print("=" * 62)


if __name__ == "__main__":
    main()
