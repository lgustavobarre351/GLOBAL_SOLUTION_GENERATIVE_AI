"""
GAIE - Generative AI for Engineering
Script 2: Pré-processamento e Engenharia de Atributos

Etapas:
  1. Carrega data/solar_wind_dataset.csv
  2. Limpeza: remove duplicatas, trata outliers físicos
  3. Feature engineering: bz_negativo, Newell coupling, pressão dinâmica,
     janelas temporais (3h e 6h), componentes cíclicas, interações
  4. EDA: gera outputs/eda_solar.png
  5. Split 70/15/15 e escalonamento com RobustScaler
  6. Salva artefatos para uso pelos scripts seguintes
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# Colunas de features utilizadas nos modelos (ordem fixa para consistência)
FEATURE_COLS = [
    "velocidade_vento",
    "bz",
    "bz_negativo",
    "densidade_protons",
    "campo_bt",
    "pressao_dinamica",
    "temperatura",
    "cme",
    "velocidade_cme",
    "flare_classe",
    "newell_coupling",
    "cme_bz_interacao",
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
    "bz_media_3h",
    "velocidade_media_3h",
    "bz_media_6h",
    "velocidade_media_6h",
]


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicatas e trata outliers físicos com limites realistas do vento solar.
    Limites baseados em observações do satélite ACE/WIND (1996–2023).
    """
    n_inicial = len(df)

    # Remove duplicatas exatas
    df = df.drop_duplicates()

    # Limites físicos reais para cada variável
    limites = {
        "velocidade_vento": (200, 1200),   # km/s — valor máximo histórico ~1100
        "bz":               (-80, 30),     # nT
        "densidade_protons": (0.5, 200),   # p/cc
        "campo_bt":         (0, 100),      # nT
        "pressao_dinamica": (0.1, 60),     # nPa
        "temperatura":      (1, 500),      # eV
        "velocidade_cme":   (0, 4000),     # km/s
        "kp_index":         (0, 9),        # escala KP
    }

    for col, (lb, ub) in limites.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lb, upper=ub)

    n_final = len(df)
    print(f"  Limpeza: {n_inicial} → {n_final} registros ({n_inicial - n_final} removidos)")
    return df.reset_index(drop=True)


def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas com base em física do plasma e heliofísica.
    """
    df = df.copy()

    # Bz negativo (sul) — principal driver de reconexão magnética
    df["bz_negativo"] = np.maximum(-df["bz"], 0)

    # Função de acoplamento de Newell (2008): ε ≈ v^(4/3) × Bs^(2/3)
    # Quantifica a transferência de energia do vento solar para a magnetosfera
    df["newell_coupling"] = np.where(
        df["bz_negativo"] > 0,
        (df["velocidade_vento"] ** (4 / 3)) * (df["bz_negativo"] ** (2 / 3)),
        0.0,
    )

    # Interação CME × Bz sul — CMEs com campo sul intenso causam as maiores tempestades
    df["cme_bz_interacao"] = df["cme"] * df["bz_negativo"]

    # Componentes cíclicas do tempo (sin/cos para preservar periodicidade)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["hora_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hora_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        df["mes_sin"]  = np.sin(2 * np.pi * ts.dt.month / 12)
        df["mes_cos"]  = np.cos(2 * np.pi * ts.dt.month / 12)
    else:
        for col in ["hora_sin", "hora_cos", "mes_sin", "mes_cos"]:
            df[col] = 0.0

    # Janelas temporais (médias móveis de 3h e 6h)
    # Capturam a história recente do vento solar antes de uma tempestade
    df["bz_media_3h"]           = df["bz"].rolling(3, min_periods=1).mean()
    df["velocidade_media_3h"]   = df["velocidade_vento"].rolling(3, min_periods=1).mean()
    df["bz_media_6h"]           = df["bz"].rolling(6, min_periods=1).mean()
    df["velocidade_media_6h"]   = df["velocidade_vento"].rolling(6, min_periods=1).mean()

    return df


def gerar_eda(df: pd.DataFrame, path_saida: str):
    """Gera painel de Análise Exploratória de Dados (EDA)."""
    os.makedirs(os.path.dirname(path_saida), exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("GAIE — Análise Exploratória: Vento Solar & Tempestades Geomagnéticas",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    cor_hist = "#4fc3f7"
    cor_fundo = "#161b22"

    for ax in axes.flat:
        ax.set_facecolor(cor_fundo)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    def estilizar(ax, titulo, xlabel, ylabel="Frequência"):
        ax.set_title(titulo, color="white", fontsize=10)
        ax.set_xlabel(xlabel, color="#8b949e", fontsize=8)
        ax.set_ylabel(ylabel, color="#8b949e", fontsize=8)
        ax.tick_params(colors="#8b949e")

    # 1. Distribuição KP Index
    axes[0, 0].hist(df["kp_index"], bins=40, color=cor_hist, edgecolor="#0d1117", alpha=0.85)
    estilizar(axes[0, 0], "Distribuição do KP Index", "KP Index")
    axes[0, 0].axvline(5, color="#ff6b6b", linestyle="--", linewidth=1.2, label="G1")
    axes[0, 0].legend(fontsize=8, labelcolor="white")

    # 2. Distribuição Bz
    axes[0, 1].hist(df["bz"], bins=50, color="#ef5350", edgecolor="#0d1117", alpha=0.85)
    estilizar(axes[0, 1], "Distribuição do Campo Bz (IMF Sul)", "Bz (nT)")

    # 3. Velocidade do vento solar
    axes[0, 2].hist(df["velocidade_vento"], bins=40, color="#66bb6a", edgecolor="#0d1117", alpha=0.85)
    estilizar(axes[0, 2], "Distribuição da Velocidade do Vento Solar", "Velocidade (km/s)")

    # 4. KP por nível G (boxplot)
    g_labels = ["G0", "G1", "G2", "G3", "G4", "G5"]
    dados_box = [df[df["nivel_g"] == i]["kp_index"].values for i in range(6)]
    dados_box = [d for d in dados_box if len(d) > 0]
    bp = axes[1, 0].boxplot(dados_box, patch_artist=True, notch=False)
    cores_g = ["#4caf50", "#ffeb3b", "#ff9800", "#f44336", "#9c27b0", "#b71c1c"]
    for patch, cor in zip(bp["boxes"], cores_g):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        for item in bp[element]:
            item.set_color("#8b949e")
    axes[1, 0].set_xticklabels(g_labels[:len(dados_box)], color="#8b949e")
    estilizar(axes[1, 0], "KP Index por Nível de Tempestade G", "Nível G", "KP Index")

    # 5. Correlação Bz × KP Index
    axes[1, 1].scatter(df["bz"], df["kp_index"], alpha=0.15, s=5, color=cor_hist)
    estilizar(axes[1, 1], "Bz × KP Index", "Bz (nT)", "KP Index")

    # 6. Correlação Velocidade × KP Index
    axes[1, 2].scatter(df["velocidade_vento"], df["kp_index"], alpha=0.15, s=5, color="#ab47bc")
    estilizar(axes[1, 2], "Velocidade × KP Index", "Velocidade (km/s)", "KP Index")

    # 7. Contagem por nível G
    contagem = df["nivel_g"].value_counts().sort_index()
    cores_bar = [cores_g[i] for i in contagem.index]
    axes[2, 0].bar([g_labels[i] for i in contagem.index], contagem.values,
                   color=cores_bar, edgecolor="#0d1117", alpha=0.85)
    estilizar(axes[2, 0], "Amostras por Nível de Tempestade", "Nível G")

    # 8. Newell Coupling × KP Index
    axes[2, 1].scatter(df["newell_coupling"], df["kp_index"], alpha=0.15, s=5, color="#ffa726")
    estilizar(axes[2, 1], "Acoplamento Newell × KP Index", "Acoplamento Newell ε", "KP Index")

    # 9. Matriz de correlação (principais features)
    cols_corr = ["velocidade_vento", "bz_negativo", "densidade_protons",
                 "campo_bt", "pressao_dinamica", "cme", "newell_coupling", "kp_index"]
    corr = df[cols_corr].corr()
    sns.heatmap(corr, ax=axes[2, 2], cmap="coolwarm", center=0, annot=True,
                fmt=".2f", annot_kws={"size": 7}, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    axes[2, 2].set_title("Matriz de Correlação", color="white", fontsize=10)
    axes[2, 2].tick_params(colors="#8b949e", labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path_saida, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ EDA salva em: {path_saida}")


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("outputs", "eda"), exist_ok=True)

    print("\n" + "=" * 60)
    print("  GAIE — Script 2: Pré-processamento e Engenharia de Atributos")
    print("=" * 60)

    # Carrega dataset
    caminho_csv = os.path.join("data", "solar_wind_dataset.csv")
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Dataset não encontrado: {caminho_csv}\nExecute primeiro: python 1_coleta_dados.py")

    df = pd.read_csv(caminho_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    print(f"\n[1/5] Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")

    # Limpeza
    print("\n[2/5] Limpeza de dados...")
    df = limpar_dados(df)

    # Feature engineering
    print("\n[3/5] Engenharia de atributos...")
    df = engenharia_features(df)
    print(f"  Features criadas: bz_negativo, newell_coupling, cme_bz_interacao,")
    print(f"  hora_sin/cos, mes_sin/cos, bz_media_3/6h, velocidade_media_3/6h")

    # EDA
    print("\n[4/5] Gerando análise exploratória (EDA)...")
    gerar_eda(df, os.path.join("outputs", "eda", "eda_solar.png"))

    # Verificar que todas as feature cols existem
    faltando = [c for c in FEATURE_COLS if c not in df.columns]
    if faltando:
        raise ValueError(f"Features ausentes no DataFrame: {faltando}")

    # Split 70 / 15 / 15
    print("\n[5/5] Split e escalonamento...")
    X = df[FEATURE_COLS].values
    y_reg = df["kp_index"].values.astype(float)
    y_clf = df["nivel_g"].values.astype(int)

    # Primeira divisão: 70% treino, 30% resto
    X_train, X_temp, y_reg_train, y_reg_temp, y_clf_train, y_clf_temp = train_test_split(
        X, y_reg, y_clf, test_size=0.30, random_state=42, stratify=y_clf
    )
    # Divide o restante em 15% validação e 15% teste (50/50 do temp)
    X_val, X_test, y_reg_val, y_reg_test, y_clf_val, y_clf_test = train_test_split(
        X_temp, y_reg_temp, y_clf_temp, test_size=0.50, random_state=42, stratify=y_clf_temp
    )

    print(f"  Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]} | Teste: {X_test.shape[0]}")

    # RobustScaler — robusto a outliers (usa mediana e IQR)
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Salva artefatos
    np.save(os.path.join("data", "X_train.npy"), X_train_s)
    np.save(os.path.join("data", "X_val.npy"),   X_val_s)
    np.save(os.path.join("data", "X_test.npy"),  X_test_s)
    np.save(os.path.join("data", "y_reg_train.npy"), y_reg_train)
    np.save(os.path.join("data", "y_reg_val.npy"),   y_reg_val)
    np.save(os.path.join("data", "y_reg_test.npy"),  y_reg_test)
    np.save(os.path.join("data", "y_clf_train.npy"), y_clf_train)
    np.save(os.path.join("data", "y_clf_val.npy"),   y_clf_val)
    np.save(os.path.join("data", "y_clf_test.npy"),  y_clf_test)

    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join("data", "feature_cols.pkl"))

    # Salva também os dados NÃO escalonados para SHAP (TreeExplainer prefere valores originais)
    np.save(os.path.join("data", "X_test_raw.npy"), X_test)
    np.save(os.path.join("data", "X_train_raw.npy"), X_train)

    print(f"\n  Artefatos salvos:")
    print(f"    models/scaler.pkl  | data/feature_cols.pkl")
    print(f"    data/X_{{train,val,test}}.npy  | data/y_{{reg,clf}}_{{train,val,test}}.npy")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
