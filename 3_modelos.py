"""
GAIE - Generative AI for Engineering
Script 3: Treinamento e Comparação de Modelos

Modelos de REGRESSÃO (prever KP Index contínuo 0–9):
  - Random Forest Regressor
  - XGBoost Regressor
  - Ridge Regression (baseline)

Modelos de CLASSIFICAÇÃO (prever nível G0–G5):
  - Random Forest Classifier
  - XGBoost Classifier
  - Logistic Regression (baseline)

Métricas de regressão : RMSE, MAE, R²
Métricas de classificação: Accuracy, F1-weighted, Confusion Matrix
"""

import os
import json
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix,
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

# ── Paleta de cores do tema espacial ────────────────────────────────────────
FUNDO  = "#0d1117"
PAINEL = "#161b22"
BORDA  = "#30363d"
TEXTO  = "#e6edf3"
SUBTXT = "#8b949e"
AZUL   = "#4fc3f7"
VERDE  = "#66bb6a"
ROXO   = "#ab47bc"


def carregar_dados():
    """Carrega os arrays pré-processados gerados pelo script 2."""
    dados = {}
    for split in ["train", "val", "test"]:
        dados[f"X_{split}"]     = np.load(os.path.join("data", f"X_{split}.npy"))
        dados[f"y_reg_{split}"] = np.load(os.path.join("data", f"y_reg_{split}.npy"))
        dados[f"y_clf_{split}"] = np.load(os.path.join("data", f"y_clf_{split}.npy"))
    print(f"  Treino: {dados['X_train'].shape[0]} | Validação: {dados['X_val'].shape[0]} | Teste: {dados['X_test'].shape[0]}")
    return dados


def avaliar_regressao(modelo, X, y, nome: str) -> dict:
    """Calcula métricas de regressão para um modelo."""
    pred = modelo.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae  = mean_absolute_error(y, pred)
    r2   = r2_score(y, pred)
    print(f"    {nome:30s} → RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")
    return {"modelo": nome, "RMSE": rmse, "MAE": mae, "R2": r2}


def avaliar_classificacao(modelo, X, y, nome: str) -> dict:
    """Calcula métricas de classificação para um modelo."""
    pred = modelo.predict(X)
    acc  = accuracy_score(y, pred)
    f1   = f1_score(y, pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y, pred)
    print(f"    {nome:30s} → Acc={acc:.4f} | F1-w={f1:.4f}")
    return {"modelo": nome, "Accuracy": acc, "F1_weighted": f1, "confusion_matrix": cm.tolist()}


def plotar_comparacao_regressao(resultados: list, path_saida: str):
    """Gera gráfico comparativo dos modelos de regressão."""
    nomes  = [r["modelo"] for r in resultados]
    rmses  = [r["RMSE"] for r in resultados]
    maes   = [r["MAE"]  for r in resultados]
    r2s    = [r["R2"]   for r in resultados]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(FUNDO)
    fig.suptitle("Comparação de Modelos — Regressão (KP Index)", color=TEXTO, fontsize=13, fontweight="bold")

    def barra(ax, vals, titulo, cor, fmt=".4f"):
        bars = ax.bar(nomes, vals, color=cor, edgecolor=BORDA, alpha=0.85)
        ax.set_facecolor(PAINEL)
        ax.set_title(titulo, color=TEXTO, fontsize=11)
        ax.tick_params(colors=SUBTXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDA)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    format(val, fmt), ha="center", va="bottom", color=TEXTO, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.2)

    barra(axes[0], rmses, "RMSE ↓ (menor é melhor)", AZUL)
    barra(axes[1], maes,  "MAE ↓ (menor é melhor)",  ROXO)
    barra(axes[2], r2s,   "R² ↑ (maior é melhor)",   VERDE)

    plt.tight_layout()
    plt.savefig(path_saida, dpi=150, bbox_inches="tight", facecolor=FUNDO)
    plt.close()
    print(f"  ✓ Gráfico salvo: {path_saida}")


def plotar_comparacao_classificacao(resultados: list, path_saida: str):
    """Gera gráfico comparativo dos modelos de classificação com matrizes de confusão."""
    n = len(resultados)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 11))
    fig.patch.set_facecolor(FUNDO)
    fig.suptitle("Comparação de Modelos — Classificação (Nível G)", color=TEXTO, fontsize=13, fontweight="bold")

    nomes  = [r["modelo"] for r in resultados]
    accs   = [r["Accuracy"] for r in resultados]
    f1s    = [r["F1_weighted"] for r in resultados]

    # Barras de métricas (linha 0)
    for i, res in enumerate(resultados):
        ax = axes[0, i] if n > 1 else axes[0]
        ax.set_facecolor(PAINEL)
        metricas = {"Accuracy": res["Accuracy"], "F1-weighted": res["F1_weighted"]}
        bars = ax.bar(list(metricas.keys()), list(metricas.values()),
                      color=[AZUL, VERDE], edgecolor=BORDA, alpha=0.85)
        ax.set_title(res["modelo"], color=TEXTO, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors=SUBTXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDA)
        for bar, val in zip(bars, metricas.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", color=TEXTO, fontsize=10)

    # Matrizes de confusão (linha 1)
    labels = ["G0", "G1", "G2", "G3", "G4", "G5"]
    for i, res in enumerate(resultados):
        ax = axes[1, i] if n > 1 else axes[1]
        cm = np.array(res["confusion_matrix"])
        n_classes = cm.shape[0]
        lbs = labels[:n_classes]
        sns.heatmap(cm, ax=ax, annot=True, fmt="d", cmap="Blues",
                    xticklabels=lbs, yticklabels=lbs, linewidths=0.5)
        ax.set_title(f"Conf. Matrix — {res['modelo']}", color=TEXTO, fontsize=9)
        ax.set_xlabel("Previsto", color=SUBTXT, fontsize=9)
        ax.set_ylabel("Real", color=SUBTXT, fontsize=9)
        ax.tick_params(colors=SUBTXT, labelsize=8)
        ax.set_facecolor(PAINEL)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path_saida, dpi=150, bbox_inches="tight", facecolor=FUNDO)
    plt.close()
    print(f"  ✓ Gráfico salvo: {path_saida}")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("outputs", "modelos"), exist_ok=True)

    print("\n" + "=" * 60)
    print("  GAIE — Script 3: Treinamento e Comparação de Modelos")
    print("=" * 60)

    print("\n[1/4] Carregando dados pré-processados...")
    dados = carregar_dados()
    X_train, X_val, X_test = dados["X_train"], dados["X_val"], dados["X_test"]
    y_reg_train, y_reg_val, y_reg_test = dados["y_reg_train"], dados["y_reg_val"], dados["y_reg_test"]
    y_clf_train, y_clf_val, y_clf_test = dados["y_clf_train"], dados["y_clf_val"], dados["y_clf_test"]

    # Combina treino + validação para treino final (common practice após seleção de hiperparâmetros)
    X_tv     = np.vstack([X_train, X_val])
    y_reg_tv = np.concatenate([y_reg_train, y_reg_val])
    y_clf_tv = np.concatenate([y_clf_train, y_clf_val])

    n_classes = len(np.unique(y_clf_tv))

    # ═══════════════════════════════════════════════════════════════
    # BLOCO 1: MODELOS DE REGRESSÃO
    # ═══════════════════════════════════════════════════════════════
    print("\n[2/4] Treinando modelos de REGRESSÃO (alvo: KP Index)...")

    modelos_reg = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0
        ),
        "Ridge (baseline)": Ridge(alpha=1.0),
    }

    resultados_reg = []
    melhor_reg_rmse = float("inf")
    melhor_reg_modelo = None
    melhor_reg_nome = ""

    for nome, modelo in modelos_reg.items():
        print(f"  Treinando {nome}...")
        modelo.fit(X_tv, y_reg_tv)
        metricas = avaliar_regressao(modelo, X_test, y_reg_test, nome)
        resultados_reg.append(metricas)
        if metricas["RMSE"] < melhor_reg_rmse:
            melhor_reg_rmse = metricas["RMSE"]
            melhor_reg_modelo = modelo
            melhor_reg_nome = nome

    print(f"\n  ★ Melhor regressão: {melhor_reg_nome} (RMSE={melhor_reg_rmse:.4f})")
    joblib.dump(melhor_reg_modelo, os.path.join("models", "best_regressor.pkl"))
    joblib.dump(melhor_reg_nome,   os.path.join("models", "best_regressor_name.pkl"))

    # ═══════════════════════════════════════════════════════════════
    # BLOCO 2: MODELOS DE CLASSIFICAÇÃO
    # ═══════════════════════════════════════════════════════════════
    print("\n[3/4] Treinando modelos de CLASSIFICAÇÃO (alvo: Nível G)...")

    modelos_clf = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0,
            use_label_encoder=False, eval_metric="mlogloss",
            num_class=n_classes,
        ),
        "Logistic Reg. (baseline)": LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced", C=1.0
        ),
    }

    resultados_clf = []
    melhor_clf_f1 = -1
    melhor_clf_modelo = None
    melhor_clf_nome = ""

    for nome, modelo in modelos_clf.items():
        print(f"  Treinando {nome}...")
        modelo.fit(X_tv, y_clf_tv)
        metricas = avaliar_classificacao(modelo, X_test, y_clf_test, nome)
        resultados_clf.append(metricas)
        if metricas["F1_weighted"] > melhor_clf_f1:
            melhor_clf_f1 = metricas["F1_weighted"]
            melhor_clf_modelo = modelo
            melhor_clf_nome = nome

    print(f"\n  ★ Melhor classificação: {melhor_clf_nome} (F1={melhor_clf_f1:.4f})")
    joblib.dump(melhor_clf_modelo, os.path.join("models", "best_classifier.pkl"))
    joblib.dump(melhor_clf_nome,   os.path.join("models", "best_classifier_name.pkl"))

    # ═══════════════════════════════════════════════════════════════
    # BLOCO 3: SALVAR MÉTRICAS E PLOTS
    # ═══════════════════════════════════════════════════════════════
    print("\n[4/4] Gerando gráficos comparativos...")

    metricas_finais = {
        "regressao": resultados_reg,
        "classificacao": [{k: v for k, v in r.items() if k != "confusion_matrix"}
                          for r in resultados_clf],
        "melhor_regressao": melhor_reg_nome,
        "melhor_classificacao": melhor_clf_nome,
    }
    with open(os.path.join("outputs", "modelos", "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(metricas_finais, f, ensure_ascii=False, indent=2)

    plotar_comparacao_regressao(resultados_reg, os.path.join("outputs", "modelos", "comparacao_regressao.png"))
    plotar_comparacao_classificacao(resultados_clf, os.path.join("outputs", "modelos", "comparacao_classificacao.png"))

    print("\n" + "=" * 60)
    print("  Resumo de Resultados:")
    print(f"  Melhor Regressão   : {melhor_reg_nome} — RMSE={melhor_reg_rmse:.4f}")
    print(f"  Melhor Classificação: {melhor_clf_nome} — F1={melhor_clf_f1:.4f}")
    print("\n  Artefatos salvos:")
    print("    models/best_regressor.pkl  |  models/best_classifier.pkl")
    print("    outputs/comparacao_regressao.png  |  outputs/comparacao_classificacao.png")
    print("    outputs/metricas.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
