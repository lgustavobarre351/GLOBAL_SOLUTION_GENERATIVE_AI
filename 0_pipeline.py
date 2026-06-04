"""
GAIE - Generative AI for Engineering
Script 0: Pipeline Orquestrador

Executa todos os scripts do pipeline em sequência:
  1. Coleta de dados
  2. Pré-processamento e feature engineering
  3. Treinamento e comparação de modelos
  4. Interpretabilidade com SHAP

Ao final, exibe sumário completo dos artefatos gerados.

Uso: python 0_pipeline.py
"""

import os
import sys
import time
import subprocess
import platform

# Configura a chave da NASA antes de tudo
os.environ["NASA_API_KEY"] = "jwgRl7xlQuKDImPOikXJL9zTw9hnQxCKxuohl25e"

# Garante que o diretório de trabalho é o mesmo do script
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       GAIE — Generative AI for Engineering                   ║
║       Pipeline de ML: Previsão de Tempestades Geomagnéticas  ║
║       Integrado ao HELIOS Space Intelligence Platform        ║
╚══════════════════════════════════════════════════════════════╝
""")


def instalar_dependencias():
    """Instala dependências do requirements.txt se ausentes."""
    print("[PRÉ] Verificando e instalando dependências...")
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    resultado = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_path, "--quiet"],
        capture_output=True, text=True
    )
    if resultado.returncode != 0:
        print(f"  ⚠️ Aviso na instalação: {resultado.stderr[:300]}")
    else:
        print("  ✓ Dependências verificadas")


def executar_script(nome: str) -> bool:
    """
    Executa um script Python como subprocesso.
    Retorna True se bem-sucedido, False caso contrário.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), nome)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # garante encoding correto no Windows

    inicio = time.time()
    resultado = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,   # mostra saída em tempo real
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    duracao = time.time() - inicio

    if resultado.returncode != 0:
        print(f"\n  ✗ ERRO ao executar {nome} (código {resultado.returncode})")
        return False

    print(f"\n  ✓ {nome} concluído em {duracao:.1f}s")
    return True


def verificar_artefatos():
    """Verifica e lista todos os artefatos gerados pelo pipeline."""
    print("\n" + "=" * 62)
    print("  SUMÁRIO DE ARTEFATOS GERADOS")
    print("=" * 62)

    artefatos = {
        "Dados": [
            ("data/solar_wind_dataset.csv", "Dataset principal (≥1000 linhas)"),
            ("data/X_train.npy",             "Arrays de treino"),
            ("data/X_test.npy",              "Arrays de teste"),
            ("data/feature_cols.pkl",        "Lista de features"),
        ],
        "Modelos": [
            ("models/scaler.pkl",             "RobustScaler treinado"),
            ("models/best_regressor.pkl",     "Melhor modelo de regressão"),
            ("models/best_classifier.pkl",    "Melhor modelo de classificação"),
        ],
        "Outputs — EDA": [
            ("outputs/eda/eda_solar.png",                       "Análise exploratória (EDA)"),
        ],
        "Outputs — Modelos": [
            ("outputs/modelos/comparacao_regressao.png",        "Comparação modelos regressão"),
            ("outputs/modelos/comparacao_classificacao.png",    "Comparação modelos classificação"),
            ("outputs/modelos/metricas.json",                   "Métricas completas JSON"),
        ],
        "Outputs — SHAP": [
            ("outputs/shap/summary_regressao.png",              "Summary plot regressão"),
            ("outputs/shap/bar_regressao.png",                  "Bar plot regressão"),
            ("outputs/shap/dependence_bz_regressao.png",        "Dependence plot Bz"),
            ("outputs/shap/waterfall_extremo_regressao.png",    "Waterfall caso extremo"),
            ("outputs/shap/summary_classificacao.png",          "Summary plot classificação"),
            ("outputs/shap/bar_classificacao.png",              "Bar plot classificação"),
            ("outputs/shap/interpretacao_shap.txt",             "Relatório SHAP textual"),
        ],
    }

    total = 0
    ok = 0
    for categoria, itens in artefatos.items():
        print(f"\n  📁 {categoria}:")
        for path, descricao in itens:
            total += 1
            existe = os.path.exists(path)
            if existe:
                ok += 1
                tamanho = os.path.getsize(path)
                tam_str = f"{tamanho/1024:.1f} KB" if tamanho < 1024**2 else f"{tamanho/1024**2:.1f} MB"
                print(f"    ✓ {path:<48} ({tam_str})")
            else:
                print(f"    ✗ {path:<48} AUSENTE")

    print(f"\n  Total: {ok}/{total} artefatos gerados com sucesso")
    print("=" * 62)


def main():
    banner()

    # Instala dependências
    instalar_dependencias()

    # Define scripts em ordem de execução
    scripts = [
        "1_coleta_dados.py",
        "2_preprocessamento.py",
        "3_modelos.py",
        "4_shap_interpretabilidade.py",
    ]

    tempo_total_inicio = time.time()
    falhas = []

    for i, script in enumerate(scripts, 1):
        print(f"\n{'='*62}")
        print(f"  ETAPA {i}/{len(scripts)}: {script}")
        print("=" * 62)

        sucesso = executar_script(script)
        if not sucesso:
            falhas.append(script)
            print(f"\n  ⚠️ Falha em {script}. Continuando para próxima etapa...")

    tempo_total = time.time() - tempo_total_inicio

    # Verifica artefatos gerados
    verificar_artefatos()

    # Sumário final
    print(f"\n  ⏱️  Tempo total do pipeline: {tempo_total:.1f}s")

    if falhas:
        print(f"\n  ⚠️  Scripts com falha: {', '.join(falhas)}")
        print("     Verifique os logs acima para detalhes.")
    else:
        print("\n  ✅ Pipeline concluído com sucesso!")

    print("""
╔══════════════════════════════════════════════════════════════╗
║   PRÓXIMO PASSO: Iniciar a aplicação Streamlit               ║
║                                                              ║
║   streamlit run 5_app_streamlit.py                           ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
