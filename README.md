# 🌌 GAIE — Geomagnetic AI Engine

> **Camada preditiva de Machine Learning integrada ao [HELIOS Space Intelligence Platform](https://helius-zeta.vercel.app/)**  
> Previsão em tempo real da intensidade de tempestades geomagnéticas usando dados do vento solar

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.42+-green.svg)](https://shap.readthedocs.io)

## 🔗 Links de Entrega

| | Link |
|---|---|
| **Repositório GitHub** | https://github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI |
| **Aplicação em funcionamento** | https://globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app |

---

## 👥 Mission Crew — Integrantes

> **FIAP · Global Solution 2026/1 · Generative AI for Engineering · Turma 4ESPY**

| Nome | RM |
|------|----|
| Julia Azevedo Lins | RM99690 |
| Luis Gustavo Barreto Garrido | RM99210 |
| Victor Hugo Aranda Forte | RM99667 |
| Guilherme Akio | RM98582 |
| Felipe Cortez | RM99750 |

---

## 🎯 Contexto do Problema

Tempestades geomagnéticas são distúrbios no campo magnético da Terra causados pela interação do vento solar com a magnetosfera. Seus impactos são devastadores:

| Evento | Data | Impacto |
|--------|------|---------|
| Quebec Blackout | Março 1989 | 6 milhões sem energia por 9h, prejuízo ~$2 bilhões |
| Halloween Storms | Outubro 2003 | Falha em 30 satélites, blackout HF global |
| Tempestade de Carrington | Setembro 1859 | Se repetisse hoje: até US$2,6 trilhões (Lloyd's 2013) |

O **KP Index** (0–9) é o índice geomagnético planetário principal. Quando ≥5, uma **tempestade G1–G5** está em andamento. Prever sua intensidade com horas de antecedência é crucial para:
- Proteção de satélites e sistemas GPS
- Gestão de redes elétricas
- Segurança de astronautas
- Aviação em rotas polares

---

## 📡 Fonte dos Dados

| Fonte | Tipo | URL |
|-------|------|-----|
| **NASA DONKI API** | Eventos CME e tempestades GST em tempo real | `https://api.nasa.gov/DONKI` |
| **NOAA SWPC** | KP Index histórico (endpoint público) | `https://services.swpc.noaa.gov` |
| **Dataset Sintético** | 1.300 amostras com distribuições físicas reais | Gerado localmente |

O dataset sintético é baseado em distribuições estatísticas do vento solar documentadas em:
- **Borovsky & Denton (2006)**: diferenças entre tempestades CME e CIR
- **Newell et al. (2008)**: função de acoplamento vento solar–magnetosfera

---

## 🔬 Metodologia

### Feature Engineering

| Feature | Fórmula / Descrição | Importância Física |
|---------|---------------------|--------------------|
| `bz_negativo` | `max(-Bz, 0)` | Componente sul do IMF — driver de reconexão magnética |
| `newell_coupling` | `v^(4/3) × Bs^(2/3)` | Taxa de transferência de energia (Newell 2008) |
| `cme_bz_interacao` | `CME × Bz_sul` | Sinergia entre CME e campo sul — mais geoefetivo |
| `pressao_dinamica` | `0.5 × ρ × v²` | Compressão da magnetosfera |
| `bz_media_3h/6h` | Médias móveis | Pré-condicionamento geomagnético |
| `hora_sin/cos` | `sin/cos(2π×h/24)` | Variação diurna cíclica |
| `mes_sin/cos` | `sin/cos(2π×m/12)` | Efeito Russell-McPherron (equinócios) |

**Total: 20 features** após engenharia de atributos.

### Modelos Testados

#### Regressão (KP Index contínuo, 0–9)
| Modelo | Tipo |
|--------|------|
| **Random Forest** | Ensemble de árvores de decisão |
| **XGBoost** | Gradient boosting otimizado |
| **Ridge** | Regressão linear regularizada (baseline) |

#### Classificação (Nível G0–G5)
| Modelo | Tipo |
|--------|------|
| **Random Forest** | Ensemble com class_weight='balanced' |
| **XGBoost** | Gradient boosting multi-classe |
| **Logistic Regression** | Modelo linear (baseline) |

---

## 📊 Resultados Obtidos

### Regressão — KP Index

| Modelo | RMSE ↓ | MAE ↓ | R² ↑ |
|--------|--------|-------|------|
| Random Forest | ~0.42 | ~0.31 | ~0.89 |
| **XGBoost** | **~0.38** | **~0.28** | **~0.91** |
| Ridge (baseline) | ~1.15 | ~0.88 | ~0.51 |

### Classificação — Nível G

| Modelo | Accuracy ↑ | F1-weighted ↑ |
|--------|-----------|---------------|
| Random Forest | ~0.88 | ~0.87 |
| **XGBoost** | **~0.90** | **~0.89** |
| Logistic Reg. | ~0.71 | ~0.69 |

> *Valores aproximados — execute o pipeline para obter métricas exatas no seu ambiente.*

---

## 🔍 Interpretação com SHAP

A análise SHAP revela a física por trás das previsões:

### Top 5 Features Mais Importantes (Regressão)

1. **`bz_negativo`** — O campo Bz sul é o maior driver. Bz < -10 nT quase sempre precede tempestades G2+. A reconexão magnética na magnetopausa inicia quando Bz aponta para o sul.

2. **`newell_coupling`** — Função ε = v^(4/3) × Bs^(2/3) integra velocidade e Bz de forma não-linear, capturando melhor a eficiência de transferência de energia do que as variáveis isoladas.

3. **`cme`** — Presença de CME adiciona ~1–2 pontos ao KP. CMEs são a principal causa de tempestades G3-G5 porque transportam campos magnéticos intensos e plasma denso.

4. **`velocidade_vento`** — Maior velocidade = maior energia cinética = maior compressão da magnetosfera. Fundamental em tempestades tipo CIR (Corotating Interaction Region).

5. **`pressao_dinamica`** — P = ρv²/2 comprime a magnetosfera e reduz o raio de Alfvén, amplificando os efeitos mesmo quando Bz é moderado.

---

## ⚙️ Instruções de Execução

### Pré-requisitos
- Python 3.10+
- pip

### Instalação e Execução

```bash
# 1. Clonar o repositório
git clone <url-do-repositorio>
cd gaie_helios

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar o pipeline completo (todos os scripts em sequência)
python 0_pipeline.py

# 4. Iniciar a aplicação web
streamlit run 5_app_streamlit.py
```

### Execução Individual de Scripts

```bash
# Apenas coleta de dados
python 1_coleta_dados.py

# Apenas pré-processamento + EDA
python 2_preprocessamento.py

# Apenas treinamento dos modelos
python 3_modelos.py

# Apenas análise SHAP
python 4_shap_interpretabilidade.py
```

### Variável de Ambiente (opcional)

```bash
# Windows PowerShell
$env:NASA_API_KEY = "sua_chave_aqui"

# Linux/macOS
export NASA_API_KEY="sua_chave_aqui"
```

---

## 🗂️ Estrutura do Projeto

```
gaie_helios/
├── 0_pipeline.py                   # Orquestrador — executa tudo em sequência
├── 1_coleta_dados.py               # Coleta NASA DONKI + NOAA + geração sintética
├── 2_preprocessamento.py           # Limpeza + feature engineering + EDA
├── 3_modelos.py                    # Treino, validação e comparação de modelos
├── 4_shap_interpretabilidade.py    # Análise SHAP com plots e relatório físico
├── 5_app_streamlit.py              # Aplicação web interativa
├── requirements.txt                # Dependências Python
├── README.md                       # Esta documentação
├── data/
│   ├── solar_wind_dataset.csv      # Dataset principal (1300+ linhas, 13 colunas)
│   ├── X_{train,val,test}.npy      # Arrays pré-processados
│   └── feature_cols.pkl            # Lista de features dos modelos
├── models/
│   ├── best_regressor.pkl          # Melhor modelo de regressão
│   ├── best_classifier.pkl         # Melhor modelo de classificação
│   └── scaler.pkl                  # RobustScaler treinado
├── outputs/
│   ├── eda_solar.png               # Painel EDA completo
│   ├── comparacao_regressao.png    # Comparação RMSE/MAE/R²
│   ├── comparacao_classificacao.png # Comparação Accuracy/F1/ConfMatrix
│   ├── metricas.json               # Métricas completas em JSON
│   └── interpretacao_shap.txt      # Relatório físico SHAP
└── shap_plots/
    ├── summary_regressao.png       # Beeswarm plot — regressão
    ├── bar_regressao.png           # Importância média — regressão
    ├── dependence_bz_regressao.png # Dependence plot do campo Bz
    ├── waterfall_extremo_regressao.png # Waterfall de caso extremo
    ├── summary_classificacao.png   # Beeswarm plot — classificação
    └── bar_classificacao.png       # Importância média — classificação
```

---

## 🔗 Links

- **Aplicação em funcionamento**: [GAIE — Previsão de Tempestades Geomagnéticas · Streamlit](https://globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app/)
- **Plataforma HELIOS**: [https://helius-zeta.vercel.app/](https://helius-zeta.vercel.app/)
- **NASA DONKI API**: [https://ccmc.gsfc.nasa.gov/tools/DONKI/](https://ccmc.gsfc.nasa.gov/tools/DONKI/)
- **NOAA SWPC**: [https://www.swpc.noaa.gov/](https://www.swpc.noaa.gov/)

---

## 📚 Referências Científicas

1. **Newell, P.T. et al. (2008)**. A solar wind magnetosphere coupling function controlling multitude of geophysical phenomena. *Journal of Geophysical Research*, 113, A09218. doi:10.1029/2007JA012825

2. **Borovsky, J.E. & Denton, M.H. (2006)**. Differences between CME-driven storms and CIR-driven storms. *Journal of Geophysical Research*, 111, A07S08. doi:10.1029/2005JA011447

3. **Richardson, I.G. & Cane, H.V. (2012)**. Near-Earth solar wind magnetic fields, plasma and energetic particle data during more than four solar cycles (1963–2011). *Journal of Geophysical Research*, 117, A08110. doi:10.1029/2011JA017364

4. **Lloyd's of London (2013)**. *Solar Storm Risk to the North American Electric Grid*. Lloyd's of London Risk Report.

---

*Projeto desenvolvido para a disciplina GAIE — Generative AI for Engineering*  
*Dados do clima espacial fornecidos por NASA/GSFC e NOAA/SWPC*
