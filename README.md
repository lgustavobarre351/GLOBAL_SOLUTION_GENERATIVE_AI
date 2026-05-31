# 🌌 GAIE — Geomagnetic AI Engine

> **Camada preditiva de Machine Learning integrada ao [HELIOS Space Intelligence Platform](https://helius-zeta.vercel.app/)**
> Previsão em tempo real da intensidade de tempestades geomagnéticas usando dados reais do vento solar

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.42+-green.svg)](https://shap.readthedocs.io)
[![FIAP](https://img.shields.io/badge/FIAP-Global%20Solution%202026-E85A1E.svg)](#)

---

## 🔗 Links de Entrega

| | Link |
|---|---|
| **Repositório GitHub** | https://github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI |
| **Aplicação em funcionamento** | https://globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app |

---

## 👥 Integrantes — Mission Crew

> **FIAP · Global Solution 2026/1 · Generative AI for Engineering · Turma 4ESPY**

| Nome | RM |
|------|----|
| Julia Azevedo Lins | RM99690 |
| Luis Gustavo Barreto Garrido | RM99210 |
| Victor Hugo Aranda Forte | RM99667 |
| Guilherme Akio | RM98582 |
| Felipe Cortez | RM99750 |

---

## 1. Contexto do Problema

Tempestades geomagnéticas são distúrbios no campo magnético da Terra causados pela interação do vento solar com a magnetosfera. Quando o campo magnético interplanetário aponta para o sul (Bz negativo), ocorre reconexão magnética na magnetopausa, injetando partículas energéticas no anel de corrente terrestre e gerando perturbações que afetam satélites, redes elétricas, GPS e comunicações de rádio.

| Evento Histórico | Data | Impacto Econômico |
|-----------------|------|-------------------|
| Grande Apagão de Quebec | Março 1989 | 6 milhões sem energia por 9h — US$ 2 bilhões |
| Halloween Storms | Outubro 2003 | 30 satélites danificados, blackout HF global |
| Tempestade de Carrington | Setembro 1859 | Repetição hoje: US$ 0,6–2,6 trilhões (Lloyd's 2013) |

O **KP Index** (escala 0–9) é o índice geomagnético planetário principal. Valores ≥ 5 caracterizam uma **tempestade G1–G5**. Prever sua intensidade com antecedência é crítico para:
- Proteção de satélites e sistemas GPS
- Gestão preventiva de redes elétricas
- Segurança de astronautas em missão
- Aviação em rotas polares

O GAIE é a camada preditiva de ML integrada ao **HELIOS Space Intelligence Platform** — que já monitora dados em tempo real da NASA DONKI e NOAA SWPC — adicionando capacidade de previsão à plataforma.

---

## 2. Fonte dos Dados

Os dados foram obtidos de **APIs públicas em tempo real**, sem custo de acesso:

| Fonte | Endpoint | Dados | Resolução |
|-------|----------|-------|-----------|
| **NOAA SWPC Solar Wind MAG** | `services.swpc.noaa.gov/products/solar-wind/mag-7-day.json` | Campo Bz, Bt, Bx, By (nT) | 1 minuto |
| **NOAA SWPC Solar Wind Plasma** | `services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json` | Velocidade (km/s), densidade (p/cc), temperatura (eV) | 1 minuto |
| **NOAA SWPC KP Index** | `services.swpc.noaa.gov/products/noaa-planetary-k-index.json` | Índice geomagnético planetário | 3 horas |
| **NASA DONKI API** | `api.nasa.gov/DONKI/FLR` e `/GST` | Eventos de flare e tempestades | Tempo real |

### Composição do Dataset Final

| Fonte | Linhas | Tipo |
|-------|--------|------|
| NOAA SWPC (MAG + Plasma + KP) | 9.749 | Dados reais do satélite DSCOVR |
| Suplemento sintético | 1.500 | Distribuições baseadas em Borovsky 2006 e Newell 2008 |
| **Total** | **11.249 linhas × 13 colunas** | Acima do mínimo exigido (1.000 × 10) |

> O suplemento sintético foi necessário porque a última semana registrou KP máximo de 3,7 (sem tempestades). Eventos raros como G3–G5 precisam estar representados no treino.

---

## 3. Pré-processamento e Engenharia de Atributos

### 3.1 Limpeza dos Dados

- Remoção de duplicatas
- Tratamento de outliers com **limites físicos reais** (ex: Bz entre -80 e +30 nT; velocidade entre 200 e 1.200 km/s)
- Merge temporal entre MAG, Plasma e KP usando `merge_asof` com tolerância de 2 minutos
- Forward-fill do KP Index (3h → 1min)

### 3.2 Engenharia de Atributos

20 features foram criadas a partir das 13 colunas brutas, cada uma fundamentada em física do plasma:

| Feature | Fórmula | Justificativa Física |
|---------|---------|----------------------|
| `bz_negativo` | `max(-Bz, 0)` | Componente sul do IMF — único driver de reconexão magnética |
| `newell_coupling` | `v^(4/3) × Bs^(2/3)` | Taxa de transferência de energia (Newell 2008) |
| `cme_bz_interacao` | `CME × Bz_sul` | Sinergia CME + campo sul — condição mais geoefetiva |
| `pressao_dinamica` | `0.5 × ρ × v²` | Compressão da magnetosfera |
| `bz_media_3h` | Média móvel 3h de Bz | Pré-condicionamento geomagnético de curto prazo |
| `bz_media_6h` | Média móvel 6h de Bz | Pré-condicionamento geomagnético de médio prazo |
| `velocidade_media_3h` | Média móvel 3h da velocidade | Tendência recente do vento solar |
| `velocidade_media_6h` | Média móvel 6h da velocidade | Tendência de longo prazo do vento solar |
| `hora_sin` / `hora_cos` | `sin/cos(2π × h / 24)` | Variação diurna cíclica do campo geomagnético |
| `mes_sin` / `mes_cos` | `sin/cos(2π × m / 12)` | Efeito Russell-McPherron (equinócios mais geoefetivos) |

### 3.3 Divisão e Escalonamento

| Conjunto | Proporção | Amostras |
|----------|-----------|---------|
| Treino | 70% | 7.874 |
| Validação | 15% | 1.687 |
| Teste | 15% | 1.688 |

- Divisão **estratificada por nível G** para garantir representação de tempestades em todos os conjuntos
- Escalonamento com **RobustScaler** — robusto a outliers de eventos extremos (usa mediana e IQR em vez de média e desvio padrão)

---

## 4. Modelos Testados e Comparação

O problema foi abordado em duas frentes: **regressão** (valor contínuo do KP) e **classificação** (nível categórico G0–G5).

### 4.1 Modelos de Regressão (alvo: KP Index 0–9)

| Modelo | Descrição |
|--------|-----------|
| **Random Forest** | Ensemble de 200 árvores, max_depth=12, bagging |
| **XGBoost** | Gradient boosting, 200 estimadores, learning_rate=0.05 |
| **Ridge** | Regressão linear com regularização L2 — baseline |

### 4.2 Modelos de Classificação (alvo: Nível G0–G5)

| Modelo | Descrição |
|--------|-----------|
| **Random Forest** | Ensemble de 200 árvores, class_weight='balanced' |
| **XGBoost** | Gradient boosting multi-classe |
| **Logistic Regression** | Regressão logística multi-classe — baseline |

---

## 5. Resultados Obtidos

> Métricas calculadas no conjunto de teste (1.688 amostras nunca vistas durante o treino).

### 5.1 Regressão — KP Index

| Modelo | RMSE ↓ | MAE ↓ | R² ↑ |
|--------|--------|-------|------|
| Random Forest | 0.2814 | 0.1057 | 0.9668 |
| **XGBoost** ⭐ | **0.2768** | **0.1704** | **0.9678** |
| Ridge (baseline) | 0.6473 | 0.5292 | 0.8241 |

### 5.2 Classificação — Nível G

| Modelo | Accuracy ↑ | F1-weighted ↑ |
|--------|-----------|---------------|
| Random Forest | 0.9757 | 0.9759 |
| **XGBoost** ⭐ | **0.9787** | **0.9772** |
| Logistic Reg. (baseline) | 0.9710 | 0.9741 |

### 5.3 Análise dos Resultados

O **XGBoost** venceu em ambos os problemas. O R² de 0,97 significa que o modelo explica **97% da variância** do índice geomagnético — resultado excepcional para um fenômeno geofísico naturalmente estocástico.

A superioridade sobre o modelo linear (Ridge R²=0,82) confirma que as relações entre vento solar e atividade geomagnética são **fundamentalmente não-lineares**, algo que o gradient boosting captura melhor que modelos lineares.

---

## 6. Interpretabilidade com SHAP

A análise SHAP (SHapley Additive exPlanations) foi aplicada com **TreeExplainer** para quantificar a contribuição de cada variável em cada previsão individual.

### Top 5 Variáveis Mais Importantes

1. **`bz_negativo`** — O campo Bz sul é o maior driver. Bz < -10 nT quase sempre precede tempestades G2+. A reconexão magnética na magnetopausa injeta partículas no anel de corrente terrestre.

2. **`newell_coupling`** — A função ε = v^(4/3) × Bs^(2/3) integra velocidade e Bz de forma não-linear, capturando a eficiência de transferência de energia melhor do que as variáveis isoladas.

3. **`cme`** — Presença de CME adiciona ~1–2 pontos ao KP. CMEs são a principal causa de tempestades G3–G5 históricas, transportando plasma denso e campos magnéticos intensos.

4. **`velocidade_vento`** — Maior velocidade = maior energia cinética = maior compressão da magnetosfera. Fundamental em tempestades tipo CIR (Corotating Interaction Region).

5. **`pressao_dinamica`** — P = ρv²/2 comprime a magnetosfera e reduz o raio de Alfvén, amplificando os efeitos mesmo quando Bz é moderado.

### Plots Gerados

| Plot | Descrição |
|------|-----------|
| Summary Plot (beeswarm) | Distribuição do impacto de cada feature em todas as amostras |
| Bar Plot | Importância média absoluta — ranking global das features |
| Dependence Plot (Bz) | Como o impacto do Bz varia com a velocidade do vento solar |
| Waterfall Plot | Explicação detalhada do caso extremo do dataset |

---

## 7. Deploy da Aplicação

A aplicação foi deployada no **Streamlit Cloud** e está acessível publicamente:

🔗 **https://globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app**

### Funcionalidades

- **Previsão em tempo real** — sliders para simular parâmetros do vento solar
- **Gauge do KP Index** — visualização em velocímetro com escala G0–G5
- **Probabilidades por classe** — distribuição de probabilidade do classificador
- **Fatores de risco** — gráfico com intensidade de cada variável
- **Aba SHAP** — plots de interpretabilidade integrados
- **Aba Métricas** — comparação completa dos modelos com gráficos
- **Design HELIOS** — tema espacial escuro integrado à identidade visual da plataforma

---

## 8. Instruções para Execução

### Pré-requisitos
- Python 3.10+
- pip

### Passo a Passo

```bash
# 1. Clonar o repositório
git clone https://github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI.git
cd GLOBAL_SOLUTION_GENERATIVE_AI/gaie_helios

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar o pipeline completo (coleta → pré-processamento → modelos → SHAP)
python 0_pipeline.py

# 4. Iniciar a aplicação web
streamlit run 5_app_streamlit.py
```

A aplicação abrirá automaticamente em `http://localhost:8501`.

### Execução Individual por Etapa

```bash
python 1_coleta_dados.py          # Coleta NOAA SWPC + NASA DONKI
python 2_preprocessamento.py      # Limpeza + features + EDA
python 3_modelos.py               # Treinamento + comparação
python 4_shap_interpretabilidade.py  # Análise SHAP
```

### Variável de Ambiente da NASA API

```bash
# Windows PowerShell
$env:NASA_API_KEY = "sua_chave_aqui"

# Linux / macOS
export NASA_API_KEY="sua_chave_aqui"
```

> A chave gratuita pode ser obtida em: https://api.nasa.gov/

---

## 9. Estrutura do Projeto

```
gaie_helios/
├── 0_pipeline.py                    # Orquestrador — executa tudo em sequência
├── 1_coleta_dados.py                # Coleta NOAA SWPC + NASA DONKI + sintético
├── 2_preprocessamento.py            # Limpeza + 20 features + EDA + split + scaler
├── 3_modelos.py                     # Treino, validação e comparação de 6 modelos
├── 4_shap_interpretabilidade.py     # SHAP TreeExplainer + plots + relatório físico
├── 5_app_streamlit.py               # Aplicação web interativa (deploy)
├── requirements.txt                 # Dependências Python
├── README.md                        # Esta documentação
├── data/
│   ├── solar_wind_dataset.csv       # Dataset (11.249 linhas × 13 colunas)
│   ├── X_{train,val,test}.npy       # Arrays pré-processados e escalonados
│   └── feature_cols.pkl             # Lista das 20 features dos modelos
├── models/
│   ├── best_regressor.pkl           # XGBoost regressão (melhor RMSE)
│   ├── best_classifier.pkl          # XGBoost classificação (melhor F1)
│   └── scaler.pkl                   # RobustScaler treinado
├── outputs/
│   ├── eda_solar.png                # Painel EDA (9 gráficos)
│   ├── comparacao_regressao.png     # Comparação RMSE / MAE / R²
│   ├── comparacao_classificacao.png # Comparação Accuracy / F1 / Conf. Matrix
│   ├── metricas.json                # Métricas completas em JSON
│   └── interpretacao_shap.txt       # Relatório físico das variáveis SHAP
└── shap_plots/
    ├── summary_regressao.png        # Beeswarm — regressão
    ├── bar_regressao.png            # Importância média — regressão
    ├── dependence_bz_regressao.png  # Dependence plot do campo Bz
    ├── waterfall_extremo_regressao.png  # Waterfall — caso extremo
    ├── summary_classificacao.png    # Beeswarm — classificação
    └── bar_classificacao.png        # Importância média — classificação
```

---

## 10. Referências Científicas

1. **Newell, P.T. et al. (2008)**. A solar wind magnetosphere coupling function controlling multitude of geophysical phenomena. *Journal of Geophysical Research*, 113, A09218. doi:10.1029/2007JA012825

2. **Borovsky, J.E. & Denton, M.H. (2006)**. Differences between CME-driven storms and CIR-driven storms. *Journal of Geophysical Research*, 111, A07S08. doi:10.1029/2005JA011447

3. **Richardson, I.G. & Cane, H.V. (2012)**. Near-Earth solar wind magnetic fields, plasma and energetic particle data during more than four solar cycles. *Journal of Geophysical Research*, 117, A08110. doi:10.1029/2011JA017364

4. **Lloyd's of London (2013)**. *Solar Storm Risk to the North American Electric Grid*. Lloyd's of London Risk Report.

---

*Projeto desenvolvido para a disciplina Generative AI for Engineering — FIAP 2026/1*
*Dados fornecidos por NASA/GSFC (DONKI API) e NOAA/SWPC (Solar Wind Monitor)*
