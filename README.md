# 🌌 HELIOS — Space Intelligence Platform

> **Plataforma web de monitoramento de clima espacial em tempo real, integrada a satélites governamentais da NASA e NOAA**
> Global Solution 2026/1 · FIAP · Generative AI for Engineering · Turma 4ESPY

[![HELIOS](https://img.shields.io/badge/HELIOS-Space%20Intelligence-E85A1E.svg)](https://helius-zeta.vercel.app/)
[![FIAP](https://img.shields.io/badge/FIAP-Global%20Solution%202026-1B3A6B.svg)](#)

---

## 🚀 O que é o HELIOS

O **HELIOS** é uma plataforma web de monitoramento de clima espacial em tempo real. Ela integra dados de satélites governamentais da NASA e NOAA e os apresenta em uma interface de dashboard modular, cobrindo cinco frentes:

| Módulo | O que faz |
|--------|-----------|
| 🛸 **Agenda de Lançamentos** | Acompanhamento de lançamentos orbitais em tempo real |
| ☀️ **Eventos Solares** | Monitoramento de flares, CMEs e tempestades geomagnéticas via NASA DONKI |
| 🛰️ **Rastreamento de Satélites** | Posição de satélites em órbita em tempo real |
| 🤖 **Previsão por IA** | Forecast de clima espacial por Machine Learning — **componente GAIE** |
| ⚡ **Energia Solar** | Otimização de potencial fotovoltaico por região no território brasileiro |

A proposta do projeto é tornar acessível o que antes exigia sistemas técnicos especializados — condições do vento solar, alertas de flares, posição de satélites, potencial fotovoltaico por região — reunindo tudo em uma única interface, com dados atualizados automaticamente e sem necessidade de conhecimento técnico prévio para interpretar os resultados.

🔗 **Plataforma HELIOS:** https://helius-zeta.vercel.app/

---

## 🤖 GAIE — O Componente de IA do HELIOS

O **GAIE (Geomagnetic AI Engine)** é a camada preditiva de Machine Learning do HELIOS. Enquanto os demais módulos da plataforma monitoram e exibem o que está acontecendo agora, o GAIE adiciona a capacidade de **prever o que vai acontecer nas próximas horas** — transformando o HELIOS de um sistema reativo em um sistema preditivo.

Especificamente, o GAIE resolve o problema de previsão de **tempestades geomagnéticas**: dado o estado atual do vento solar (medido pelo satélite DSCOVR no ponto L1, a 1,5 milhão de km da Terra), qual será a intensidade da perturbação geomagnética nos próximos momentos? A resposta é dada em duas formas complementares: o **KP Index** (valor contínuo 0–9) e o **Nível G** (classificação G0–G5 da escala oficial NOAA).

```
HELIOS Platform
├── Lançamentos Orbitais
├── Eventos Solares (NASA DONKI)
├── Rastreamento de Satélites
├── Energia Solar Brasil
└── 🤖 GAIE — Previsão por IA  ← este repositório
    ├── Monitoramento ao vivo (dados reais NOAA)
    ├── Forecast 48h com bandas de incerteza
    └── Simulação manual de cenários
```

---

## 🔗 Links de Entrega

| | Link |
|---|---|
| 📁 **Repositório GitHub (GAIE)** | https://github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI |
| 🤖 **Aplicação GAIE (Streamlit)** | https://globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app |
| 🌌 **Plataforma HELIOS** | https://helius-zeta.vercel.app/ |

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app)
[![XGBoost](https://img.shields.io/badge/XGBoost-R²_0.97-orange.svg)](#)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-green.svg)](#)

---

## 👥 Equipe — Mission Crew

**FIAP · Global Solution 2026/1 · Generative AI for Engineering · Turma 4ESPY**

| Nome | RM |
|------|----|
| Julia Azevedo Lins | RM98690 |
| Luis Gustavo Barreto Garrido | RM99210 |
| Victor Hugo Aranda Forte | RM99667 |
| Guilherme Akio | RM98582 |
| Felipe Cortez | RM99750 |

---

## ✅ Checklist de Critérios de Avaliação

> Guia rápido para o avaliador localizar cada critério no README e no código.

| Critério | Peso | Seção no README | Script |
|----------|------|-----------------|--------|
| Definição do problema e qualidade dos dados | 15 pts | [Seções 1 e 2](#1-contexto-do-problema--conexão-com-a-economia-espacial) | `1_coleta_dados.py` |
| Pré-processamento e engenharia de atributos | 20 pts | [Seção 3](#3-pré-processamento-e-engenharia-de-atributos) | `2_preprocessamento.py` |
| Aplicação e comparação de modelos | 20 pts | [Seção 4](#4-modelos-aplicados-e-comparação) | `3_modelos.py` |
| Validação e análise de métricas | 15 pts | [Seção 5](#5-resultados-e-validação) | `3_modelos.py` |
| Interpretabilidade com SHAP | 10 pts | [Seção 6](#6-interpretabilidade-com-shap) | `4_shap_interpretabilidade.py` |
| Deploy da aplicação | 10 pts | [Seção 7](#7-deploy-da-aplicação) | `5_app_streamlit.py` |
| Organização do código e README no GitHub | 10 pts | [Seção 9](#9-estrutura-do-projeto) | Todos os scripts |

---

## 1. Contexto do Problema — Conexão com a Economia Espacial

### O Problema

Tempestades geomagnéticas são distúrbios no campo magnético da Terra causados pela interação do vento solar com a magnetosfera. São um risco direto à **infraestrutura crítica da economia espacial**: satélites de comunicação, sistemas GPS, redes de energia elétrica, aviação polar e operações de astronautas.

| Evento Histórico | Data | Impacto Econômico |
|-----------------|------|-------------------|
| Grande Apagão de Quebec | Março 1989 | 6 milhões sem energia por 9h — US$ 2 bilhões em danos |
| Halloween Storms | Outubro 2003 | 30 satélites danificados, blackout de rádio HF global |
| Tempestade de Carrington | Setembro 1859 | Repetição hoje custaria US$ 0,6–2,6 trilhões (Lloyd's 2013) |

### Conexão com a Economia Espacial

O GAIE responde diretamente ao desafio proposto pela FIAP: **usar dados orbitais e tecnologia espacial para resolver um problema real**. O sistema:

- Consome **dados em tempo real de satélites governamentais** (DSCOVR da NOAA, ACE da NASA)
- Protege **infraestrutura espacial e terrestre** dependente de clima espacial estável
- Usa **APIs da NASA** (`api.nasa.gov/DONKI`) e **NOAA SWPC** como fontes primárias
- Integra-se ao **HELIOS Space Intelligence Platform** — plataforma de monitoramento espacial já em produção

### Alinhamento com ODS

| ODS | Conexão com o GAIE |
|-----|-------------------|
| **ODS 9** — Indústria, Inovação e Infraestrutura | Protege redes elétricas e satélites contra apagões geomagnéticos |
| **ODS 13** — Ação Climática | Clima espacial é uma ameaça ambiental global com impacto mensurável |
| **ODS 11** — Cidades Sustentáveis | Previne colapso de infraestrutura urbana crítica em eventos extremos |

### Quem sofre sem esse sistema?

- **Operadores de satélites** — um único satélite custa US$ 50–500 milhões; tempestades G4–G5 podem danificá-los permanentemente
- **Gestores de redes elétricas** — sem aviso, as correntes induzidas queimam transformadores (como Quebec 1989)
- **Aviação** — rotas polares perdem comunicação HF em tempestades G2+
- **Astronautas na ISS** — precisam de abrigo de radiação durante tempestades intensas

---

## 2. Fonte dos Dados

Os dados foram obtidos de **APIs públicas em tempo real** de agências governamentais, sem custo de acesso:

| Fonte | Endpoint | Dados coletados | Resolução |
|-------|----------|-----------------|-----------|
| **NOAA SWPC Solar Wind MAG** | `services.swpc.noaa.gov/products/solar-wind/mag-7-day.json` | Campo Bz, Bt, Bx, By (nT) | **1 minuto** |
| **NOAA SWPC Solar Wind Plasma** | `services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json` | Velocidade (km/s), densidade (p/cc), temperatura (eV) | **1 minuto** |
| **NOAA SWPC KP Index** | `services.swpc.noaa.gov/products/noaa-planetary-k-index.json` | Índice geomagnético planetário | **3 horas** |
| **NASA DONKI FLR** | `api.nasa.gov/DONKI/FLR` | Eventos de flare solar (classe B/C/M/X) | Tempo real |
| **NASA DONKI GST** | `api.nasa.gov/DONKI/GST` | Tempestades geomagnéticas confirmadas | Tempo real |

### Composição do Dataset Final

| Fonte | Linhas | Colunas | Observação |
|-------|--------|---------|------------|
| NOAA SWPC (dados reais do satélite DSCOVR) | 9.749 | 13 | Semana de 21–27/05/2026 |
| Suplemento sintético (distribuições físicas reais) | 1.500 | 13 | Necessário: semana foi quieta (KP máx 3,7) |
| **TOTAL** | **11.249** | **13** | **Mínimo exigido: 1.000 × 10 ✅** |

> **Por que dados sintéticos?** O período de coleta coincidiu com semana de baixa atividade solar (KP máximo 3,7 — sem nenhuma tempestade formal). Para que os modelos aprendam a identificar tempestades G3–G5, essas condições precisam estar representadas no treino. O suplemento foi gerado com distribuições estatísticas baseadas em Borovsky & Denton (2006) e Newell et al. (2008), ambos trabalhos científicos de referência em física do plasma solar.

---

## 3. Pré-processamento e Engenharia de Atributos

**Script:** `2_preprocessamento.py`

### 3.1 Limpeza dos Dados

- Remoção de duplicatas exatas
- Tratamento de outliers com **limites físicos reais** documentados na literatura (ex: Bz entre −80 e +30 nT; velocidade entre 200 e 1.200 km/s)
- Merge temporal entre MAG, Plasma e KP Index usando `pandas.merge_asof` com tolerância de 2 minutos
- Forward-fill do KP Index (resolução 3h → granularidade 1min)
- Parse de timestamps com formato misto (dados reais têm milissegundos, sintéticos não)

### 3.2 Engenharia de Atributos — 20 Features

A partir das 13 colunas brutas, foram criadas 20 features com justificativa física para cada uma:

| Feature | Fórmula | Justificativa Física |
|---------|---------|----------------------|
| `bz_negativo` | `max(-Bz, 0)` | Componente sul do IMF — único vetor de reconexão magnética |
| `newell_coupling` | `v^(4/3) × Bs^(2/3)` | Função de acoplamento de Newell (2008) — taxa de transferência de energia |
| `cme_bz_interacao` | `CME × Bz_sul` | Sinergia: CME com campo sul é a condição mais geoefetiva |
| `pressao_dinamica` | `0.5 × ρ × v²` | Compressão da magnetosfera |
| `bz_media_3h` | Média móvel 3h | Pré-condicionamento geomagnético de curto prazo |
| `bz_media_6h` | Média móvel 6h | Pré-condicionamento de médio prazo |
| `velocidade_media_3h` | Média móvel 3h | Tendência recente da velocidade |
| `velocidade_media_6h` | Média móvel 6h | Tendência de longo prazo |
| `hora_sin` / `hora_cos` | `sin/cos(2π × h/24)` | Variação diurna cíclica (preserva continuidade 23h→0h) |
| `mes_sin` / `mes_cos` | `sin/cos(2π × m/12)` | Efeito Russell-McPherron (equinócios mais suscetíveis) |

### 3.3 Divisão e Escalonamento

| Conjunto | Proporção | Amostras | Finalidade |
|----------|-----------|---------|------------|
| Treino | 70% | 7.874 | Aprendizado dos modelos |
| Validação | 15% | 1.687 | Ajuste e seleção de hiperparâmetros |
| Teste | 15% | 1.688 | Avaliação final imparcial |

- Divisão **estratificada por nível G** — garante presença de tempestades raras (G3–G5) nos três conjuntos
- Escalonamento com **RobustScaler** — usa mediana e IQR (robusto a outliers de eventos extremos, ao contrário do StandardScaler)
- Dados de teste **nunca vistos** durante treino ou validação

---

## 4. Modelos Aplicados e Comparação

**Script:** `3_modelos.py`

O problema foi abordado em **duas frentes**: regressão (valor contínuo do KP) e classificação (nível categórico G0–G5). Para cada frente, três algoritmos foram treinados e comparados, incluindo um modelo linear como baseline.

### 4.1 Regressão — prever KP Index (0–9, contínuo)

| Modelo | Tipo | Hiperparâmetros principais |
|--------|------|---------------------------|
| **Random Forest** | Ensemble (bagging) | 200 árvores, max_depth=12 |
| **XGBoost** | Gradient Boosting | 200 estimadores, learning_rate=0.05, subsample=0.8 |
| **Ridge** | Linear (baseline) | alpha=1.0 |

### 4.2 Classificação — prever Nível G (G0 a G5, categórico)

| Modelo | Tipo | Hiperparâmetros principais |
|--------|------|---------------------------|
| **Random Forest** | Ensemble (bagging) | 200 árvores, class_weight='balanced' |
| **XGBoost** | Gradient Boosting | 200 estimadores, multi-classe, eval_metric='mlogloss' |
| **Logistic Regression** | Linear (baseline) | C=1.0, class_weight='balanced', max_iter=2000 |

### 4.3 Critério de Seleção do Melhor Modelo

- Regressão: menor **RMSE** no conjunto de teste
- Classificação: maior **F1-weighted** no conjunto de teste (métrica robusta a desbalanceamento de classes)

---

## 5. Resultados e Validação

**Métricas calculadas no conjunto de teste — 1.688 amostras nunca vistas durante treino ou validação.**

### 5.1 Regressão — KP Index

| Modelo | RMSE ↓ | MAE ↓ | R² ↑ |
|--------|--------|-------|------|
| Random Forest | 0.2814 | 0.1057 | 0.9668 |
| ⭐ **XGBoost — MELHOR** | **0.2768** | **0.1704** | **0.9678** |
| Ridge (baseline) | 0.6473 | 0.5292 | 0.8241 |

### 5.2 Classificação — Nível G

| Modelo | Accuracy ↑ | F1-weighted ↑ |
|--------|-----------|---------------|
| Random Forest | 0.9757 | 0.9759 |
| ⭐ **XGBoost — MELHOR** | **0.9787** | **0.9772** |
| Logistic Reg. (baseline) | 0.9710 | 0.9741 |

> Matriz de confusão gerada para os três modelos de classificação — disponível em `outputs/comparacao_classificacao.png`

### 5.3 Análise e Justificativa da Escolha

O **XGBoost** foi selecionado como melhor modelo em ambos os problemas pelos seguintes motivos:

- **R² = 0,97** — explica 97% da variância do índice geomagnético. O RMSE de 0,28 em uma escala de 0–9 representa erro médio de ~3% da faixa total.
- **F1-weighted = 0,977** — alta precisão na classificação de todos os 6 níveis G, inclusive os eventos raros.
- A **superioridade sobre o modelo linear** (Ridge R²=0,82) confirma que as relações entre vento solar e atividade geomagnética são **fundamentalmente não-lineares** — o gradient boosting captura essas interações melhor que modelos lineares.
- O **Random Forest** ficou em segundo, próximo ao XGBoost, validando que modelos ensemble são a abordagem correta para este problema.

---

## 6. Interpretabilidade com SHAP

**Script:** `4_shap_interpretabilidade.py`

A análise SHAP (SHapley Additive exPlanations) foi aplicada usando **TreeExplainer** nos modelos XGBoost para quantificar a contribuição individual de cada variável em cada previsão — tornando o modelo explicável e auditável.

### Plots Gerados (disponíveis em `shap_plots/`)

| Arquivo | Tipo | O que mostra |
|---------|------|-------------|
| `summary_regressao.png` | Beeswarm | Distribuição do impacto de cada feature em todas as 1.688 amostras |
| `bar_regressao.png` | Barras | Importância média absoluta — ranking das 20 features |
| `dependence_bz_regressao.png` | Dispersão | Como o impacto do Bz varia com a velocidade do vento solar |
| `waterfall_extremo_regressao.png` | Waterfall | Decomposição da previsão para o caso de tempestade mais severa |
| `summary_classificacao.png` | Beeswarm | Impacto das features na classificação G |
| `bar_classificacao.png` | Barras | Importância média para classificação |

> Relatório textual com interpretação física: `outputs/interpretacao_shap.txt`

### Top 5 Variáveis Mais Importantes — Regressão

| Rank | Feature | Impacto SHAP | Explicação Física |
|------|---------|-------------|-------------------|
| 1 | `bz_negativo` | ★★★★★ | Componente sul do IMF. Bz < −10 nT quase sempre precede tempestades G2+. É o gatilho primário da reconexão magnética. |
| 2 | `newell_coupling` | ★★★★☆ | Função ε = v^(4/3) × Bs^(2/3). Integra velocidade e Bz de forma não-linear, capturando a eficiência de transferência de energia. |
| 3 | `cme` | ★★★★☆ | CMEs adicionam ~1–2 pontos ao KP. São a principal causa de tempestades G3–G5 históricas. |
| 4 | `velocidade_vento` | ★★★☆☆ | Maior velocidade = maior energia cinética = maior compressão da magnetosfera. |
| 5 | `pressao_dinamica` | ★★★☆☆ | P = ρv²/2 comprime a magnetosfera e amplifica os efeitos mesmo com Bz moderado. |

**Validação científica:** A hierarquia de importância descoberta pelo algoritmo coincide com o que décadas de pesquisa em física do plasma estabeleceram — o modelo está capturando fenômenos reais, não artefatos estatísticos.

---

## 7. Deploy da Aplicação

**Script:** `5_app_streamlit.py` · **Plataforma:** Streamlit Cloud

🚀 **URL pública:** https://globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app

### Funcionalidades da Interface

| Aba | Funcionalidade |
|-----|----------------|
| ⚡ Previsão em Tempo Real | Sliders para todos os parâmetros do vento solar + gauge KP + badge G0–G5 + probabilidades por classe |
| 🔍 Interpretabilidade SHAP | Todos os 6 plots SHAP gerados + relatório físico textual |
| 📊 Métricas dos Modelos | Tabelas comparativas + gráficos de RMSE/MAE/R² e Accuracy/F1/Confusion Matrix + EDA |
| 🛸 Sobre o GAIE | Contexto, metodologia, fontes, referências científicas e equipe |

### Como testar o modelo no app

Para simular uma **tempestade severa (G3–G4)**, ajuste os sliders:
- Campo Bz → **−30 nT**
- Velocidade → **750 km/s**
- Ativar **CME**
- Flare → **Classe M**

O gauge deve saltar para KP ≈ 7–8 e o badge mostrar G3–G4.

---

## 8. Instruções para Execução

### Pré-requisitos
- Python 3.10+
- pip
- Git

### Passo a Passo Completo

```bash
# 1. Clonar o repositório
git clone https://github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI.git
cd GLOBAL_SOLUTION_GENERATIVE_AI/gaie_helios

# 2. Instalar todas as dependências
pip install -r requirements.txt

# 3. Executar o pipeline completo
#    (coleta de dados → pré-processamento → treino → SHAP)
python 0_pipeline.py

# 4. Iniciar a aplicação web local
streamlit run 5_app_streamlit.py
# Abrirá em: http://localhost:8501
```

### Execução por Etapa (opcional)

```bash
python 1_coleta_dados.py             # Etapa 1: Coleta NOAA + NASA DONKI
python 2_preprocessamento.py         # Etapa 2: Limpeza + features + EDA
python 3_modelos.py                  # Etapa 3: Treino + comparação + métricas
python 4_shap_interpretabilidade.py  # Etapa 4: Análise SHAP + plots
```

### Variável de Ambiente (NASA API Key)

```bash
# Windows PowerShell
$env:NASA_API_KEY = "sua_chave_aqui"

# Linux / macOS
export NASA_API_KEY="sua_chave_aqui"
```

> Chave gratuita obtida em: https://api.nasa.gov/ — o pipeline funciona sem ela (usa dados sintéticos como fallback).

---

## 9. Estrutura do Projeto

```
gaie_helios/
│
├── 0_pipeline.py                     # Orquestrador — roda todos os scripts em sequência
├── 1_coleta_dados.py                 # Coleta NOAA SWPC + NASA DONKI + suplemento sintético
├── 2_preprocessamento.py             # Limpeza + 20 features + EDA + split + RobustScaler
├── 3_modelos.py                      # 6 modelos (3 reg. + 3 clf.) + métricas + gráficos
├── 4_shap_interpretabilidade.py      # SHAP TreeExplainer + 6 plots + relatório físico
├── 5_app_streamlit.py                # App web Streamlit (deploy)
├── requirements.txt                  # Dependências Python
├── README.md                         # Esta documentação
│
├── data/
│   ├── solar_wind_dataset.csv        # Dataset final (11.249 linhas × 13 colunas)
│   ├── X_train.npy / X_val.npy / X_test.npy   # Arrays pré-processados
│   ├── y_reg_*.npy / y_clf_*.npy     # Targets de regressão e classificação
│   └── feature_cols.pkl              # Lista das 20 features dos modelos
│
├── models/
│   ├── best_regressor.pkl            # XGBoost regressão — RMSE=0.2768, R²=0.9678
│   ├── best_classifier.pkl           # XGBoost classificação — F1=0.9772, Acc=0.9787
│   └── scaler.pkl                    # RobustScaler treinado (fit apenas no treino)
│
├── outputs/
│   ├── eda_solar.png                 # Painel EDA com 9 gráficos exploratórios
│   ├── comparacao_regressao.png      # Comparação RMSE / MAE / R² entre os 3 modelos
│   ├── comparacao_classificacao.png  # Accuracy / F1 / Confusion Matrix entre os 3 modelos
│   ├── metricas.json                 # Todas as métricas em JSON estruturado
│   └── interpretacao_shap.txt        # Relatório físico das variáveis mais importantes
│
└── shap_plots/
    ├── summary_regressao.png         # Beeswarm — impacto de cada feature (regressão)
    ├── bar_regressao.png             # Importância média absoluta (regressão)
    ├── dependence_bz_regressao.png   # Dependence plot: Bz × velocidade
    ├── waterfall_extremo_regressao.png  # Explicação do caso mais severo
    ├── summary_classificacao.png     # Beeswarm — impacto (classificação)
    └── bar_classificacao.png         # Importância média (classificação)
```

---

## 10. Referências Científicas

1. **Newell, P.T. et al. (2008)**. A solar wind magnetosphere coupling function controlling multitude of geophysical phenomena. *Journal of Geophysical Research*, 113, A09218. doi:10.1029/2007JA012825

2. **Borovsky, J.E. & Denton, M.H. (2006)**. Differences between CME-driven storms and CIR-driven storms. *Journal of Geophysical Research*, 111, A07S08. doi:10.1029/2005JA011447

3. **Richardson, I.G. & Cane, H.V. (2012)**. Near-Earth solar wind magnetic fields, plasma and energetic particle data during more than four solar cycles. *Journal of Geophysical Research*, 117, A08110. doi:10.1029/2011JA017364

4. **Lloyd's of London (2013)**. *Solar Storm Risk to the North American Electric Grid*. Lloyd's of London Risk Report.

---

*FIAP — Generative AI for Engineering · Global Solution 2026/1 · Turma 4ESPY*
*Dados fornecidos por NASA/GSFC (DONKI API) e NOAA/SWPC — Satélite DSCOVR, Ponto L1*
