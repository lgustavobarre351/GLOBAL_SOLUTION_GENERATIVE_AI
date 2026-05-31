const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, Header, Footer, TableOfContents, ExternalHyperlink,
  PageBreak, LevelFormat
} = require("docx");
const fs = require("fs");

// ── Cores ────────────────────────────────────────────────────────────
const AZUL_ESCURO  = "1B3A6B";
const AZUL_MEDIO  = "2E75B6";
const AZUL_CLARO  = "D5E8F0";
const CINZA_FUNDO = "F2F2F2";
const BRANCO      = "FFFFFF";

// ── Bordas de tabela ─────────────────────────────────────────────────
const borda = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const bordas = { top: borda, bottom: borda, left: borda, right: borda };

// ── Helpers ──────────────────────────────────────────────────────────
function paragrafo(texto, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.LEFT,
    spacing: { before: opts.before ?? 80, after: opts.after ?? 80 },
    children: [new TextRun({
      text: texto,
      bold: opts.bold || false,
      italics: opts.italic || false,
      size: opts.size || 24,
      font: opts.font || "Arial",
      color: opts.color || "000000",
    })]
  });
}

function titulo(texto, nivel, numStr) {
  const sizes = { 1: 36, 2: 28, 3: 26 };
  const headings = {
    1: HeadingLevel.HEADING_1,
    2: HeadingLevel.HEADING_2,
    3: HeadingLevel.HEADING_3
  };
  return new Paragraph({
    heading: headings[nivel],
    spacing: { before: nivel === 1 ? 400 : 240, after: 160 },
    children: [new TextRun({
      text: numStr ? `${numStr}  ${texto}` : texto,
      bold: true,
      size: sizes[nivel] || 26,
      font: "Arial",
      color: nivel === 1 ? AZUL_ESCURO : AZUL_MEDIO,
    })]
  });
}

function espacador(pts = 1) {
  return new Paragraph({ spacing: { before: pts * 40, after: pts * 40 }, children: [] });
}

function linhaDivisoria() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: AZUL_MEDIO, space: 1 } },
    spacing: { before: 80, after: 80 },
    children: []
  });
}

function itemLista(texto, negrito_prefix) {
  const children = [];
  if (negrito_prefix) {
    children.push(new TextRun({ text: negrito_prefix, bold: true, size: 22, font: "Arial" }));
    children.push(new TextRun({ text: texto, size: 22, font: "Arial" }));
  } else {
    children.push(new TextRun({ text: texto, size: 22, font: "Arial" }));
  }
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children
  });
}

function codigo(texto) {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    shading: { fill: CINZA_FUNDO, type: ShadingType.CLEAR },
    indent: { left: 360 },
    children: [new TextRun({ text: texto, font: "Courier New", size: 20, color: "2D2D2D" })]
  });
}

function celulaHeader(texto, largura) {
  return new TableCell({
    borders: bordas,
    width: { size: largura, type: WidthType.DXA },
    shading: { fill: AZUL_ESCURO, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 160, right: 160 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: texto, bold: true, size: 22, font: "Arial", color: BRANCO })]
    })]
  });
}

function celulaAlternada(texto, largura, sombreado, alinhamento) {
  return new TableCell({
    borders: bordas,
    width: { size: largura, type: WidthType.DXA },
    shading: { fill: sombreado ? "EBF3FB" : BRANCO, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 160, right: 160 },
    children: [new Paragraph({
      alignment: alinhamento || AlignmentType.LEFT,
      children: [new TextRun({ text: texto, size: 22, font: "Arial" })]
    })]
  });
}

function celulaDestaque(texto, largura, sombreado) {
  return new TableCell({
    borders: bordas,
    width: { size: largura, type: WidthType.DXA },
    shading: { fill: sombreado ? "EBF3FB" : BRANCO, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 160, right: 160 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: texto, bold: texto === "0.9678" || texto === "0.9787", size: 22, font: "Arial", color: (texto === "0.2768" || texto === "0.9787") ? "1B6B1B" : "000000" })]
    })]
  });
}

// ── Tabela de regressão ──────────────────────────────────────────────
function tabelaRegressao() {
  const colunas = [3600, 1920, 1920, 1920];
  const linhas_dados = [
    ["Random Forest", "0.2814", "0.1057", "0.9668"],
    ["XGBoost", "0.2768", "0.1704", "0.9678"],
    ["Ridge (baseline)", "0.6473", "0.5292", "0.8241"],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colunas,
    rows: [
      new TableRow({
        tableHeader: true,
        children: [
          celulaHeader("Modelo", colunas[0]),
          celulaHeader("RMSE ↓", colunas[1]),
          celulaHeader("MAE ↓", colunas[2]),
          celulaHeader("R² ↑", colunas[3]),
        ]
      }),
      ...linhas_dados.map((linha, i) =>
        new TableRow({
          children: linha.map((cel, j) =>
            j === 0
              ? celulaAlternada(cel, colunas[j], i % 2 === 1)
              : celulaDestaque(cel, colunas[j], i % 2 === 1)
          )
        })
      ),
      new TableRow({
        children: [
          new TableCell({
            columnSpan: 4,
            borders: bordas,
            shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 160, right: 160 },
            width: { size: 9360, type: WidthType.DXA },
            children: [new Paragraph({
              children: [
                new TextRun({ text: "Melhor modelo: ", bold: true, size: 22, font: "Arial" }),
                new TextRun({ text: "XGBoost — RMSE=0.2768, R²=0.9678", size: 22, font: "Arial", color: "1B6B1B" }),
              ]
            })]
          })
        ]
      })
    ]
  });
}

function tabelaClassificacao() {
  const colunas = [4320, 2520, 2520];
  const linhas_dados = [
    ["Random Forest", "0.9757", "0.9759"],
    ["XGBoost", "0.9787", "0.9772"],
    ["Logistic Regression (baseline)", "0.9710", "0.9741"],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colunas,
    rows: [
      new TableRow({
        tableHeader: true,
        children: [
          celulaHeader("Modelo", colunas[0]),
          celulaHeader("Accuracy ↑", colunas[1]),
          celulaHeader("F1-weighted ↑", colunas[2]),
        ]
      }),
      ...linhas_dados.map((linha, i) =>
        new TableRow({
          children: linha.map((cel, j) =>
            j === 0
              ? celulaAlternada(cel, colunas[j], i % 2 === 1)
              : celulaDestaque(cel, colunas[j], i % 2 === 1)
          )
        })
      ),
      new TableRow({
        children: [
          new TableCell({
            columnSpan: 3,
            borders: bordas,
            shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 160, right: 160 },
            width: { size: 9360, type: WidthType.DXA },
            children: [new Paragraph({
              children: [
                new TextRun({ text: "Melhor modelo: ", bold: true, size: 22, font: "Arial" }),
                new TextRun({ text: "XGBoost — F1=0.9772, Accuracy=0.9787", size: 22, font: "Arial", color: "1B6B1B" }),
              ]
            })]
          })
        ]
      })
    ]
  });
}

function tabelaArquivos() {
  const colunas = [3200, 6160];
  const arquivos = [
    ["0_pipeline.py", "Orquestrador — executa todos os scripts em sequência"],
    ["1_coleta_dados.py", "Coleta NOAA SWPC + NASA DONKI + geração sintética"],
    ["2_preprocessamento.py", "Limpeza, feature engineering e EDA"],
    ["3_modelos.py", "Treinamento e comparação de 6 modelos"],
    ["4_shap_interpretabilidade.py", "Análise SHAP com plots e relatório físico"],
    ["5_app_streamlit.py", "Aplicação web interativa"],
    ["requirements.txt", "Dependências Python"],
    ["README.md", "Documentação completa do projeto"],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colunas,
    rows: [
      new TableRow({
        tableHeader: true,
        children: [celulaHeader("Arquivo", colunas[0]), celulaHeader("Descrição", colunas[1])]
      }),
      ...arquivos.map((linha, i) =>
        new TableRow({
          children: [
            new TableCell({
              borders: bordas,
              width: { size: colunas[0], type: WidthType.DXA },
              shading: { fill: i % 2 === 1 ? "EBF3FB" : BRANCO, type: ShadingType.CLEAR },
              margins: { top: 80, bottom: 80, left: 160, right: 160 },
              children: [new Paragraph({
                children: [new TextRun({ text: linha[0], font: "Courier New", size: 20, bold: true })]
              })]
            }),
            celulaAlternada(linha[1], colunas[1], i % 2 === 1)
          ]
        })
      )
    ]
  });
}

// ── Monta o documento ────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 640, hanging: 360 } } }
        }]
      },
      {
        reference: "numeros",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 640, hanging: 360 } } }
        }]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 24, color: "000000" } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: AZUL_ESCURO },
        paragraph: { spacing: { before: 400, after: 160 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: AZUL_MEDIO },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "404040" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 }
      }
    ]
  },
  sections: [
    // ══════════════════════════════════════════════
    // CAPA
    // ══════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      children: [
        espacador(10),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 120 },
          children: [new TextRun({ text: "Generative AI for Engineering", size: 24, font: "Arial", color: "666666" })]
        }),
        linhaDivisoria(),
        espacador(6),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 200 },
          children: [new TextRun({ text: "GAIE", size: 72, bold: true, font: "Arial", color: AZUL_ESCURO })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 160 },
          children: [new TextRun({ text: "Geomagnetic AI Engine", size: 44, bold: true, font: "Arial", color: AZUL_MEDIO })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 600 },
          children: [new TextRun({
            text: "Pipeline Completo de Machine Learning para Previsao de Tempestades Geomagneticas",
            size: 28, font: "Arial", color: "444444", italics: true
          })]
        }),
        linhaDivisoria(),
        espacador(6),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 80 },
          children: [new TextRun({ text: "GitHub", size: 22, bold: true, font: "Arial", color: "444444" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 160 },
          children: [new ExternalHyperlink({
            link: "https://github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI",
            children: [new TextRun({ text: "github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI", style: "Hyperlink", size: 20, font: "Arial" })]
          })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 80 },
          children: [new TextRun({ text: "Aplicacao em Funcionamento", size: 22, bold: true, font: "Arial", color: "444444" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 600 },
          children: [new ExternalHyperlink({
            link: "https://globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app",
            children: [new TextRun({ text: "globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app", style: "Hyperlink", size: 20, font: "Arial" })]
          })]
        }),
        linhaDivisoria(),
        new Paragraph({ children: [new PageBreak()] }),
      ]
    },

    // ══════════════════════════════════════════════
    // CONTEUDO
    // ══════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: AZUL_MEDIO, space: 1 } },
            spacing: { before: 0, after: 160 },
            children: [
              new TextRun({ text: "GAIE — Geomagnetic AI Engine", bold: true, size: 20, font: "Arial", color: AZUL_ESCURO }),
              new TextRun({ text: "   |   Generative AI for Engineering", size: 20, font: "Arial", color: "888888" }),
            ]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: AZUL_MEDIO, space: 1 } },
            alignment: AlignmentType.CENTER,
            spacing: { before: 120, after: 0 },
            children: [
              new TextRun({ text: "Pagina ", size: 18, font: "Arial", color: "888888" }),
              new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Arial", color: "888888" }),
              new TextRun({ text: " de ", size: 18, font: "Arial", color: "888888" }),
              new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, font: "Arial", color: "888888" }),
            ]
          })]
        })
      },
      children: [
        // SUMARIO
        new Paragraph({
          heading: HeadingLevel.HEADING_1,
          children: [new TextRun({ text: "Sumario", bold: true, size: 36, font: "Arial", color: AZUL_ESCURO })]
        }),
        new TableOfContents("Sumario", { hyperlink: true, headingStyleRange: "1-3" }),
        new Paragraph({ children: [new PageBreak()] }),

        // ── 1. CONTEXTO DO PROBLEMA ──────────────────────────────
        titulo("Contexto do Problema", 1, "1."),
        linhaDivisoria(),
        paragrafo(
          "Tempestades geomagneticas sao disturbios no campo magnetico da Terra causados pela interacao do vento solar com a magnetosfera. Quando o campo magnetico interplanetario aponta para o sul (Bz negativo), ocorre reconexao magnetica na magnetopausa, injetando particulas energeticas no anel de corrente e gerando perturbacoes que afetam satelites, redes eletricas, GPS e comunicacoes de radio.",
          { after: 120 }
        ),
        titulo("Impactos Economicos Documentados", 2, "1.1"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2400, 2160, 4800],
          rows: [
            new TableRow({
              tableHeader: true,
              children: [
                celulaHeader("Evento", 2400),
                celulaHeader("Data", 2160),
                celulaHeader("Impacto", 4800),
              ]
            }),
            new TableRow({ children: [
              celulaAlternada("Quebec Blackout", 2400, false),
              celulaAlternada("Marco 1989", 2160, false),
              celulaAlternada("6 milhoes sem energia por 9h — US$ 2 bilhoes de prejuizo", 4800, false),
            ]}),
            new TableRow({ children: [
              celulaAlternada("Halloween Storms", 2400, true),
              celulaAlternada("Outubro 2003", 2160, true),
              celulaAlternada("30 satelites danificados, blackout de radio HF global", 4800, true),
            ]}),
            new TableRow({ children: [
              celulaAlternada("Tempestade de Carrington", 2400, false),
              celulaAlternada("Setembro 1859", 2160, false),
              celulaAlternada("Repeticao hoje causaria US$ 0,6 a 2,6 trilhoes (Lloyd's 2013)", 4800, false),
            ]}),
          ]
        }),
        espacador(2),
        paragrafo(
          "O KP Index (0-9) e o indice geomagnetico planetario principal. Quando atinge 5 ou mais, uma tempestade G1-G5 esta em andamento. Prever sua intensidade com antecedencia e crucial para protecao de satelites e sistemas GPS, gestao de redes eletricas, seguranca de astronautas e aviacao em rotas polares.",
          { after: 120 }
        ),
        titulo("Sobre o GAIE e o HELIOS", 2, "1.2"),
        paragrafo(
          "O GAIE e a camada preditiva de Machine Learning integrada ao HELIOS Space Intelligence Platform, plataforma que ja monitora dados em tempo real da NASA DONKI e NOAA SWPC. Enquanto o HELIOS exibe o que esta acontecendo agora, o GAIE prevê o que vai acontecer nas proximas horas com base nos padroes aprendidos nos dados historicos.",
          { after: 80 }
        ),
        new Paragraph({
          spacing: { before: 80, after: 160 },
          children: [
            new TextRun({ text: "HELIOS: ", bold: true, size: 22, font: "Arial" }),
            new ExternalHyperlink({
              link: "https://helius-zeta.vercel.app/",
              children: [new TextRun({ text: "https://helius-zeta.vercel.app/", style: "Hyperlink", size: 22, font: "Arial" })]
            })
          ]
        }),
        new Paragraph({ children: [new PageBreak()] }),

        // ── 2. FONTE DOS DADOS ───────────────────────────────────
        titulo("Fonte dos Dados", 1, "2."),
        linhaDivisoria(),
        paragrafo("Foram utilizadas quatro fontes de dados, todas publicas e sem custo de acesso:", { after: 120 }),
        titulo("NOAA SWPC Solar Wind MAG", 2, "2.1"),
        paragrafo("Dados magneticos do vento solar com resolucao de 1 minuto, fornecidos pelo satelite DSCOVR posicionado no ponto Lagrangiano L1, a 1,5 milhao de km da Terra. Fornece as componentes do campo magnetico interplanetario: Bz, Bt, Bx e By em nanoTesla.", { after: 80 }),
        new Paragraph({
          spacing: { before: 40, after: 120 },
          children: [
            new TextRun({ text: "Endpoint: ", bold: true, size: 22, font: "Arial" }),
            new ExternalHyperlink({
              link: "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
              children: [new TextRun({ text: "services.swpc.noaa.gov/products/solar-wind/mag-7-day.json", style: "Hyperlink", size: 20, font: "Arial" })]
            })
          ]
        }),
        titulo("NOAA SWPC Solar Wind Plasma", 2, "2.2"),
        paragrafo("Dados de plasma do vento solar com resolucao de 1 minuto. Fornece velocidade (km/s), densidade de protons (p/cc) e temperatura (eV).", { after: 80 }),
        new Paragraph({
          spacing: { before: 40, after: 120 },
          children: [
            new TextRun({ text: "Endpoint: ", bold: true, size: 22, font: "Arial" }),
            new ExternalHyperlink({
              link: "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
              children: [new TextRun({ text: "services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json", style: "Hyperlink", size: 20, font: "Arial" })]
            })
          ]
        }),
        titulo("NOAA SWPC KP Index", 2, "2.3"),
        paragrafo("Indice geomagnetico planetario atualizado a cada 3 horas. Escala de 0 a 9, onde 5 ou mais indica tempestade geomagnetica ativa.", { after: 80 }),
        titulo("NASA DONKI API", 2, "2.4"),
        paragrafo("API da NASA para eventos espaciais em tempo real. Utilizada para coleta de flares solares (FLR) e tempestades geomagneticas (GST). Requer API Key gratuita em api.nasa.gov.", { after: 120 }),
        titulo("Composicao do Dataset Final", 2, "2.5"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3600, 3000, 2760],
          rows: [
            new TableRow({ tableHeader: true, children: [
              celulaHeader("Fonte", 3600), celulaHeader("Linhas", 3000), celulaHeader("Tipo", 2760)
            ]}),
            new TableRow({ children: [
              celulaAlternada("NOAA MAG + Plasma + KP", 3600, false),
              celulaAlternada("9.749", 3000, false),
              celulaAlternada("Dados reais (satelite)", 2760, false),
            ]}),
            new TableRow({ children: [
              celulaAlternada("Sintetico (Borovsky 2006 / Newell 2008)", 3600, true),
              celulaAlternada("1.500", 3000, true),
              celulaAlternada("Complemento para tempestades", 2760, true),
            ]}),
            new TableRow({ children: [
              new TableCell({
                borders: bordas, width: { size: 3600, type: WidthType.DXA },
                shading: { fill: AZUL_ESCURO, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 160, right: 160 },
                children: [new Paragraph({ children: [new TextRun({ text: "TOTAL", bold: true, size: 22, font: "Arial", color: BRANCO })] })]
              }),
              new TableCell({
                borders: bordas, width: { size: 3000, type: WidthType.DXA },
                shading: { fill: AZUL_ESCURO, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 160, right: 160 },
                children: [new Paragraph({ children: [new TextRun({ text: "11.249 linhas x 13 colunas", bold: true, size: 22, font: "Arial", color: BRANCO })] })]
              }),
              new TableCell({
                borders: bordas, width: { size: 2760, type: WidthType.DXA },
                shading: { fill: AZUL_ESCURO, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 160, right: 160 },
                children: [new Paragraph({ children: [new TextRun({ text: "Acima do minimo exigido", bold: true, size: 22, font: "Arial", color: BRANCO })] })]
              }),
            ]})
          ]
        }),
        new Paragraph({ children: [new PageBreak()] }),

        // ── 3. METODOLOGIA ───────────────────────────────────────
        titulo("Metodologia", 1, "3."),
        linhaDivisoria(),
        titulo("Engenharia de Atributos", 2, "3.1"),
        paragrafo("Foram criadas 20 features a partir das 13 colunas brutas originais, cada uma com justificativa fisica:", { after: 80 }),
        itemLista("bz_negativo = max(-Bz, 0): ", "isola a componente sul do campo magnetico, unico driver de reconexao magnetica. Bz positivo nao causa tempestades."),
        itemLista("newell_coupling = v^(4/3) x Bs^(2/3): ", "funcao de acoplamento de Newell (2008), mede a eficiencia de transferencia de energia do vento solar para a magnetosfera de forma nao-linear."),
        itemLista("cme_bz_interacao = CME x Bz_sul: ", "captura a sinergia entre CME ativo e campo sul intenso — a combinacao mais geoefetiva conhecida."),
        itemLista("pressao_dinamica = 0.5 x rho x v²: ", "compressao da magnetosfera. Alta pressao reduz o raio de standoff e amplifica outros efeitos."),
        itemLista("bz_media_3h e bz_media_6h: ", "medias moveis que capturam pre-condicionamento geomagnetico recente."),
        itemLista("velocidade_media_3h e velocidade_media_6h: ", "tendencia recente da velocidade do vento solar."),
        itemLista("hora_sin/cos e mes_sin/cos: ", "componentes ciclicas para capturar variacao diurna e efeito Russell-McPherron (tempestades mais frequentes nos equinocios)."),
        espacador(2),
        titulo("Pre-processamento", 2, "3.2"),
        itemLista("Remocao de outliers com limites fisicos reais (ex: Bz entre -80 e +30 nT)", ""),
        itemLista("Split estratificado: 70% treino / 15% validacao / 15% teste", ""),
        itemLista("Normalizacao: RobustScaler (robusto a outliers de tempestade — usa mediana e IQR)", ""),
        espacador(2),
        titulo("Modelos Treinados", 2, "3.3"),
        paragrafo("Foram treinados 6 modelos em dois problemas distintos com os mesmos dados:", { after: 80 }),
        new Paragraph({
          spacing: { before: 80, after: 40 },
          children: [new TextRun({ text: "Regressao — prever KP Index continuo (0 a 9):", bold: true, size: 22, font: "Arial", color: AZUL_MEDIO })]
        }),
        itemLista("Random Forest Regressor (200 arvores, max_depth=12)", ""),
        itemLista("XGBoost Regressor (200 estimadores, learning_rate=0.05)", ""),
        itemLista("Ridge Regression — baseline linear", ""),
        new Paragraph({
          spacing: { before: 120, after: 40 },
          children: [new TextRun({ text: "Classificacao — prever nivel G0 a G5:", bold: true, size: 22, font: "Arial", color: AZUL_MEDIO })]
        }),
        itemLista("Random Forest Classifier (200 arvores, class_weight=balanced)", ""),
        itemLista("XGBoost Classifier (200 estimadores, multi-classe)", ""),
        itemLista("Logistic Regression — baseline linear", ""),
        new Paragraph({ children: [new PageBreak()] }),

        // ── 4. RESULTADOS ────────────────────────────────────────
        titulo("Resultados Obtidos", 1, "4."),
        linhaDivisoria(),
        titulo("Regressao — KP Index", 2, "4.1"),
        paragrafo("Metricas calculadas no conjunto de teste (15% dos dados, nunca vistos durante o treino):", { after: 120 }),
        tabelaRegressao(),
        espacador(2),
        titulo("Classificacao — Nivel G", 2, "4.2"),
        paragrafo("Metricas de classificacao multi-classe para prever o nivel de tempestade G0 a G5:", { after: 120 }),
        tabelaClassificacao(),
        espacador(2),
        titulo("Analise dos Resultados", 2, "4.3"),
        paragrafo(
          "O XGBoost venceu em ambos os problemas. Com dados reais (nao-lineares), o gradient boosting captura melhor as interacoes complexas entre variaveis do vento solar do que modelos lineares. O R2 de 0.97 indica que o modelo explica 97% da variacao no KP Index — um resultado excepcional para dados geomagneticos.",
          { after: 80 }
        ),
        paragrafo(
          "O Ridge, modelo linear baseline, obteve R2=0.82 — suficientemente alto para confirmar que existe uma relacao linear de base, mas o XGBoost captura as nao-linearidades fisicas adicionais (reconexao magnetica nao e um processo linear).",
          { after: 80 }
        ),
        new Paragraph({ children: [new PageBreak()] }),

        // ── 5. SHAP ──────────────────────────────────────────────
        titulo("Interpretabilidade com SHAP", 1, "5."),
        linhaDivisoria(),
        paragrafo(
          "A analise SHAP (SHapley Additive exPlanations) foi aplicada usando TreeExplainer nos modelos XGBoost para quantificar a contribuicao de cada variavel em cada previsao individual. A metodologia vem da teoria dos jogos: o valor de Shapley de uma feature e sua contribuicao media considerando todas as combinacoes possiveis de features.",
          { after: 120 }
        ),
        titulo("Top 5 Variaveis Mais Importantes", 2, "5.1"),
        new Paragraph({
          spacing: { before: 80, after: 40 },
          children: [
            new TextRun({ text: "1. bz_negativo", bold: true, size: 24, font: "Arial", color: AZUL_MEDIO }),
          ]
        }),
        paragrafo("A componente sul do campo magnetico interplanetario e o maior driver. Quando Bz fica abaixo de -10 nT, a reconexao magnetica na magnetopausa injeta particulas energeticas no anel de corrente, elevando rapidamente o KP Index. E a variavel com maior SHAP medio absoluto em todos os experimentos.", { after: 80 }),
        new Paragraph({
          spacing: { before: 80, after: 40 },
          children: [new TextRun({ text: "2. newell_coupling", bold: true, size: 24, font: "Arial", color: AZUL_MEDIO })]
        }),
        paragrafo("A funcao epsilon = v^(4/3) x Bs^(2/3) integra velocidade e Bz de forma nao-linear, capturando a eficiencia de transferencia de energia de forma mais precisa do que as variaveis isoladas. Desenvolvida por Newell et al. (2008) a partir de decadas de dados observacionais.", { after: 80 }),
        new Paragraph({
          spacing: { before: 80, after: 40 },
          children: [new TextRun({ text: "3. cme (evento ativo)", bold: true, size: 24, font: "Arial", color: AZUL_MEDIO })]
        }),
        paragrafo("Presenca de ejecao de massa coronal adiciona 1 a 2 pontos ao KP. CMEs sao a principal causa de tempestades G3-G5 historicas — transportam plasma denso e campos magneticos intensos que interagem com a magnetosfera por 24 a 48 horas.", { after: 80 }),
        new Paragraph({
          spacing: { before: 80, after: 40 },
          children: [new TextRun({ text: "4. velocidade_vento", bold: true, size: 24, font: "Arial", color: AZUL_MEDIO })]
        }),
        paragrafo("Maior velocidade equivale a maior energia cinetica e maior compressao da magnetosfera. Fundamental em tempestades CIR (Corotating Interaction Region) causadas por correntes de vento solar rapido.", { after: 80 }),
        new Paragraph({
          spacing: { before: 80, after: 40 },
          children: [new TextRun({ text: "5. pressao_dinamica", bold: true, size: 24, font: "Arial", color: AZUL_MEDIO })]
        }),
        paragrafo("P = rho x v^2 / 2 comprime a magnetosfera mesmo quando Bz e moderado, amplificando o efeito de outras variaveis e reduzindo o raio de Alfven.", { after: 120 }),
        titulo("Graficos SHAP Gerados", 2, "5.2"),
        itemLista("Summary Plot (beeswarm): distribuicao do impacto de cada feature em todas as amostras", ""),
        itemLista("Bar Plot: importancia media absoluta das features (ranking global)", ""),
        itemLista("Dependence Plot (Bz): como o impacto do Bz varia com a velocidade do vento solar", ""),
        itemLista("Waterfall Plot: explicacao detalhada do caso mais extremo do dataset", ""),
        new Paragraph({ children: [new PageBreak()] }),

        // ── 6. INSTRUCOES ────────────────────────────────────────
        titulo("Instrucoes para Execucao", 1, "6."),
        linhaDivisoria(),
        titulo("Pre-requisitos", 2, "6.1"),
        itemLista("Python 3.10 ou superior", ""),
        itemLista("pip (gerenciador de pacotes Python)", ""),
        itemLista("Git (para clonar o repositorio)", ""),
        espacador(2),
        titulo("Passo a Passo", 2, "6.2"),
        new Paragraph({
          spacing: { before: 120, after: 40 },
          children: [new TextRun({ text: "Passo 1 — Clonar o repositorio", bold: true, size: 22, font: "Arial" })]
        }),
        codigo("git clone https://github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI"),
        codigo("cd GLOBAL_SOLUTION_GENERATIVE_AI/gaie_helios"),
        new Paragraph({
          spacing: { before: 120, after: 40 },
          children: [new TextRun({ text: "Passo 2 — Instalar dependencias", bold: true, size: 22, font: "Arial" })]
        }),
        codigo("pip install -r requirements.txt"),
        new Paragraph({
          spacing: { before: 120, after: 40 },
          children: [new TextRun({ text: "Passo 3 — Executar o pipeline completo", bold: true, size: 22, font: "Arial" })]
        }),
        codigo("python 0_pipeline.py"),
        paragrafo("Este comando executa automaticamente todos os scripts em sequencia: coleta de dados, pre-processamento, treinamento dos modelos e analise SHAP.", { after: 80 }),
        new Paragraph({
          spacing: { before: 120, after: 40 },
          children: [new TextRun({ text: "Passo 4 — Iniciar a aplicacao", bold: true, size: 22, font: "Arial" })]
        }),
        codigo("streamlit run 5_app_streamlit.py"),
        paragrafo("A aplicacao abrira automaticamente em http://localhost:8501", { after: 120 }),
        titulo("Estrutura de Arquivos", 2, "6.3"),
        tabelaArquivos(),
        new Paragraph({ children: [new PageBreak()] }),

        // ── 7. LINKS ─────────────────────────────────────────────
        titulo("Links de Entrega", 1, "7."),
        linhaDivisoria(),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2800, 6560],
          rows: [
            new TableRow({ tableHeader: true, children: [
              celulaHeader("Recurso", 2800), celulaHeader("Link", 6560)
            ]}),
            new TableRow({ children: [
              celulaAlternada("Repositorio GitHub", 2800, false),
              new TableCell({
                borders: bordas, width: { size: 6560, type: WidthType.DXA },
                shading: { fill: BRANCO, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 160, right: 160 },
                children: [new Paragraph({ children: [new ExternalHyperlink({
                  link: "https://github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI",
                  children: [new TextRun({ text: "github.com/lgustavobarre351/GLOBAL_SOLUTION_GENERATIVE_AI", style: "Hyperlink", size: 20, font: "Arial" })]
                })]})]
              })
            ]}),
            new TableRow({ children: [
              celulaAlternada("Aplicacao Streamlit", 2800, true),
              new TableCell({
                borders: bordas, width: { size: 6560, type: WidthType.DXA },
                shading: { fill: "EBF3FB", type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 160, right: 160 },
                children: [new Paragraph({ children: [new ExternalHyperlink({
                  link: "https://globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app",
                  children: [new TextRun({ text: "globalsolutiongenerativeai-gkw5rmitemjc8d7ue7mvub.streamlit.app", style: "Hyperlink", size: 20, font: "Arial" })]
                })]})]
              })
            ]}),
            new TableRow({ children: [
              celulaAlternada("Plataforma HELIOS", 2800, false),
              new TableCell({
                borders: bordas, width: { size: 6560, type: WidthType.DXA },
                shading: { fill: BRANCO, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 160, right: 160 },
                children: [new Paragraph({ children: [new ExternalHyperlink({
                  link: "https://helius-zeta.vercel.app/",
                  children: [new TextRun({ text: "helius-zeta.vercel.app", style: "Hyperlink", size: 20, font: "Arial" })]
                })]})]
              })
            ]}),
          ]
        }),
        espacador(4),

        // ── 8. REFERENCIAS ───────────────────────────────────────
        titulo("Referencias Cientificas", 1, "8."),
        linhaDivisoria(),
        new Paragraph({
          numbering: { reference: "numeros", level: 0 },
          spacing: { before: 80, after: 120 },
          children: [
            new TextRun({ text: "Newell, P.T. et al. (2008). ", bold: true, size: 22, font: "Arial" }),
            new TextRun({ text: "A solar wind magnetosphere coupling function controlling multitude of geophysical phenomena. ", italics: true, size: 22, font: "Arial" }),
            new TextRun({ text: "Journal of Geophysical Research, 113, A09218. doi:10.1029/2007JA012825", size: 22, font: "Arial" }),
          ]
        }),
        new Paragraph({
          numbering: { reference: "numeros", level: 0 },
          spacing: { before: 80, after: 120 },
          children: [
            new TextRun({ text: "Borovsky, J.E. & Denton, M.H. (2006). ", bold: true, size: 22, font: "Arial" }),
            new TextRun({ text: "Differences between CME-driven storms and CIR-driven storms. ", italics: true, size: 22, font: "Arial" }),
            new TextRun({ text: "Journal of Geophysical Research, 111, A07S08. doi:10.1029/2005JA011447", size: 22, font: "Arial" }),
          ]
        }),
        new Paragraph({
          numbering: { reference: "numeros", level: 0 },
          spacing: { before: 80, after: 120 },
          children: [
            new TextRun({ text: "Richardson, I.G. & Cane, H.V. (2012). ", bold: true, size: 22, font: "Arial" }),
            new TextRun({ text: "Near-Earth solar wind magnetic fields, plasma and energetic particle data during more than four solar cycles. ", italics: true, size: 22, font: "Arial" }),
            new TextRun({ text: "Journal of Geophysical Research, 117, A08110. doi:10.1029/2011JA017364", size: 22, font: "Arial" }),
          ]
        }),
        new Paragraph({
          numbering: { reference: "numeros", level: 0 },
          spacing: { before: 80, after: 120 },
          children: [
            new TextRun({ text: "Lloyd's of London (2013). ", bold: true, size: 22, font: "Arial" }),
            new TextRun({ text: "Solar Storm Risk to the North American Electric Grid. ", italics: true, size: 22, font: "Arial" }),
            new TextRun({ text: "Lloyd's of London Risk Report.", size: 22, font: "Arial" }),
          ]
        }),
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("C:\\Users\\luisg\\Desktop\\GAIA\\gaie_helios\\GAIE_Documentacao.docx", buffer);
  console.log("Documento gerado: GAIE_Documentacao.docx");
}).catch(err => {
  console.error("Erro:", err);
  process.exit(1);
});
