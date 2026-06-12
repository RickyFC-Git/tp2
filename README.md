# Retail Vision Intelligence System
### TP2 LIACD 2025/2026

Sistema de inspecao continua de prateleiras de supermercado com memoria, capaz de analisar imagens com LLM multimodal, aprender regras do gestor em linguagem natural, e integrar contexto historico via RAG.

---

## Estrutura do Projeto

```
tp2/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── download_dataset.py     -- Fase 0: construcao do dataset
├── evaluate.py             -- Harness de avaliacao
├── data/
│   ├── images/             -- Dataset (gerado pelo download_dataset.py)
│   ├── inspections/        -- Resultados de inspecao (gerado em runtime)
│   ├── rules/              -- Regras persistidas (gerado em runtime)
│   ├── logs/               -- Logs de execucao (gerado em runtime)
│   └── reports/            -- Relatorios gerados (gerado em runtime)
├── src/
│   ├── shelf_inspector.py  -- Componente 1: analise visual
│   ├── rule_engine.py      -- Componente 2: motor de regras
│   ├── rag_memory.py       -- Componente 3: memoria RAG
│   ├── report_generator.py -- Componente 4: gerador de relatorios
│   └── interface.py        -- Componente 5: interface CLI
├── prompts/
│   ├── shelf_inspector_A.txt
│   ├── shelf_inspector_B.txt
│   ├── shelf_inspector_C.txt
│   ├── rule_engine_converter.txt
│   ├── rule_engine_ambiguidades.txt
│   ├── rag_summary.txt
│   ├── rag_resposta.txt
│   ├── report_sumario.txt
│   ├── report_recomendacoes.txt
│   └── llm_judge_*.txt
├── vectorstore/            -- ChromaDB persistente (gerado em runtime)
└── cache/                  -- Cache de resultados API (gerado em runtime)
```

---

## Instalacao

### 1. Clonar o repositorio e instalar dependencias

```bash
git clone <url-do-repositorio>
cd tp2
pip install -r requirements.txt
```

### 2. Configurar a chave de API

```bash
cp .env.example .env
```

Abre o ficheiro `.env` e preenche a chave:

```
GEMINI_API_KEY=a_tua_chave_aqui
```

Obter chave gratuita: https://aistudio.google.com -> Get API Key

### 3. Construir o dataset (exemplo)

Descarregar os ZIPs do Roboflow manualmente (conta gratuita necessaria):
- `vazia.zip`      -> universe.roboflow.com/fyp-ormnr/supermarket-empty-shelf-detector
- `planograma.zip` -> universe.roboflow.com/fyp-ormnr/empty-shelf-detector
- `desordenada.zip`-> universe.roboflow.com/object-detection-5pf5v/packaging-defect-detection-wbcpk
- `ambigua.zip`    -> universe.roboflow.com/planogram-pc7rp/planogram-pyyeu

Colocar os ZIPs em `data/zips/` e correr:

```bash
python download_dataset.py
```

---

## Uso

### Iniciar a interface interativa

```bash
python src/interface.py
```

### Comandos disponiveis

```
# Inspecao
inspect Z_S1 --image data/images/normal/normal_0001.jpg
inspect Z_S1 --image data/images/normal/normal_0001.jpg --strategy A
inspect all --images-dir data/images/normal/

# Regras
add rule "Avisa-me quando a prateleira inferior estiver mais de 40% vazia"
list rules
delete rule RULE_001
test rule RULE_001 --image data/images/vazia/vazia_0001.jpg
show rule RULE_001

# Historico
history "quando foi a ultima vez que Z_S1 teve problemas?"
history "que zonas tiveram mais issues esta semana?"
compare Z_S1 Z_S2 --period 7

# Relatorios
report --session today
report --zone Z_S1 --period 14

# Sistema
status
help
exit
```

### Usar componentes individualmente

```bash
# Inspecionar uma imagem
python src/shelf_inspector.py data/images/vazia/vazia_0001.jpg --zone Z_S1
python src/shelf_inspector.py data/images/vazia/vazia_0001.jpg --all-strategies

# Gerir regras
python src/rule_engine.py add "Avisa-me quando o fill rate cair abaixo de 50%"
python src/rule_engine.py list
python src/rule_engine.py delete RULE_001

# Indexar inspecoes no RAG
python src/rag_memory.py index-all data/inspections/
python src/rag_memory.py query "problemas na zona Z_S1"
python src/rag_memory.py stats

# Gerar relatorio
python src/report_generator.py --session data/inspections/
python src/report_generator.py --zone Z_S1 --period 14
```

---

## Avaliacao

```bash
# Com ground truth fornecido pelo professor
python evaluate.py --images-dir test_images/ --output evaluation_report.json

# Com ground truth proprio (ficheiro JSON com anotacoes)
python evaluate.py --images-dir test_images/ --ground-truth ground_truth.json --output evaluation_report.json

# Avaliar as 3 estrategias de prompting
python evaluate.py --images-dir test_images/ --strategy all --output evaluation_report.json
```

### Formato do ground_truth.json

```json
[
  {
    "image_path": "test_images/img_001.jpg",
    "zone_id": "Z_S1",
    "overall_status": "warning",
    "issues": [
      {"type": "empty_shelf", "location": "prateleira inferior", "severity": "high"}
    ],
    "shelf_fill_rate_min": 0.4,
    "shelf_fill_rate_max": 0.7
  }
]
```

---

## Dataset

| Categoria     | Quantidade | Fonte                                          | Licenca          |
|---------------|------------|------------------------------------------------|------------------|
| Normal        | 150        | SKU-110K (Goldman et al., 2019)                | Academica        |
| Vazia         | 100        | Roboflow: fyp-ormnr/supermarket-empty-shelf    | Open Source      |
| Planograma    | 100        | Roboflow: cardatasetcombine/planogram-compliance| Open Source     |
| Desordenada   | 80         | Roboflow: packaging-defect-detection           | Open Source      |
| Ambigua       | 70         | Roboflow: planogram-pc7rp/planogram-pyyeu      | Open Source      |
| **Total**     | **500**    |                                                |                  |

---

## Modelo

- **Principal**: Google Gemini 1.5 Flash (API gratuita, 1500 req/dia, 15 req/min)
- **Embeddings**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (local, gratuito)
- **Vector Store**: ChromaDB (local, persistente em disco)

---

## Notas Tecnicas

- A chave de API nunca deve ser commitada no repositorio (esta no `.gitignore`)
- O cache MD5 evita chamadas duplicadas a API
- O sistema funciona em modo cache-only quando a quota diaria e esgotada
- `temperature=0` em todos os testes para reproducibilidade maxima