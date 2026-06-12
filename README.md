# Retail Vision Intelligence System — TP2 LIACD

Este repositório contém a implementação do Sistema de Inteligência Visual para Gestão de Retalho, desenvolvido no âmbito da unidade curricular de Linguagem de Interação e Integração de Modelos de Grande Escala (LIACD).

O sistema processa imagens de prateleiras de supermercado através de modelos LLM multimodais, deteta anomalias operacionais (como ruturas de stock ou desalinhamento de produtos), converte regras de negócio em linguagem natural para formatos executáveis (JSON) e disponibiliza uma memória contextual baseada em RAG (Retrieval-Augmented Generation).

---

## Autor
* Nome: Ricardo Fernandes da Costa
* Número de Aluno: 53732
* Ano Letivo: 2025/2026

---

## Estrutura do Projeto

O projeto respeita a arquitetura obrigatória de isolamento de componentes exigida pelo enunciado:

```text
tp2/
│
├── data/
│   ├── images/
│   │   ├── normal/        # Amostras do SKU-110K
│   │   ├── vazia/         # Prateleiras vazias
│   │   ├── planograma/    # Inconformidades de plano
│   │   ├── desordenada/   # Prateleiras desorganizadas
│   │   └── ambigua/       # Casos dúbios
│   └── reports/           # Relatórios gerados pelo sistema
│
├── src/                   # CODIGO FONTE OPERACIONAL
│   ├── __init__.py
│   ├── shelf_inspector.py # Motor de Inspeção Visual (Multimodal)
│   ├── rule_engine.py     # Tradutor e Motor de Regras NLP
│   ├── rag_memory.py      # Gestor de Memória Vetorial (RAG)
│   └── report_generator.py# Agregador e Gerador de Relatórios
│
├── .env                   # Variáveis de ambiente e chaves de API (Ignorado no Git)
├── requirements.txt       # Dependências de pacotes Python
├── download_dataset.py    # Script de setup automatizado de imagens
├── interface.py           # Interface de Linha de Comando (CLI) principal
└── evaluate.py            # Harness de Avaliação Unificado do Sistema