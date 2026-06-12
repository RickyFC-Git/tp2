"""
evaluate.py -- Harness de Avaliacao do Sistema
===============================================
Uso:
  python evaluate.py --images-dir data/images/ --ground-truth ground_truth.json --strategy all --output evaluation_report.json
"""

import os
import json
import time
import argparse
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "openai/gpt-4o-mini"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Importa componentes -- tenta src/ primeiro, depois raiz ──────────────────
try:
    from src import shelf_inspector, rule_engine, rag_memory, report_generator
    log.info("Componentes carregados de src/")
except ImportError:
    try:
        import shelf_inspector, rule_engine, rag_memory, report_generator
        log.info("Componentes carregados da raiz")
    except ImportError as e:
        log.error(f"Erro ao importar componentes: {e}")
        shelf_inspector = rule_engine = rag_memory = report_generator = None

COMPONENTES_OK = all([shelf_inspector, rule_engine, rag_memory, report_generator])

# ── Regras de teste ───────────────────────────────────────────────────────────
REGRAS_TESTE = [
    ("Avisa-me quando a prateleira inferior estiver mais de 30% vazia", True, False),
    ("Na zona Z_S1, se nao houver produtos de laticinios visiveis, e critico", True, False),
    ("Quando o fill rate cair abaixo de 60% entre as 10h e as 13h, avisa-me", True, False),
    ("Se um produto estiver tombado, considera sempre severidade alta", True, False),
    ("Avisa-me quando a prateleira estiver vazia", False, True),
    ("Notifica-me se houver problemas", False, True),
    ("Alerta para situacoes na loja", False, True),
]

QUERIES_RAG = [
    {"query": "Quando foi a ultima vez que a zona Z_S1 teve problemas de prateleira vazia?", "inspection_ids_relevantes": []},
    {"query": "Que zonas tiveram mais issues de planograma?", "inspection_ids_relevantes": []},
    {"query": "Existe algum padrao nos problemas detetados?", "inspection_ids_relevantes": []},
]

# ── Prompts LLM-as-judge ──────────────────────────────────────────────────────
PROMPT_HALLUCINATION = """Es um avaliador rigoroso de sistemas de visao computacional.
Avalia se a descricao contem afirmacoes nao verificaveis sem ver a imagem original.

DESCRICAO: "{descricao}"
CONTEXTO: tipo={tipo}, localizacao={localizacao}, severidade={severidade}

Criterios: marcas especificas sem evidencia, quantidades exatas, inferencias causais sem base visual.

Responde APENAS com JSON valido:
{{"score": 0.0, "e_alucinacao": false, "justificacao": "..."}}
score: 0.0 (sem alucinacao) a 1.0 (alucinacao clara)"""

PROMPT_RELEVANCE = """Es um avaliador de sistemas RAG.
QUERY: "{query}"
RESPOSTA: "{resposta}"
Avalia se a resposta e relevante e util para a query.
Responde APENAS com JSON valido:
{{"score": 0.0, "justificacao": "..."}}
score: 0.0 (irrelevante) a 1.0 (perfeitamente relevante)"""

PROMPT_FAITHFULNESS = """Es um avaliador de fidelidade RAG.
CONTEXTO: {contexto}
RESPOSTA: "{resposta}"
Avalia se as afirmacoes da resposta sao suportadas pelo contexto.
Responde APENAS com JSON valido:
{{"score": 0.0, "afirmacoes_sem_suporte": [], "justificacao": "..."}}
score: 0.0 (infiel) a 1.0 (fiel)"""

PROMPT_REPORT = """Es um gestor de loja experiente a avaliar um relatorio de inspecao.
RELATORIO: {relatorio}
Avalia (0-10 cada): sumario executivo, problemas por zona, recomendacoes, utilidade geral.
Responde APENAS com JSON valido:
{{"sumario_score": 0, "problemas_score": 0, "recomendacoes_score": 0, "utilidade_score": 0, "score_total": 0.0, "pontos_fortes": [], "pontos_fracos": [], "justificacao": "..."}}"""


# ── LLM-as-judge via OpenRouter ───────────────────────────────────────────────
def chamar_judge(prompt: str) -> dict:
    if not OPENROUTER_API_KEY:
        return {"score": 0.0, "erro": "OPENROUTER_API_KEY em falta"}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tp2-liacd",
        "X-Title": "Retail Vision Intelligence System",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=30,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                linhas = raw.split("\n")
                raw = "\n".join(linhas[1:-1] if linhas[-1].strip() == "```" else linhas[1:])
            return json.loads(raw)
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = (2 ** attempt) * 5
                log.warning(f"Rate limit. Aguardando {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"Erro HTTP judge: {e}")
                return {"score": 0.0, "erro": str(e)}
        except Exception as e:
            log.error(f"Erro judge: {e}")
            return {"score": 0.0, "erro": str(e)}

    return {"score": 0.0, "erro": "Falha apos 3 tentativas"}


# ── Analise visual ────────────────────────────────────────────────────────────
def avaliar_analise_visual(ground_truth: list[dict], strategy: str = "B") -> dict:
    strategies_a_testar = ["A", "B", "C"] if strategy == "all" else [strategy]
    resultados = {}

    for strat in strategies_a_testar:
        log.info(f"A avaliar estrategia {strat}...")
        total = len(ground_truth)
        json_parse_ok = 0
        issues_detetados = 0
        issues_gt_total = 0
        false_positives = 0
        issues_pred_total = 0
        severity_corretos = 0
        severity_total = 0
        hallucination_scores = []

        for item in ground_truth:
            image_path = item.get("image_path")
            if not Path(image_path).exists():
                log.warning(f"Imagem nao encontrada: {image_path}")
                total -= 1
                continue

            try:
                resultado = shelf_inspector.inspect_shelf(
                    image_path, strategy=strat, zone_id=item.get("zone_id", "Z_TEST")
                )
            except Exception as e:
                log.error(f"Erro ao inspecionar {image_path}: {e}")
                continue

            if "_parse_error" not in resultado:
                json_parse_ok += 1

            gt_issues = item.get("issues", [])
            pred_issues = resultado.get("issues", [])
            issues_gt_total += len(gt_issues)
            issues_pred_total += len(pred_issues)

            for gt in gt_issues:
                if any(p.get("type") == gt.get("type") for p in pred_issues):
                    issues_detetados += 1

            for pred in pred_issues:
                if not any(g.get("type") == pred.get("type") for g in gt_issues):
                    false_positives += 1

            for gt in gt_issues:
                match = next((p for p in pred_issues if p.get("type") == gt.get("type")), None)
                if match:
                    severity_total += 1
                    if match.get("severity") == gt.get("severity"):
                        severity_corretos += 1

            for pred in pred_issues:
                descricao = pred.get("description", "")
                if descricao:
                    res = chamar_judge(PROMPT_HALLUCINATION.format(
                        descricao=descricao,
                        tipo=pred.get("type", ""),
                        localizacao=pred.get("location", ""),
                        severidade=pred.get("severity", ""),
                    ))
                    hallucination_scores.append(res.get("score", 0.0))
                    time.sleep(0.5)

        resultados[strat] = {
            "strategy": strat,
            "total_imagens": total,
            "issue_detection_rate": round(issues_detetados / issues_gt_total, 3) if issues_gt_total > 0 else None,
            "false_positive_rate": round(false_positives / issues_pred_total, 3) if issues_pred_total > 0 else None,
            "severity_accuracy": round(severity_corretos / severity_total, 3) if severity_total > 0 else None,
            "json_parse_rate": round(json_parse_ok / total, 3) if total > 0 else 0,
            "hallucination_rate": round(sum(hallucination_scores) / len(hallucination_scores), 3) if hallucination_scores else None,
        }
        log.info(f"  [{strat}] JPR={resultados[strat]['json_parse_rate']:.1%}")

    return resultados


# ── RAG ───────────────────────────────────────────────────────────────────────
def avaliar_rag(queries_gt: list[dict]) -> dict:
    resultados = {}

    for strat in ["hybrid", "by_issue"]:
        log.info(f"A avaliar RAG estrategia {strat}...")
        try:
            recall_result = rag_memory.avaliar_recall_at_k(queries_gt, strategy=strat, k=3)
            resultados[f"recall_at_3_{strat}"] = recall_result.get("recall_at_3", 0)
        except Exception as e:
            log.error(f"Erro Recall@3 ({strat}): {e}")
            resultados[f"recall_at_3_{strat}"] = None

    faithfulness_scores = []
    relevance_scores = []

    for item in queries_gt:
        query = item.get("query", "")
        try:
            resultado = rag_memory.responder_query(query)
            resposta = resultado.get("resposta", "")
            chunks = resultado.get("chunks_usados", [])

            if resposta and chunks:
                contexto = "\n".join([c.get("texto", "") for c in chunks[:3]])
                faith = chamar_judge(PROMPT_FAITHFULNESS.format(
                    contexto=contexto[:1000], resposta=resposta[:500]
                ))
                faithfulness_scores.append(faith.get("score", 0.0))
                time.sleep(0.5)

                rel = chamar_judge(PROMPT_RELEVANCE.format(
                    query=query, resposta=resposta[:500]
                ))
                relevance_scores.append(rel.get("score", 0.0))
                time.sleep(0.5)
        except Exception as e:
            log.error(f"Erro RAG query '{query}': {e}")

    resultados["faithfulness"] = round(sum(faithfulness_scores) / len(faithfulness_scores), 3) if faithfulness_scores else None
    resultados["answer_relevance"] = round(sum(relevance_scores) / len(relevance_scores), 3) if relevance_scores else None
    return resultados


# ── Rule Engine ───────────────────────────────────────────────────────────────
def avaliar_rule_engine() -> dict:
    log.info("A avaliar rule engine...")

    total = len(REGRAS_TESTE)
    parse_ok = 0
    ambiguidade_correta = 0
    correctness_ok = 0
    total_nao_ambiguas = sum(1 for _, v, a in REGRAS_TESTE if v and not a)
    detalhes = []

    inspecao_sintetica = {
        "inspection_id": "INS_SYNTH_001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "zone_id": "Z_S1",
        "overall_status": "warning",
        "shelf_fill_rate": 0.55,
        "issues": [
            {"type": "empty_shelf", "location": "prateleira inferior", "severity": "high"},
            {"type": "misaligned", "location": "prateleira central", "severity": "medium"},
        ],
    }

    for texto, esperado_valido, tem_ambiguidade in REGRAS_TESTE:
        try:
            # Tenta converter_regra (nome correto no teu rule_engine.py)
            if hasattr(rule_engine, "converter_regra"):
                regra = rule_engine.converter_regra(texto)
            elif hasattr(rule_engine, "converter_regra_nlp"):
                regra = rule_engine.converter_regra_nlp(texto)
            else:
                raise AttributeError("Funcao de conversao de regra nao encontrada no rule_engine")

            time.sleep(1)

            is_valid_json = isinstance(regra, dict) and "rule_id" in regra
            if is_valid_json:
                parse_ok += 1

            ambiguidades = regra.get("validation", {}).get("ambiguities", [])
            sistema_detetou = len(ambiguidades) > 0

            if tem_ambiguidade == sistema_detetou:
                ambiguidade_correta += 1

            if esperado_valido and not tem_ambiguidade and is_valid_json:
                try:
                    if hasattr(rule_engine, "verificar_condicoes"):
                        rule_engine.verificar_condicoes(regra, inspecao_sintetica)
                    correctness_ok += 1
                except Exception:
                    pass

            detalhes.append({
                "texto": texto[:70],
                "esperado_valido": esperado_valido,
                "tem_ambiguidade": tem_ambiguidade,
                "parse_ok": is_valid_json,
                "ambiguidade_detetada": sistema_detetou,
                "ambiguidades": ambiguidades,
            })

        except Exception as e:
            log.error(f"Erro ao testar regra '{texto[:50]}': {e}")
            detalhes.append({"texto": texto[:70], "erro": str(e)})

    return {
        "rule_parse_rate": round(parse_ok / total, 3) if total > 0 else 0,
        "rule_correctness": round(correctness_ok / total_nao_ambiguas, 3) if total_nao_ambiguas > 0 else 0,
        "ambiguity_detection": round(ambiguidade_correta / total, 3) if total > 0 else 0,
        "total_regras_testadas": total,
        "detalhes": detalhes,
    }


# ── Relatorio ─────────────────────────────────────────────────────────────────
def avaliar_relatorio(inspecoes: list[dict]) -> dict:
    if not inspecoes:
        return {"nota": "Sem inspecoes para gerar relatorio"}
    try:
        relatorio = report_generator.gerar_relatorio(inspecoes, guardar=False)
        resultado = chamar_judge(PROMPT_REPORT.format(relatorio=relatorio[:3000]))
        resultado["relatorio_gerado"] = True
        return resultado
    except Exception as e:
        return {"erro": str(e), "relatorio_gerado": False}


# ── Main ──────────────────────────────────────────────────────────────────────
def executar_avaliacao(
    images_dir: str,
    ground_truth_path: Optional[str] = None,
    output_path: str = "evaluation_report.json",
    strategy: str = "B",
) -> dict:
    print("=" * 60)
    print("  TP2 LIACD -- Harness de Avaliacao (OpenRouter / gpt-4o-mini)")
    print("=" * 60)

    if not COMPONENTES_OK:
        print("[ERRO] Componentes do sistema nao carregaram.")
        return {}

    if not OPENROUTER_API_KEY:
        print("[ERRO] OPENROUTER_API_KEY nao definida no .env")
        return {}

    # Carrega ground truth
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path, encoding="utf-8") as f:
            ground_truth = json.load(f)
        print(f"[OK] Ground truth: {len(ground_truth)} imagens anotadas")
    else:
        images_path = Path(images_dir)
        ficheiros = sorted(
            list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        )[:10]
        ground_truth = [
            {"image_path": str(f), "zone_id": "Z_TEST", "overall_status": "unknown", "issues": []}
            for f in ficheiros
        ]
        print(f"[AVISO] Sem ground truth. A usar {len(ground_truth)} imagens sem anotacoes.")

    relatorio = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "images_dir": images_dir,
        "total_imagens_teste": len(ground_truth),
        "strategy_principal": strategy,
        "modelo": MODEL_NAME,
    }

    # 1. Analise visual
    print("\n[1/4] A avaliar analise visual...")
    try:
        relatorio["analise_visual"] = avaliar_analise_visual(ground_truth, strategy=strategy)
    except Exception as e:
        relatorio["analise_visual"] = {"erro": str(e)}

    # 2. RAG
    print("\n[2/4] A avaliar sistema RAG...")
    try:
        insp_dir = Path("data/inspections")
        if insp_dir.exists():
            ids = list({f.stem.split("_strategy")[0] for f in insp_dir.glob("*.json")})
            for q in QUERIES_RAG:
                if not q["inspection_ids_relevantes"] and ids:
                    q["inspection_ids_relevantes"] = ids[:2]
        relatorio["rag"] = avaliar_rag(QUERIES_RAG)
    except Exception as e:
        relatorio["rag"] = {"erro": str(e)}

    # 3. Rule Engine
    print("\n[3/4] A avaliar rule engine...")
    try:
        relatorio["rule_engine"] = avaliar_rule_engine()
    except Exception as e:
        relatorio["rule_engine"] = {"erro": str(e)}

    # 4. LLM-as-judge no relatorio
    print("\n[4/4] A avaliar relatorio (LLM-as-judge)...")
    try:
        insp_dir = Path("data/inspections")
        strat_rel = "B" if strategy == "all" else strategy
        inspecoes_geradas = []
        if insp_dir.exists():
            for f in sorted(insp_dir.glob(f"*_strategy{strat_rel}.json"))[:5]:
                with open(f, encoding="utf-8") as fp:
                    inspecoes_geradas.append(json.load(fp))
        relatorio["avaliacao_relatorio"] = avaliar_relatorio(inspecoes_geradas)
    except Exception as e:
        relatorio["avaliacao_relatorio"] = {"erro": str(e)}

    # Guarda
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    # Imprime resumo
    print("\n" + "=" * 60)
    print("  RESULTADOS")
    print("=" * 60)

    av = relatorio.get("analise_visual", {})
    for strat_key, metricas in av.items():
        if not isinstance(metricas, dict) or "json_parse_rate" not in metricas:
            continue
        print(f"\n  Estrategia {strat_key}:")
        nomes = {
            "issue_detection_rate": "Issue Detection Rate",
            "false_positive_rate":  "False Positive Rate ",
            "severity_accuracy":    "Severity Accuracy   ",
            "json_parse_rate":      "JSON Parse Rate     ",
            "hallucination_rate":   "Hallucination Rate  ",
        }
        for chave, nome in nomes.items():
            valor = metricas.get(chave)
            if valor is None:
                print(f"    {nome}: N/A")
            else:
                print(f"    {nome}: {valor:.1%}")

    rag = relatorio.get("rag", {})
    if rag and "erro" not in rag:
        print(f"\n  RAG:")
        r3h = rag.get("recall_at_3_hybrid")
        r3b = rag.get("recall_at_3_by_issue")
        fth = rag.get("faithfulness")
        rel = rag.get("answer_relevance")
        print(f"    Recall@3 (hybrid)   : {f'{r3h:.1%}' if r3h is not None else 'N/A'}")
        print(f"    Recall@3 (by_issue) : {f'{r3b:.1%}' if r3b is not None else 'N/A'}")
        print(f"    Faithfulness        : {f'{fth:.1%}' if fth is not None else 'N/A'}")
        print(f"    Answer Relevance    : {f'{rel:.1%}' if rel is not None else 'N/A'}")

    re_m = relatorio.get("rule_engine", {})
    if re_m and "erro" not in re_m:
        print(f"\n  Rule Engine:")
        print(f"    Rule Parse Rate     : {re_m.get('rule_parse_rate', 0):.1%}")
        print(f"    Rule Correctness    : {re_m.get('rule_correctness', 0):.1%}")
        print(f"    Ambiguity Detection : {re_m.get('ambiguity_detection', 0):.1%}")

    ar = relatorio.get("avaliacao_relatorio", {})
    if ar and "score_total" in ar:
        print(f"\n  Relatorio (LLM-as-judge):")
        print(f"    Score total         : {ar.get('score_total', 0):.1f}/10")

    print(f"\n[OK] Relatorio guardado em: {output_path}")
    print("=" * 60)

    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harness de avaliacao do Retail Vision Intelligence System")
    parser.add_argument("--images-dir", required=True, help="Pasta com imagens de teste")
    parser.add_argument("--ground-truth", help="Ficheiro JSON com ground truth (opcional)")
    parser.add_argument("--output", default="evaluation_report.json", help="Ficheiro de saida")
    parser.add_argument("--strategy", choices=["A", "B", "C", "all"], default="B",
                        help="Estrategia a avaliar. 'all' avalia A, B e C")
    args = parser.parse_args()

    executar_avaliacao(
        images_dir=args.images_dir,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        strategy=args.strategy,
    )