import os
import json
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import shelf_inspector
    import rule_engine
    import rag_memory
    import report_generator
    COMPONENTES_OK = True
except ImportError as e:
    log.error(f"Erro ao importar componentes: {e}")
    COMPONENTES_OK = False

REGRAS_TESTE = [
    ("Avisa-me quando a prateleira inferior estiver mais de 30% vazia", True, False),
    ("Na zona Z_S1, se nao houver produtos de laticinios visiveis, e critico", True, False),
    ("Quando o fill rate cair abaixo de 60% entre as 10h e as 13h, avisa-me", True, False),
    ("Se um produto estiver tombado, considera sempre severidade alta", True, False),
    ("Avisa-me quando a prateleira estiver vazia", False, True),
    ("Notifica-me se houver problemas", False, True),
    ("Alerta para situacoes anormais", False, True),
]

QUERIES_RAG = [
    {"query": "Quando foi a ultima vez que a zona Z_S1 teve problemas de prateleira vazia?", "inspection_ids_relevantes": []},
    {"query": "Que zonas tiveram mais issues de planograma?", "inspection_ids_relevantes": []},
    {"query": "Existe algum padrao nos problemas detetados?", "inspection_ids_relevantes": []},
]

PROMPT_JUDGE_HALLUCINATION = """Es um avaliador rigoroso de sistemas de visao computacional.

Tens a descricao de um issue detetado num sistema de inspecao de prateleiras.
Avalia se a descricao contem afirmacoes que nao sao verificaveis sem ver a imagem original.

DESCRICAO DO ISSUE: "{descricao}"
CONTEXTO: tipo={tipo}, localizacao={localizacao}, severidade={severidade}

Criterios de alucinacao:
- Menciona marcas especificas sem evidencia
- Afirma quantidades exatas sem confirmacao
- Descreve cores/formas de forma demasiado especifica
- Faz inferencias causais sem base visual

Responde APENAS com JSON:
{{"score": 0.0, "e_alucinacao": false, "justificacao": "..."}}
score: 0.0 (sem alucinacao) a 1.0 (alucinacao clara)"""

PROMPT_JUDGE_RELEVANCE = """Es um avaliador de sistemas de recuperacao de informacao.

QUERY: "{query}"
RESPOSTA: "{resposta}"

Avalia se a resposta e relevante e util para a query.
Criterios: aborda a pergunta diretamente? Contem informacao especifica? Referencias a IDs e datas?

Responde APENAS com JSON:
{{"score": 0.0, "justificacao": "..."}}
score: 0.0 (irrelevante) a 1.0 (perfeitamente relevante)"""

PROMPT_JUDGE_FAITHFULNESS = """Es um avaliador de fidelidade em sistemas RAG.

CONTEXTO RECUPERADO:
{contexto}

RESPOSTA GERADA: "{resposta}"

Avalia se as afirmacoes da resposta sao suportadas pelos chunks recuperados.

Responde APENAS com JSON:
{{"score": 0.0, "afirmacoes_sem_suporte": [], "justificacao": "..."}}
score: 0.0 (totalmente infiel) a 1.0 (totalmente fiel)"""

PROMPT_JUDGE_REPORT = """Es um gestor de loja de retalho experiente a avaliar um relatorio de inspecao.

RELATORIO:
{relatorio}

Avalia nos seguintes criterios (0-10 cada):
1. Sumario executivo claro e acionavel
2. Problemas por zona bem descritos
3. Recomendacoes especificas e executaveis
4. Utilidade geral para o gestor

Responde APENAS com JSON:
{{
  "sumario_score": 0,
  "problemas_score": 0,
  "recomendacoes_score": 0,
  "utilidade_score": 0,
  "score_total": 0.0,
  "pontos_fortes": [],
  "pontos_fracos": [],
  "justificacao": "..."
}}"""


def chamar_judge(prompt: str) -> dict:
    """Chama o Gemini como juiz e faz parse do JSON devolvido."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        raw = response.text.strip()
        if raw.startswith("```"):
            linhas = raw.split("\n")
            raw = "\n".join(linhas[1:-1] if linhas[-1].strip() == "```" else linhas[1:])
        return json.loads(raw)
    except Exception as e:
        log.error(f"Erro no LLM-as-judge: {e}")
        return {"score": 0.0, "erro": str(e)}


def avaliar_analise_visual(ground_truth: list[dict], strategy: str = "B") -> dict:
    """Avalia o shelf_inspector. Se strategy='all', avalia A, B e C."""
    strategies_a_testar = ["A", "B", "C"] if strategy == "all" else [strategy]
    resultados_por_strategy = {}

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
                    res = chamar_judge(PROMPT_JUDGE_HALLUCINATION.format(
                        descricao=descricao,
                        tipo=pred.get("type", ""),
                        localizacao=pred.get("location", ""),
                        severidade=pred.get("severity", ""),
                    ))
                    hallucination_scores.append(res.get("score", 0.0))
                    time.sleep(0.5)

        resultados_por_strategy[strat] = {
            "strategy": strat,
            "total_imagens": total,
            "issue_detection_rate": round(issues_detetados / issues_gt_total, 3) if issues_gt_total > 0 else None,
            "false_positive_rate": round(false_positives / issues_pred_total, 3) if issues_pred_total > 0 else None,
            "severity_accuracy": round(severity_corretos / severity_total, 3) if severity_total > 0 else None,
            "json_parse_rate": round(json_parse_ok / total, 3) if total > 0 else 0,
            "hallucination_rate": round(sum(hallucination_scores) / len(hallucination_scores), 3) if hallucination_scores else None,
        }

        log.info(f"  [{strat}] JPR={resultados_por_strategy[strat]['json_parse_rate']:.1%}")

    return resultados_por_strategy


def avaliar_rag(queries_gt: list[dict]) -> dict:
    """Avalia o sistema RAG."""
    resultados = {}

    for strat in ["hybrid", "by_issue"]:
        log.info(f"A avaliar RAG estrategia {strat}...")
        recall_result = rag_memory.avaliar_recall_at_k(queries_gt, strategy=strat, k=3)
        resultados[f"recall_at_3_{strat}"] = recall_result.get("recall_at_3", 0)

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

                faith = chamar_judge(PROMPT_JUDGE_FAITHFULNESS.format(
                    contexto=contexto[:1000], resposta=resposta[:500]
                ))
                faithfulness_scores.append(faith.get("score", 0.0))
                time.sleep(0.5)

                rel = chamar_judge(PROMPT_JUDGE_RELEVANCE.format(
                    query=query, resposta=resposta[:500]
                ))
                relevance_scores.append(rel.get("score", 0.0))
                time.sleep(0.5)

        except Exception as e:
            log.error(f"Erro ao avaliar RAG para '{query}': {e}")

    resultados["faithfulness"] = round(sum(faithfulness_scores) / len(faithfulness_scores), 3) if faithfulness_scores else 0
    resultados["answer_relevance"] = round(sum(relevance_scores) / len(relevance_scores), 3) if relevance_scores else 0

    return resultados


def avaliar_rule_engine() -> dict:
    """Avalia o rule engine."""
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
            regra = rule_engine.converter_regra(texto)
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
                    rule_engine.verificar_condicoes(regra, inspecao_sintetica)
                    correctness_ok += 1
                except Exception:
                    pass

            detalhes.append({
                "texto": texto[:60],
                "esperado_valido": esperado_valido,
                "tem_ambiguidade": tem_ambiguidade,
                "parse_ok": is_valid_json,
                "ambiguidade_detetada": sistema_detetou,
            })

        except Exception as e:
            log.error(f"Erro ao converter regra: {e}")
            detalhes.append({"texto": texto[:60], "erro": str(e)})

    return {
        "rule_parse_rate": round(parse_ok / total, 3) if total > 0 else 0,
        "rule_correctness": round(correctness_ok / total_nao_ambiguas, 3) if total_nao_ambiguas > 0 else 0,
        "ambiguity_detection": round(ambiguidade_correta / total, 3) if total > 0 else 0,
        "total_regras_testadas": total,
        "detalhes": detalhes,
    }


def avaliar_relatorio(inspecoes: list[dict]) -> dict:
    """Avalia um relatorio gerado via LLM-as-judge."""
    if not inspecoes:
        return {"nota": "Sem inspecoes para gerar relatorio"}
    try:
        relatorio = report_generator.gerar_relatorio(inspecoes, guardar=False)
        resultado = chamar_judge(PROMPT_JUDGE_REPORT.format(relatorio=relatorio[:3000]))
        resultado["relatorio_gerado"] = True
        return resultado
    except Exception as e:
        return {"erro": str(e), "relatorio_gerado": False}


def executar_avaliacao(
    images_dir: str,
    ground_truth_path: Optional[str] = None,
    output_path: str = "evaluation_report.json",
    strategy: str = "B",
) -> dict:
    """Executa o harness completo de avaliacao."""
    print("=" * 60)
    print("  TP2 LIACD -- Harness de Avaliacao")
    print("=" * 60)

    if not COMPONENTES_OK:
        print("[ERRO] Componentes do sistema nao carregaram.")
        return {}

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
        print("        IDR, FPR e Severity Accuracy nao serao calculados.")

    relatorio = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "images_dir": images_dir,
        "total_imagens_teste": len(ground_truth),
        "strategy_principal": strategy,
    }

    print("\n[1/4] A avaliar analise visual...")
    try:
        relatorio["analise_visual"] = avaliar_analise_visual(ground_truth, strategy=strategy)
    except Exception as e:
        relatorio["analise_visual"] = {"erro": str(e)}

    print("\n[2/4] A avaliar sistema RAG...")
    try:
        insp_dir = Path("data/inspections")
        ids = [f.stem.split("_strategy")[0] for f in insp_dir.glob("*.json")]
        for q in QUERIES_RAG:
            if not q["inspection_ids_relevantes"] and ids:
                q["inspection_ids_relevantes"] = ids[:2]
        relatorio["rag"] = avaliar_rag(QUERIES_RAG)
    except Exception as e:
        relatorio["rag"] = {"erro": str(e)}

    print("\n[3/4] A avaliar rule engine...")
    try:
        relatorio["rule_engine"] = avaliar_rule_engine()
    except Exception as e:
        relatorio["rule_engine"] = {"erro": str(e)}

    print("\n[4/4] A avaliar relatorio (LLM-as-judge)...")
    try:
        insp_dir = Path("data/inspections")
        inspecoes_geradas = []
        for f in sorted(insp_dir.glob(f"*_strategy{strategy if strategy != 'all' else 'B'}.json"))[:5]:
            with open(f, encoding="utf-8") as fp:
                inspecoes_geradas.append(json.load(fp))
        relatorio["avaliacao_relatorio"] = avaliar_relatorio(inspecoes_geradas)
    except Exception as e:
        relatorio["avaliacao_relatorio"] = {"erro": str(e)}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  RESULTADOS")
    print("=" * 60)

    av = relatorio.get("analise_visual", {})
    for strat_key, metricas in av.items():
        if isinstance(metricas, dict) and "json_parse_rate" in metricas:
            print(f"\n  Estrategia {strat_key}:")
            for metrica, valor in metricas.items():
                if metrica in ("strategy", "total_imagens"):
                    continue
                if valor is None:
                    print(f"    {metrica:<28}: N/A (sem ground truth)")
                else:
                    print(f"    {metrica:<28}: {valor:.1%}")

    rag = relatorio.get("rag", {})
    if rag and "erro" not in rag:
        print(f"\n  RAG:")
        print(f"    Recall@3 (hybrid)            : {rag.get('recall_at_3_hybrid', 0):.1%}")
        print(f"    Recall@3 (by_issue)          : {rag.get('recall_at_3_by_issue', 0):.1%}")
        print(f"    Faithfulness                 : {rag.get('faithfulness', 0):.1%}")
        print(f"    Answer Relevance             : {rag.get('answer_relevance', 0):.1%}")

    re_m = relatorio.get("rule_engine", {})
    if re_m and "erro" not in re_m:
        print(f"\n  Rule Engine:")
        print(f"    Rule Parse Rate              : {re_m.get('rule_parse_rate', 0):.1%}")
        print(f"    Rule Correctness             : {re_m.get('rule_correctness', 0):.1%}")
        print(f"    Ambiguity Detection          : {re_m.get('ambiguity_detection', 0):.1%}")

    ar = relatorio.get("avaliacao_relatorio", {})
    if ar and "score_total" in ar:
        print(f"\n  Relatorio (LLM-as-judge):")
        print(f"    Score total                  : {ar.get('score_total', 0):.1f}/10")

    print(f"\n[OK] Relatorio guardado em: {output_path}")
    print("=" * 60)

    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Harness de avaliacao do Retail Vision Intelligence System"
    )
    parser.add_argument("--images-dir", required=True,
                        help="Pasta com imagens de teste")
    parser.add_argument("--ground-truth",
                        help="Ficheiro JSON com ground truth (opcional)")
    parser.add_argument("--output", default="evaluation_report.json",
                        help="Ficheiro de saida (default: evaluation_report.json)")
    parser.add_argument("--strategy", choices=["A", "B", "C", "all"], default="B",
                        help="Estrategia a avaliar. 'all' avalia A, B e C (default: B)")

    args = parser.parse_args()

    executar_avaliacao(
        images_dir=args.images_dir,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        strategy=args.strategy,
    )