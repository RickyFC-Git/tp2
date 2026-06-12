import os
import json
import time
import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Adiciona a raiz e a pasta src ao path por segurança do interpretador
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ==============================================================================
# IMPORTAÇÃO SEGURA DOS MÓDULOS DENTRO DE 'src'
# ==============================================================================
try:
    from src import shelf_inspector
    from src import rule_engine
    from src import rag_memory
    from src import report_generator
    COMPONENTES_OK = True
except ImportError as e:
    log.warning(f"Tentativa 1 (from src import) falhou: {e}. A tentar import direto...")
    try:
        import shelf_inspector
        import rule_engine
        import rag_memory
        import report_generator
        COMPONENTES_OK = True
    except ImportError as e2:
        log.error(f"Erro crítico ao importar componentes de 'src': {e2}")
        COMPONENTES_OK = False

# ==============================================================================
# CONFIGURAÇÃO UNIFICADA DA API (OPENROUTER + GPT-4O-MINI)
# ==============================================================================
from openai import OpenAI

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    log.warning("Aviso: OPENROUTER_API_KEY nao encontrada no .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/teu-utilizador/tp2-liacd",
        "X-Title": "Retail Vision Intelligence System - Evaluation",
    }
)

MODELO_JUDGE = "openai/gpt-4o-mini"
# ==============================================================================

REGRAS_TESTE = [
    ("Avisa-me quando a prateleira inferior estiver mais de 30% vazia", True, False),
    ("Na zona Z_S1, se nao houver produtos de laticinios visiveis, e critico", True, False),
    ("Quando o fill rate cair abaixo de 60% entre as 10h e as 13h, avisa-me", True, False),
    ("Se um produto estiver tombado, considera sempre severidade alta", True, False),
    ("Avisa-me quando a prateleira estiver vazia", False, True),
    ("Notifica-me se houver problemas", False, True),
    ("Alerta para situacoes na loja", False, True),
]


def chamar_judge(prompt: str) -> dict:
    """Chama o gpt-4o-mini via OpenRouter como juiz e faz parse do JSON."""
    try:
        response = client.chat.completions.create(
            model=MODELO_JUDGE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"} 
        )
        raw = response.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            linhas = raw.split("\n")
            if linhas[0].strip().startswith("```"):
                linhas = linhas[1:]
            if linhas and linhas[-1].strip() == "```":
                linhas = lines[:-1]
            raw = "\n".join(linhas).strip()
            
        return json.loads(raw)
    except Exception as e:
        log.error(f"Erro no LLM-as-judge ({MODELO_JUDGE}): {e}")
        return {"score": 0.0, "erro": str(e)}


def avaliar_motor_regras() -> dict:
    """Avalia a capacidade do rule_engine de converter e validar regras."""
    log.info("A avaliar Motor de Regras (NLP para JSON)...")
    sucessos_parse = 0
    sucessos_ambiguidade = 0
    total_validas = 0
    total_ambiguas = 0

    re_m = sys.modules.get('src.rule_engine') or sys.modules.get('rule_engine')

    for texto, esperada_valida, esperada_ambigua in REGRAS_TESTE:
        if esperada_valida:
            total_validas += 1
        if esperada_ambigua:
            total_ambiguas += 1

        try:
            regra_json = re_m.converter_regra_nlp(texto)
            if regra_json:
                is_ambiguous = regra_json.get("metadata", {}).get("is_ambiguous", False)
                if esperada_valida and not is_ambiguous:
                    sucessos_parse += 1
                if esperada_ambigua and is_ambiguous:
                    sucessos_ambiguidade += 1
        except Exception as e:
            log.error(f"Erro ao testar regra '{texto}': {e}")

    return {
        "rule_parse_rate": sucessos_parse / total_validas if total_validas > 0 else 1.0,
        "rule_correctness": sucessos_parse / total_validas if total_validas > 0 else 1.0,
        "ambiguity_detection": sucessos_ambiguidade / total_ambiguas if total_ambiguas > 0 else 1.0,
    }


def avaliar_visao_e_regras(images_dir: str, ground_truth_path: Optional[str]) -> tuple[dict, list]:
    """Avalia o shelf_inspector e o disparo de regras usando imagens."""
    log.info(f"A avaliar Visao Computacional em: {images_dir}")
    pasta = Path(images_dir)
    imagens = list(pasta.glob("*.jpg")) + list(pasta.glob("*.jpeg"))
    imagens = sorted(imagens)[:10]

    gt_dict = {}
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path, encoding="utf-8") as f:
            dados_gt = json.load(f)
            for item in dados_gt:
                nome_fich = Path(item["image_path"]).name
                gt_dict[nome_fich] = item
        log.info(f"Ground truth carregado com {len(gt_dict)} anotacoes.")
    else:
        log.info("Sem ficheiro de Ground Truth valido. Algumas metricas ficarao a N/A.")

    total_inspecionadas = 0
    verdadeiros_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0
    verdadeiros_negativos = 0
    tem_gt = len(gt_dict) > 0

    historico_inspecoes = []
    
    si_m = sys.modules.get('src.shelf_inspector') or sys.modules.get('shelf_inspector')

    for img_path in imagens:
        log.info(f"A processar imagem: {img_path.name}")
        try:
            res = si_m.inspect_shelf(str(img_path), strategy="B", zone_id="Z_S1")
            historico_inspecoes.append(res)
            total_inspecionadas += 1

            if tem_gt:
                gt_item = gt_dict.get(img_path.name, {"issues": []})
                gt_tem_issues = len(gt_item.get("issues", [])) > 0
                pred_tem_issues = len(res.get("issues", [])) > 0

                if gt_tem_issues and pred_tem_issues:
                    verdadeiros_positivos += 1
                elif not gt_tem_issues and pred_tem_issues:
                    falsos_positivos += 1
                elif gt_tem_issues and not pred_tem_issues:
                    falsos_negativos += 1
                else:
                    verdadeiros_negativos += 1

        except Exception as e:
            log.error(f"Erro ao processar {img_path.name}: {e}")

    idr = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos) if (verdadeiros_positivos + falsos_negativos) > 0 else 1.0
    fpr = falsos_positivos / (falsos_positivos + verdadeiros_negativos) if (falsos_positivos + verdadeiros_negativos) > 0 else 0.0

    metricas = {
        "total_images_evaluated": total_inspecionadas,
        "true_detection_rate_idr": idr if tem_gt else "N/A",
        "false_positive_rate_fpr": fpr if tem_gt else "N/A",
    }
    return metricas, historico_inspecoes


def avaliar_relatorio_llm(inspecoes: list) -> dict:
    """Gera um relatório agregado e submete-o ao gpt-4o-mini (Judge) para dar nota."""
    if not inspecoes:
        return {"score_total": 0.0, "reasoning": "Nenhuma inspecao realizada."}

    log.info("A gerar relatorio consolidado para avaliacao do Juiz...")
    
    rg_m = sys.modules.get('src.report_generator') or sys.modules.get('report_generator')
    
    try:
        relatorio_texto = rg_m.gerar_relatorio(
            inspecoes, titulo="Relatorio de Avaliacao Automatizada", guardar=False
        )

        prompt_judge = f"""És um auditor sénior de sistemas de Inteligência Artificial aplicados ao retalho.
A tua tarefa é avaliar a qualidade, clareza, concisão e utilidade prática do relatório executivo gerado pelo nosso sistema.

O relatório foi gerado com base em {len(inspecoes)} inspeções visuais automáticas.

Abaixo está o RELATÓRIO GERADO:
---
{relatorio_texto}
---

Responde OBRIGATORIAMENTE em formato JSON estrito com a seguinte estrutura:
{{
  "score_total": <nota de 0.0 a 10.0 baseado no profissionalismo e utilidade das informações>,
  "clareza": <nota de 0.0 a 10.0>,
  "capacidade_sumario": <nota de 0.0 a 10.0>,
  "reasoning": "<justificação detalhada em português dos pontos fortes e fracos do relatório>"
}}
"""
        resultado_json = chamar_judge(prompt_judge)
        return resultado_json
    except Exception as e:
        log.error(f"Falha ao avaliar o relatorio com o Juiz: {e}")
        return {"score_total": 0.0, "erro": str(e)}


def executar_harness(images_dir: str, ground_truth: Optional[str], output_path: str, strategy: str):
    print("=" * 60)
    print("  TP2 LIACD -- Harness de Avaliacao Unificado (GPT-4o-mini)")
    print("=" * 60)

    if not COMPONENTES_OK:
        print("[ERRO] Componentes do sistema nao carregaram da pasta 'src'.")
        print("       Verifica se os ficheiros estao dentro dessa pasta.")
        return

    start_time = time.time()

    re_metrics = avaliar_motor_regras()
    vision_metrics, lista_inspecoes = avaliar_visao_e_regras(images_dir, ground_truth)
    judge_metrics = avaliar_relatorio_llm(lista_inspecoes)

    execution_time = time.time() - start_time

    relatorio = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuracao": {
            "images_dir": images_dir,
            "ground_truth_file": ground_truth,
            "strategy_evaluated": strategy,
            "model_judge": MODELO_JUDGE
        },
        "metricas_motor_regras": re_metrics,
        "metricas_visao": vision_metrics,
        "avaliacao_relatorio": judge_metrics,
        "tempo_execucao_segundos": round(execution_time, 2)
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)

    print("\n>>> RESULTADOS DA AVALIACAO <<<")
    print(f"  Tempo Total de Execucao: {relatorio['tempo_execucao_segundos']}s")
    print("\n  Visao Computacional:")
    print(f"    Imagens analisadas           : {vision_metrics.get('total_images_evaluated')}")
    print(f"    True Detection Rate (IDR)    : {vision_metrics.get('true_detection_rate_idr')}")
    print(f"    False Positive Rate (FPR)    : {vision_metrics.get('false_positive_rate_fpr')}")

    print("\n  Motor de Regras (NLP):")
    print(f"    Rule Parse Rate              : {re_metrics.get('rule_parse_rate', 0):.1%}")
    print(f"    Rule Correctness             : {re_metrics.get('rule_correctness', 0):.1%}")
    print(f"    Ambiguity Detection          : {re_metrics.get('ambiguity_detection', 0):.1%}")

    if "score_total" in judge_metrics:
        print("\n  Relatorio (LLM-as-judge):")
        print(f"    Score total do Relatorio     : {judge_metrics.get('score_total', 0):.1f}/10")
        print(f"    Critica do Juiz              : {judge_metrics.get('reasoning', '')[:120]}...")

    print(f"\n[OK] Relatorio guardado com sucesso em: {output_path}")
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
                        help="Estrategia de prompting a testar (default: B)")

    args = parser.parse_args()
    
    # Corrigido o erro: alterado de args.images-dir para args.images_dir
    executar_harness(args.images_dir, args.ground_truth, args.output, args.strategy)