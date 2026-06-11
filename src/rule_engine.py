import os
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

RULES_DIR = Path("data/rules")
RULES_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path("data/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROMPT_CONVERTER_REGRA = """És um assistente especializado em gestão de lojas de retalho.
O gestor de loja vai escrever uma regra em português natural e tu tens de a converter para JSON estruturado.

REGRA DO GESTOR:
"{regra}"

ZONAS DISPONÍVEIS NA LOJA: Z_S1, Z_S2, Z_S3, Z_S4 (usa null para "todas as zonas")
TIPOS DE ISSUE POSSÍVEIS: empty_shelf, wrong_product, damaged, misaligned, label_missing, other
NÍVEIS DE SEVERIDADE: low, medium, high
LOCALIZAÇÕES NA PRATELEIRA: bottom, middle, top, any

Converte esta regra para o seguinte JSON. Devolve APENAS o JSON, sem texto antes ou depois:

{{
  "rule_id": "{rule_id}",
  "created_at": "{timestamp}",
  "natural_language": "{regra}",
  "description": "reformulação clara e inequívoca em português formal",
  "conditions": {{
    "zone_filter": null,
    "time_filter": null,
    "issue_types": [],
    "severity_threshold": null,
    "fill_rate_threshold": null,
    "location_filter": "any"
  }},
  "action": {{
    "alert_level": "info|warning|critical",
    "notification_message": "template da mensagem quando a regra dispara. Usa {{zone_id}}, {{fill_rate}}, {{issue_count}}, {{timestamp}} como placeholders"
  }},
  "validation": {{
    "is_valid": true,
    "ambiguities": [],
    "assumptions": []
  }}
}}

REGRAS DE CONVERSÃO:
- Se a regra menciona percentagem de vazio (ex: "30% vazia"), converte para fill_rate_threshold = 1 - percentagem (ex: 0.70)
- Se menciona "prateleira inferior/média/superior", usa location_filter = "bottom"/"middle"/"top"
- Se menciona urgência ("crítico", "imediatamente"), usa alert_level = "critical"
- Se menciona "avisa mas não é urgente", usa alert_level = "info"
- Se a zona não é especificada, zone_filter = null (aplica a todas)
- Se o horário não é especificado, time_filter = null
- Se algo for AMBÍGUO ou em falta, lista em "ambiguities" e coloca is_valid = false
- Em "assumptions" lista o que assumiste quando a regra não era explícita"""

PROMPT_VERIFICAR_AMBIGUIDADES = """O gestor de loja escreveu esta regra:
"{regra}"

Esta regra foi convertida para JSON mas tem as seguintes ambiguidades identificadas:
{ambiguidades}

Gera uma mensagem clara e amigável em português para o gestor, explicando cada ambiguidade
e fazendo perguntas específicas para as resolver. Sê conciso e direto.
Devolve APENAS o texto da mensagem, sem formatação especial."""

def gerar_rule_id() -> str:
    """Gera um ID único para a regra baseado nas regras existentes."""
    existentes = list(RULES_DIR.glob("RULE_*.json"))
    if not existentes:
        return "RULE_001"
    numeros = []
    for f in existentes:
        try:
            numeros.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    proximo = max(numeros) + 1 if numeros else 1
    return f"RULE_{proximo:03d}"

def guardar_regra(regra: dict) -> Path:
    """Guarda uma regra em disco."""
    ficheiro = RULES_DIR / f"{regra['rule_id']}.json"
    with open(ficheiro, "w", encoding="utf-8") as f:
        json.dump(regra, f, ensure_ascii=False, indent=2)
    return ficheiro

def carregar_regra(rule_id: str) -> Optional[dict]:
    """Carrega uma regra do disco."""
    ficheiro = RULES_DIR / f"{rule_id}.json"
    if not ficheiro.exists():
        return None
    with open(ficheiro, encoding="utf-8") as f:
        return json.load(f)

def carregar_todas_regras() -> list[dict]:
    """Carrega todas as regras persistidas."""
    regras = []
    for ficheiro in sorted(RULES_DIR.glob("RULE_*.json")):
        with open(ficheiro, encoding="utf-8") as f:
            regras.append(json.load(f))
    return regras

def eliminar_regra(rule_id: str) -> bool:
    """Elimina uma regra do disco. Devolve True se sucesso."""
    ficheiro = RULES_DIR / f"{rule_id}.json"
    if not ficheiro.exists():
        return False
    ficheiro.unlink()
    return True

def chamar_gemini(prompt: str, max_retries: int = 3) -> str:
    """Chama o Gemini Flash com retry em caso de erro 429."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0},
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = (2 ** attempt) * 10
                log.warning(f"Rate limit. Aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise
    return ""

def parse_json_limpo(raw: str) -> dict:
    """Remove markdown e faz parse do JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        linhas = raw.split("\n")
        raw = "\n".join(linhas[1:-1] if linhas[-1].strip() == "```" else linhas[1:])
    return json.loads(raw.strip())

def converter_regra(texto_natural: str) -> dict:
    """
    Converte uma regra em linguagem natural para JSON estruturado.
    Devolve o dict da regra (com is_valid=False se tiver ambiguidades).
    """
    rule_id = gerar_rule_id()
    timestamp = datetime.now(timezone.utc).isoformat()

    prompt = PROMPT_CONVERTER_REGRA.format(
        regra=texto_natural,
        rule_id=rule_id,
        timestamp=timestamp,
    )

    try:
        raw = chamar_gemini(prompt)
        regra = parse_json_limpo(raw)
    except json.JSONDecodeError as e:
        log.error(f"Gemini devolveu JSON inválido: {e}")
        regra = {
            "rule_id": rule_id,
            "created_at": timestamp,
            "natural_language": texto_natural,
            "description": "Erro na conversão automática",
            "conditions": {
                "zone_filter": None,
                "time_filter": None,
                "issue_types": [],
                "severity_threshold": None,
                "fill_rate_threshold": None,
                "location_filter": "any",
            },
            "action": {
                "alert_level": "warning",
                "notification_message": texto_natural,
            },
            "validation": {
                "is_valid": False,
                "ambiguities": ["Erro interno na conversão. Tenta reformular a regra."],
                "assumptions": [],
            },
            "_parse_error": True,
        }

    regra.setdefault("rule_id", rule_id)
    regra.setdefault("created_at", timestamp)
    regra.setdefault("natural_language", texto_natural)

    return regra

def gerar_mensagem_ambiguidades(texto_natural: str, ambiguidades: list[str]) -> str:
    """Gera mensagem amigável para o gestor resolver ambiguidades."""
    if not ambiguidades:
        return ""

    prompt = PROMPT_VERIFICAR_AMBIGUIDADES.format(
        regra=texto_natural,
        ambiguidades="\n".join(f"- {a}" for a in ambiguidades),
    )

    try:
        return chamar_gemini(prompt)
    except Exception as e:
        log.error(f"Erro ao gerar mensagem de ambiguidades: {e}")
        ambigs_formatadas = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(ambiguidades))
        return (
            f"A regra tem as seguintes ambiguidades que precisam de esclarecimento:\n"
            f"{ambigs_formatadas}\n"
            f"Por favor, reformula a regra com mais detalhe."
        )

def verificar_condicoes(regra: dict, inspecao: dict) -> tuple[bool, str]:
    """
    Verifica se uma regra dispara face a um resultado de inspeção.
    Devolve (disparou: bool, motivo: str).
    """
    cond = regra.get("conditions", {})
    disparou = False
    motivos = []

    zone_filter = cond.get("zone_filter")
    if zone_filter:
        zonas = zone_filter if isinstance(zone_filter, list) else [zone_filter]
        if inspecao.get("zone_id") not in zonas:
            return False, f"Zona {inspecao.get('zone_id')} não está no filtro {zonas}"

    time_filter = cond.get("time_filter")
    if time_filter and isinstance(time_filter, dict):
        hora_start = time_filter.get("hours_start")
        hora_end = time_filter.get("hours_end")
        if hora_start is not None and hora_end is not None:
            try:
                hora_atual = datetime.fromisoformat(
                    inspecao.get("timestamp", datetime.now().isoformat())
                ).hour
                if not (hora_start <= hora_atual < hora_end):
                    return False, f"Hora {hora_atual}h fora do intervalo [{hora_start}h-{hora_end}h]"
            except Exception:
                pass

    fill_rate_threshold = cond.get("fill_rate_threshold")
    if fill_rate_threshold is not None:
        fill_rate_atual = inspecao.get("shelf_fill_rate", 1.0)
        if fill_rate_atual < fill_rate_threshold:
            disparou = True
            motivos.append(
                f"fill_rate {fill_rate_atual:.0%} abaixo do limiar {fill_rate_threshold:.0%}"
            )

    issue_types = cond.get("issue_types", [])
    if issue_types:
        issues_inspecao = [i.get("type") for i in inspecao.get("issues", [])]
        tipos_encontrados = [t for t in issue_types if t in issues_inspecao]
        if tipos_encontrados:
            disparou = True
            motivos.append(f"issues encontrados: {', '.join(tipos_encontrados)}")

    severity_threshold = cond.get("severity_threshold")
    if severity_threshold:
        ordem_severidade = {"low": 1, "medium": 2, "high": 3}
        limiar = ordem_severidade.get(severity_threshold, 0)
        issues_graves = [
            i for i in inspecao.get("issues", [])
            if ordem_severidade.get(i.get("severity", "low"), 0) >= limiar
        ]
        if issues_graves:
            disparou = True
            motivos.append(
                f"{len(issues_graves)} issue(s) com severidade >= {severity_threshold}"
            )

    location_filter = cond.get("location_filter", "any")
    if location_filter != "any" and disparou:
        mapa_loc = {
            "bottom": ["inferior", "baixo", "bottom"],
            "middle": ["central", "meio", "middle"],
            "top": ["superior", "cima", "top"],
        }
        termos = mapa_loc.get(location_filter, [])
        issues_na_loc = [
            i for i in inspecao.get("issues", [])
            if any(t in i.get("location", "").lower() for t in termos)
        ]
        if not issues_na_loc and location_filter != "any":
            disparou = False
            motivos.append(f"Nenhum issue na localização '{location_filter}'")

    if disparou:
        return True, " | ".join(motivos)
    return False, "Nenhuma condição satisfeita"

def gerar_notificacao(regra: dict, inspecao: dict, motivo: str) -> dict:
    """Gera a notificação quando uma regra dispara."""
    template = regra.get("action", {}).get(
        "notification_message",
        "Regra {rule_id} disparada na zona {zone_id}"
    )

    mensagem = template.format(
        rule_id=regra.get("rule_id", ""),
        zone_id=inspecao.get("zone_id", ""),
        fill_rate=f"{inspecao.get('shelf_fill_rate', 0):.0%}",
        issue_count=len(inspecao.get("issues", [])),
        timestamp=inspecao.get("timestamp", ""),
        inspection_id=inspecao.get("inspection_id", ""),
    )

    return {
        "rule_id": regra.get("rule_id"),
        "alert_level": regra.get("action", {}).get("alert_level", "warning"),
        "mensagem": mensagem,
        "motivo_disparo": motivo,
        "inspection_id": inspecao.get("inspection_id"),
        "zone_id": inspecao.get("zone_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def executar_regras(inspecao: dict) -> list[dict]:
    """
    Executa todas as regras guardadas contra uma inspeção.
    Produz logs e devolve lista de notificações geradas.
    """
    regras = carregar_todas_regras()
    regras_validas = [r for r in regras if r.get("validation", {}).get("is_valid", False)]

    log_entries = []
    notificacoes = []

    log.info(f"A executar {len(regras_validas)} regras para {inspecao.get('inspection_id')}")

    for regra in regras_validas:
        rule_id = regra.get("rule_id")
        disparou, motivo = verificar_condicoes(regra, inspecao)

        entry = {
            "rule_id": rule_id,
            "inspection_id": inspecao.get("inspection_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "disparou": disparou,
            "motivo": motivo,
        }
        log_entries.append(entry)

        if disparou:
            log.info(f"   {rule_id} DISPAROU: {motivo}")
            notificacao = gerar_notificacao(regra, inspecao, motivo)
            notificacoes.append(notificacao)
        else:
            log.debug(f"  - {rule_id} não disparou: {motivo}")

    if log_entries:
        log_file = LOGS_DIR / f"rules_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "inspection_id": inspecao.get("inspection_id"),
                "total_regras_verificadas": len(regras_validas),
                "total_disparadas": len(notificacoes),
                "entries": log_entries,
            }, f, ensure_ascii=False, indent=2)

    log.info(f"Execução concluída: {len(notificacoes)}/{len(regras_validas)} regras dispararam")
    return notificacoes

def cmd_add(texto: str):
    """Adiciona uma nova regra."""
    print(f"\nA converter regra: '{texto}'")
    print("A chamar Gemini...")

    regra = converter_regra(texto)
    ambiguidades = regra.get("validation", {}).get("ambiguities", [])

    if ambiguidades:
        print("\n A regra tem ambiguidades:")
        msg = gerar_mensagem_ambiguidades(texto, ambiguidades)
        print(msg)
        print("\nA regra foi guardada como INVÁLIDA. Reformula e adiciona novamente.")
        regra["validation"]["is_valid"] = False
    else:
        regra["validation"]["is_valid"] = True
        print("\n Regra convertida com sucesso:")
        print(f"  Descrição: {regra.get('description')}")
        print(f"  Alert level: {regra.get('action', {}).get('alert_level')}")
        if regra.get("validation", {}).get("assumptions"):
            print(f"  Pressupostos assumidos:")
            for a in regra["validation"]["assumptions"]:
                print(f"    - {a}")

    ficheiro = guardar_regra(regra)
    print(f"\n Regra guardada: {ficheiro}")
    return regra

def cmd_list():
    """Lista todas as regras."""
    regras = carregar_todas_regras()
    if not regras:
        print("Nenhuma regra definida.")
        return

    print(f"\n{'='*60}")
    print(f"{'ID':<12} {'VÁLIDA':<8} {'ALERT':<10} DESCRIÇÃO")
    print(f"{'='*60}")
    for r in regras:
        valida = "" if r.get("validation", {}).get("is_valid") else ""
        alert = r.get("action", {}).get("alert_level", "?")
        desc = r.get("description", r.get("natural_language", ""))[:45]
        print(f"{r['rule_id']:<12} {valida:<8} {alert:<10} {desc}")
    print(f"{'='*60}")
    print(f"Total: {len(regras)} regras")

def cmd_delete(rule_id: str):
    """Elimina uma regra."""
    if eliminar_regra(rule_id):
        print(f" Regra {rule_id} eliminada.")
    else:
        print(f" Regra {rule_id} não encontrada.")

def cmd_test(rule_id: str, inspection_path: str):
    """Testa uma regra contra uma inspeção existente."""
    regra = carregar_regra(rule_id)
    if not regra:
        print(f" Regra {rule_id} não encontrada.")
        return

    if not Path(inspection_path).exists():
        print(f" Ficheiro de inspeção não encontrado: {inspection_path}")
        return

    with open(inspection_path, encoding="utf-8") as f:
        inspecao = json.load(f)

    disparou, motivo = verificar_condicoes(regra, inspecao)

    print(f"\nTeste da regra {rule_id}")
    print(f"Inspeção: {inspecao.get('inspection_id')}")
    print(f"Zona: {inspecao.get('zone_id')}")
    print(f"Fill rate: {inspecao.get('shelf_fill_rate', 'N/A')}")
    print(f"Issues: {len(inspecao.get('issues', []))}")
    print(f"\nResultado: {'DISPAROU' if disparou else 'Não disparou'}")
    print(f"Motivo: {motivo}")

    if disparou:
        notif = gerar_notificacao(regra, inspecao, motivo)
        print(f"\nNotificação gerada:")
        print(f"  [{notif['alert_level'].upper()}] {notif['mensagem']}")

def cmd_show(rule_id: str):
    """Mostra os detalhes de uma regra."""
    regra = carregar_regra(rule_id)
    if not regra:
        print(f" Regra {rule_id} não encontrada.")
        return
    print(json.dumps(regra, ensure_ascii=False, indent=2))

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rule Engine — Motor de regras em linguagem natural")
    sub = parser.add_subparsers(dest="cmd")

    # add
    p_add = sub.add_parser("add", help="Adiciona uma nova regra")
    p_add.add_argument("texto", help="Regra em linguagem natural")

    # list
    sub.add_parser("list", help="Lista todas as regras")

    # delete
    p_del = sub.add_parser("delete", help="Elimina uma regra")
    p_del.add_argument("rule_id", help="ID da regra (ex: RULE_001)")

    # test
    p_test = sub.add_parser("test", help="Testa uma regra contra uma inspeção")
    p_test.add_argument("rule_id", help="ID da regra")
    p_test.add_argument("--inspection", required=True, help="Caminho para o JSON de inspeção")

    # show
    p_show = sub.add_parser("show", help="Mostra detalhes de uma regra")
    p_show.add_argument("rule_id", help="ID da regra")

    args = parser.parse_args()

    if args.cmd == "add":
        cmd_add(args.texto)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "delete":
        cmd_delete(args.rule_id)
    elif args.cmd == "test":
        cmd_test(args.rule_id, args.inspection)
    elif args.cmd == "show":
        cmd_show(args.rule_id)
    else:
        parser.print_help()