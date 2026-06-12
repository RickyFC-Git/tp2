import os
import json
import logging
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

RULES_DIR = Path("data/rules")
RULES_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path("data/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "openai/gpt-4o-mini"

PROMPT_CONVERTER_REGRA = """És um assistente especializado em gestão de lojas de retalho.
O gestor de loja vai escrever uma regra em português natural e tu tens de a converter para JSON estruturado.

REGRA DO GESTOR:
"{regra}"

ZONAS DISPONÍVEIS NA LOJA: Z_S1, Z_S2, Z_S3, Z_S4 (usa null para "todas as zonas")
TIPOS DE ISSUE POSSÍVEIS: empty_shelf, wrong_product, damaged, misaligned, label_missing, other
NÍVEIS DE SEVERIDADE: low, medium, high
LOCALIZAÇÕES NA PRATELEIRA: bottom, middle, top, any

Converte esta regra para o seguinte JSON. Devolve APENAS o JSON estruturado dentro de um bloco de código markdown ```json, sem texto antes ou depois:

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
- Se a regra menciona percentagem de vazio (ex: "30% vazia"), converte para fill_rate_threshold = 0.70 (pois ocupação esperada é 100 - 30 = 70). Se disser ocupação menor que 80%, fill_rate_threshold = 80.0 ou 0.80.
- Se menciona "prateleira inferior/média/superior", usa location_filter = "bottom"/"middle"/"top"
- Se menciona urgência ("crítico", "imediatamente"), usa alert_level = "critical"
- Se menciona "avisa mas não é urgente", usa alert_level = "info"
- Se a zona não é especificada, zone_filter = null (aplica a todas)
- Se o horário não é especificado, time_filter = null
- Se algo for AMBÍGUO ou em falta para poder mapear os campos estruturados com certeza, lista em "ambiguities" e coloca is_valid = false
- Em "assumptions" lista o que assumiste quando a regra não era explícita"""

PROMPT_VERIFICAR_AMBIGUIDADES = """O gestor de loja escreveu esta regra:
"{regra}"

Esta regra foi convertida para JSON mas tem as seguintes ambiguidades identificadas:
{ambiguidades}

Gera uma mensagem clara e amigável em português para o gestor, explaining cada ambiguidade
e fazendo perguntas específicas para as resolver. Sê conciso e direto.
Devolve APENAS o texto da mensagem, sem formatação especial."""

def gerar_rule_id() -> str:
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
    ficheiro = RULES_DIR / f"{regra['rule_id']}.json"
    ficheiro.write_text(json.dumps(regra, ensure_ascii=False, indent=2), encoding="utf-8")
    return ficheiro

def carregar_regra(rule_id: str) -> Optional[dict]:
    ficheiro = RULES_DIR / f"{rule_id}.json"
    if not ficheiro.exists():
        return None
    return json.loads(ficheiro.read_text(encoding="utf-8"))

def carregar_todas_regras() -> list[dict]:
    regras = []
    for ficheiro in sorted(RULES_DIR.glob("RULE_*.json")):
        regras.append(json.loads(ficheiro.read_text(encoding="utf-8")))
    return regras

def eliminar_regra(rule_id: str) -> bool:
    ficheiro = RULES_DIR / f"{rule_id}.json"
    if not ficheiro.exists():
        return False
    ficheiro.unlink()
    return True

def chamar_openrouter(prompt: str, max_retries: int = 3) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("A variável de ambiente OPENROUTER_API_KEY não está configurada no ficheiro .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                res_json = response.json()
                return res_json["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429 and attempt < max_retries - 1:
                wait = (2 ** attempt) * 5
                log.warning(f"Rate limit detetado (429). A aguardar {wait}s...")
                time.sleep(wait)
            else:
                response.raise_for_status()
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Falha ao comunicar com o OpenRouter após {max_retries} tentativas: {e}")
            time.sleep(2)
    return ""

def parse_json_limpo(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        linhas = raw.split("\n")
        if linhas[0].startswith("```"):
            linhas = linhas[1:]
        if linhas and linhas[-1].strip() == "```":
            linhas = linhas[:-1]
        raw = "\n".join(linhas)
    return json.loads(raw.strip())

def converter_regra(texto_natural: str) -> dict:
    rule_id = gerar_rule_id()
    timestamp = datetime.now(timezone.utc).isoformat()

    prompt = PROMPT_CONVERTER_REGRA.format(
        regra=texto_natural,
        rule_id=rule_id,
        timestamp=timestamp,
    )

    try:
        raw = chamar_openrouter(prompt)
        regra = parse_json_limpo(raw)
    except Exception as e:
        log.error(f"Erro ao processar resposta do LLM: {e}")
        regra = {
            "rule_id": rule_id,
            "created_at": timestamp,
            "natural_language": texto_natural,
            "description": "Erro na conversão automática via OpenRouter",
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
                "ambiguities": [f"Erro interno na conversão: {str(e)}. Tenta reformular."],
                "assumptions": [],
            }
        }

    regra.setdefault("rule_id", rule_id)
    regra.setdefault("created_at", timestamp)
    regra.setdefault("natural_language", texto_natural)

    return regra

def gerar_mensagem_ambiguidades(texto_natural: str, ambiguidades: list[str]) -> str:
    if not ambiguidades:
        return ""

    prompt = PROMPT_VERIFICAR_AMBIGUIDADES.format(
        regra=texto_natural,
        ambiguidades="\n".join(f"- {a}" for a in ambiguidades),
    )

    try:
        return chamar_openrouter(prompt)
    except Exception as e:
        log.error(f"Erro ao gerar mensagem de ambiguidades: {e}")
        ambigs_formatadas = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(ambiguidades))
        return (
            f"A regra tem as seguintes ambiguidades que precisam de esclarecimento:\n"
            f"{ambigs_formatadas}\n"
            f"Por favor, reformula a regra com mais detalhe."
        )

def verificar_condicoes(regra: dict, inspecao: dict) -> tuple[bool, str]:
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
        fill_rate_atual = inspecao.get("shelf_fill_rate", 100.0)
        
        thresh = fill_rate_threshold * 100 if fill_rate_threshold <= 1.0 else fill_rate_threshold
        atual = fill_rate_atual * 100 if fill_rate_atual <= 1.0 else fill_rate_atual
        
        if atual < thresh:
            disparou = True
            motivos.append(f"Ocupação atual ({atual:.1f}%) abaixo do limiar ({thresh:.1f}%)")

    issue_types = cond.get("issue_types", [])
    if issue_types:
        issues_inspecao = [i.get("type") for i in inspecao.get("issues", [])]
        tipos_encontrados = [t for t in issue_types if t in issues_inspecao]
        if tipos_encontrados:
            disparou = True
            motivos.append(f"Anomalias encontradas do tipo: {', '.join(tipos_encontrados)}")

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
            motivos.append(f"{len(issues_graves)} anomalia(s) com severidade igual/superior a '{severity_threshold}'")

    location_filter = cond.get("location_filter", "any")
    if location_filter != "any" and disparou:
        mapa_loc = {
            "bottom": ["inferior", "baixo", "bottom", "fundo"],
            "middle": ["central", "meio", "middle", "média"],
            "top": ["superior", "cima", "top", "topo"],
        }
        termos = mapa_loc.get(location_filter, [])
        issues_na_loc = [
            i for i in inspecao.get("issues", [])
            if any(t in i.get("location", "").lower() for t in termos)
        ]
        if not issues_na_loc:
            disparou = False
            motivos.append(f"Nenhum problema localizado na secção '{location_filter}'")

    if disparou:
        return True, " | ".join(motivos)
    return False, "Nenhuma condição de ativação cumprida."

def gerar_notificacao(regra: dict, inspecao: dict, motivo: str) -> dict:
    template = regra.get("action", {}).get(
        "notification_message",
        "Regra {rule_id} disparada na zona {zone_id}"
    )

    mensagem = template
    placeholders = {
        "{zone_id}": str(inspecao.get("zone_id", "")),
        "{fill_rate}": f"{inspecao.get('shelf_fill_rate', 0)}%",
        "{issue_count}": str(len(inspecao.get("issues", []))),
        "{timestamp}": str(inspecao.get("timestamp", "")),
        "{rule_id}": str(regra.get("rule_id", ""))
    }
    for ph, val in placeholders.items():
        mensagem = mensagem.replace(ph, val)

    return {
        "alert_id": f"ALT_{inspecao.get('inspection_id', 'UNKNOWN')}_{regra.get('rule_id')}",
        "rule_id": regra.get("rule_id"),
        "alert_level": regra.get("action", {}).get("alert_level", "warning"),
        "mensagem": mensagem,
        "motivo_disparo": motivo,
        "inspection_id": inspecao.get("inspection_id"),
        "zone_id": inspecao.get("zone_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def executar_regras(inspecao: dict) -> list[dict]:
    regras = carregar_todas_regras()
    regras_validas = [r for r in regras if r.get("validation", {}).get("is_valid", False)]

    log_entries = []
    notificacoes = []

    log.info(f"A avaliar {len(regras_validas)} regras para a inspeção {inspecao.get('inspection_id')}")

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
            log.info(f"   REGRA {rule_id} ATIVADA: {motivo}")
            notificacao = gerar_notificacao(regra, inspecao, motivo)
            notificacoes.append(notificacao)

    if log_entries:
        log_file = LOGS_DIR / f"rules_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_file.write_text(json.dumps({
            "inspection_id": inspecao.get("inspection_id"),
            "total_regras_verificadas": len(regras_validas),
            "total_disparadas": len(notificacoes),
            "entries": log_entries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"Processamento concluído. {len(notificacoes)} alerta(s) emitido(s).")
    return notificacoes

def cmd_add(texto: str):
    print(f"\nA converter a regra do gestor: '{texto}'")
    print("A enviar para o OpenRouter...")

    regra = converter_regra(texto)
    ambiguidades = regra.get("validation", {}).get("ambiguities", [])

    if ambiguidades:
        print("\nAtenção: A regra submetida possui ambiguidades:")
        msg = gerar_mensagem_ambiguidades(texto, ambiguidades)
        print(f"\n[Resposta do Sistema]:\n{msg}")
        print("\nRegra guardada com o estado INVÁLIDA. O gestor necessita de a clarificar.")
        regra["validation"]["is_valid"] = False
    else:
        regra["validation"]["is_valid"] = True
        print("\nRegra convertida e validada com sucesso:")
        print(f"  Descrição formal: {regra.get('description')}")
        print(f"  Nível do Alerta: {regra.get('action', {}).get('alert_level').upper()}")
        if regra.get("validation", {}).get("assumptions"):
            print(f"  Pressupostos assumidos pela IA:")
            for a in regra["validation"]["assumptions"]:
                print(f"    - {a}")

    ficheiro = guardar_regra(regra)
    print(f"Ficheiro de configuração guardado em: {ficheiro}")
    return regra

def cmd_list():
    regras = carregar_todas_regras()
    if not regras:
        print("Nenhuma regra configurada no sistema.")
        return

    print(f"\n{'='*75}")
    print(f"{'ID':<12} {'ESTADO':<10} {'ALERTA':<12} DESCRIÇÃO COMPLETA")
    print(f"{'='*75}")
    for r in regras:
        valida = "VÁLIDA" if r.get("validation", {}).get("is_valid") else "INVÁLIDA"
        alert = r.get("action", {}).get("alert_level", "warning").upper()
        desc = r.get("description", r.get("natural_language", ""))[:45]
        print(f"{r['rule_id']:<12} {valida:<10} {alert:<12} {desc}...")
    print(f"{'='*75}")
    print(f"Total: {len(regras)} regras encontradas em {RULES_DIR}\n")

def cmd_delete(rule_id: str):
    if eliminar_regra(rule_id):
        print(f"Regra {rule_id} removida com sucesso.")
    else:
        print(f"Erro: Regra {rule_id} não encontrada.")

def cmd_test(rule_id: str, inspection_path: str):
    regra = carregar_regra(rule_id)
    if not regra:
        print(f"Erro: Regra {rule_id} não encontrada.")
        return

    path_inspecao = Path(inspection_path)
    if not path_inspecao.exists():
        print(f"Erro: Ficheiro de inspeção não encontrado em: {inspection_path}")
        return

    inspecao = json.loads(path_inspecao.read_text(encoding="utf-8"))
    disparou, motivo = verificar_condicoes(regra, inspecao)

    print(f"\n=== Teste de Execução da Regra: {rule_id} ===")
    print(f"Ficheiro de Inspeção: {inspecao.get('inspection_id')}")
    print(f"Zona Alvo: {inspecao.get('zone_id')} | Ocupação Prateleira: {inspecao.get('shelf_fill_rate')}%")
    print(f"Contagem de Anomalias: {len(inspecao.get('issues', []))}")
    print(f"{'-'*45}")
    print(f"RESULTADO: {'DISPAROU ALERTA' if disparou else 'CONFORME (Não disparou)'}")
    print(f"Motivo Técnico: {motivo}")

    if disparou:
        notif = gerar_notificacao(regra, inspecao, motivo)
        print(f"\n[Notificação Gerada]:")
        print(f"  Nível: [{notif['alert_level'].upper()}]")
        print(f"  Mensagem de Saída: {notif['mensagem']}")

def cmd_show(rule_id: str):
    regra = carregar_regra(rule_id)
    if not hasattr(regra, "get") or not regra:
        print(f"Erro: Regra {rule_id} não encontrada.")
        return
    print(json.dumps(regra, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rule Engine — Motor de Regras em Linguagem Natural (OpenRouter)")
    sub = parser.add_subparsers(dest="cmd")

    p_add = sub.add_parser("add", help="Adiciona e traduz uma regra em linguagem natural")
    p_add.add_argument("texto", help="Texto livre enviado pelo gestor de loja")

    sub.add_parser("list", help="Lista todas as regras configuradas")

    p_del = sub.add_parser("delete", help="Elimina uma regra do sistema")
    p_del.add_argument("rule_id", help="ID da regra (ex: RULE_001)")

    p_test = sub.add_parser("test", help="Testa o disparo de uma regra contra uma inspeção")
    p_test.add_argument("rule_id", help="ID da regra a testar")
    p_test.add_argument("--inspection", required=True, help="Caminho para o ficheiro JSON da inspeção")

    p_show = sub.add_parser("show", help="Exibe a estrutura JSON de uma regra")
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