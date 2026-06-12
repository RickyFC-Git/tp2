import os
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import requests

from dotenv import load_dotenv

load_dotenv()

# CONFIGURAÇÃO DE MODELO ATUALIZADA PARA GPT-4O-MINI VIA OPENROUTER
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openai/gpt-4o-mini"

REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

INSPECTIONS_DIR = Path("data/inspections")
RULES_DIR = Path("data/rules")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROMPT_SUMARIO = """És um assistente de gestão de loja de retalho. Gera um sumário executivo
de uma sessão de inspeção de prateleiras. Máximo 150 palavras. Linguagem direta e acionável.

DADOS DA SESSÃO:
- Total de inspeções: {total_inspecoes}
- Zonas inspecionadas: {zonas}
- Issues críticos: {criticos}
- Issues warning: {warnings}
- Issues ok: {oks}
- Fill rate médio: {fill_rate_medio:.0%}
- Issues mais frequentes: {issues_frequentes}

Gera APENAS o texto do sumário, sem título nem formatação markdown:"""

PROMPT_RECOMENDACOES = """És um gestor de loja de retalho experiente. Com base nos problemas
detetados nesta sessão de inspeção, gera no máximo 5 recomendações concretas e acionáveis,
ordenadas por urgência (mais urgente primeiro).

PROBLEMAS DETETADOS:
{problemas}

CONTEXTO HISTÓRICO:
{contexto_historico}

Regras:
- Cada recomendação deve ser específica o suficiente para ser executada sem interpretação adicional
- Inclui zona, prateleira e ação concreta
- Ordena por urgência real (crítico > warning > info)
- Máximo 5 recomendações

Devolve APENAS uma lista numerada, sem texto antes ou depois:"""

def carregar_inspecoes_sessao(pasta: str, horas: int = 24) -> list[dict]:
    pasta_path = Path(pasta)
    corte = datetime.now(timezone.utc) - timedelta(hours=horas)
    inspecoes = []

    if not pasta_path.exists():
        log.warning(f"A pasta de sessões '{pasta}' não existe.")
        return inspecoes

    for ficheiro in sorted(pasta_path.glob("*.json")):
        try:
            with open(ficheiro, encoding="utf-8") as f:
                insp = json.load(f)
            if "inspection_id" not in insp:
                continue
            ts = insp.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt < corte:
                        continue
                except Exception:
                    pass
            inspecoes.append(insp)
        except Exception as e:
            log.warning(f"Erro ao carregar {ficheiro.name}: {e}")

    return inspecoes


def carregar_inspecoes_zona(zona_id: str, dias: int = 14) -> list[dict]:
    corte = datetime.now(timezone.utc) - timedelta(days=dias)
    inspecoes = []

    if not INSPECTIONS_DIR.exists():
        return inspecoes

    for ficheiro in sorted(INSPECTIONS_DIR.glob("*.json")):
        try:
            with open(ficheiro, encoding="utf-8") as f:
                insp = json.load(f)
            if insp.get("zone_id") != zona_id:
                continue
            ts = insp.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt < corte:
                        continue
                except Exception:
                    pass
            inspecoes.append(insp)
        except Exception:
            continue

    return inspecoes


def calcular_estatisticas(inspecoes: list[dict]) -> dict:
    if not inspecoes:
        return {}

    zonas = list({i.get("zone_id", "?") for i in inspecoes})
    criticos = sum(1 for i in inspecoes if i.get("overall_status") == "critical")
    warnings = sum(1 for i in inspecoes if i.get("overall_status") == "warning")
    oks = sum(1 for i in inspecoes if i.get("overall_status") == "ok")

    fill_rates = []
    for i in inspecoes:
        fr = i.get("shelf_fill_rate")
        if fr is not None:
            if fr > 1.0:
                fr = fr / 100.0
            fill_rates.append(fr)
            
    fill_rate_medio = sum(fill_rates) / len(fill_rates) if fill_rates else 0

    contagem_issues = {}
    todos_issues = []
    for insp in inspecoes:
        for issue in insp.get("issues", []):
            tipo = issue.get("type", "other")
            contagem_issues[tipo] = contagem_issues.get(tipo, 0) + 1
            todos_issues.append({**issue, "zone_id": insp.get("zone_id"), "inspection_id": insp.get("inspection_id")})

    issues_frequentes = sorted(contagem_issues.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_inspecoes": len(inspecoes),
        "zonas": zonas,
        "criticos": criticos,
        "warnings": warnings,
        "oks": oks,
        "fill_rate_medio": fill_rate_medio,
        "contagem_issues": contagem_issues,
        "issues_frequentes": issues_frequentes,
        "todos_issues": todos_issues,
    }


def agrupar_por_zona(inspecoes: list[dict]) -> dict:
    por_zona = {}
    for insp in inspecoes:
        zona = insp.get("zone_id", "Z_UNKNOWN")
        if zona not in por_zona:
            por_zona[zona] = []
        por_zona[zona].append(insp)
    return por_zona


def executar_regras_sessao(inspecoes: list[dict]) -> list[dict]:
    try:
        from rule_engine import executar_regras
    except ImportError:
        log.warning("rule_engine não disponível. Regras não serão executadas.")
        return []

    todas_notificacoes = []
    for insp in inspecoes:
        notifs = executar_regras(insp)
        todas_notificacoes.extend(notifs)

    return todas_notificacoes


def obter_contexto_historico_zona(zona_id: str, status: str) -> str:
    try:
        from rag_memory import recuperar_contexto
        query = f"problemas históricos na zona {zona_id} {status}"
        chunks = recuperar_contexto(query, strategy="hybrid", top_k=3)
        if not chunks:
            return "Sem histórico anterior disponível para esta zona."

        # Mantém a integridade do texto sem truncamentos arbitrários
        historico = "\n".join([
            f"[{c['metadata'].get('inspection_id')}] {c['metadata'].get('timestamp', '')[:10]}: {c['texto']}"
            for c in chunks
            if c["metadata"].get("zone_id") == zona_id
        ])

        if not historico:
            return "Sem histórico anterior disponível para esta zona."

        return historico
    except Exception as e:
        log.warning(f"RAG não disponível para contexto por zona: {e}")
        return "Sistema RAG não disponível."


def obter_contexto_historico_geral(stats: dict) -> str:
    try:
        from rag_memory import recuperar_contexto
        issues_str = ", ".join([t for t, _ in stats.get("issues_frequentes", [])[:3]])
        query = f"padrões históricos de problemas: {issues_str}"
        chunks = recuperar_contexto(query, strategy="hybrid", top_k=5)
        if not chunks:
            return "Sem histórico disponível."
            
        return "\n".join([
            f"- [{c['metadata'].get('inspection_id')}] {c['metadata'].get('timestamp', '')[:16]} "
            f"zona {c['metadata'].get('zone_id')}: {c['texto']}"
            for c in chunks
        ])
    except Exception as e:
        log.warning(f"RAG não disponível para contexto geral: {e}")
        return "Sistema RAG não disponível."


def chamar_gemini(prompt: str) -> str:
    """Gera texto usando o OpenRouter através do SDK ou Requests com captura de erros."""
    if not OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY não encontrada no ficheiro .env")
        return "[Erro: API Key em falta]"

    # Abordagem Principal: SDK oficial 'openai'
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Retail Vision Report Generator"
            }
        )
        return response.choices[0].message.content.strip()
        
    except ImportError:
        # Fallback caso a biblioteca 'openai' não esteja no ambiente
        log.info("Biblioteca 'openai' não detetada. A executar chamada via requests...")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Retail Vision Report Generator"
        }
        
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            dados = response.json()
            return dados["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.error(f"Erro ao chamar OpenRouter (requests): {e}")
            return "[Erro ao gerar conteúdo via API OpenRouter]"
            
    except Exception as e:
        # Captura erros de rotas/créditos ou modelos inválidos da API para o programa não ir abaixo
        log.error(f"Erro na resposta da API OpenRouter (SDK): {e}")
        return f"[Erro temporário da API: {e}]"


def gerar_sumario_executivo(stats: dict) -> str:
    issues_freq_str = ", ".join([
        f"{tipo} ({count}x)" for tipo, count in stats.get("issues_frequentes", [])[:3]
    ]) or "nenhum"

    prompt = PROMPT_SUMARIO.format(
        total_inspecoes=stats.get("total_inspecoes", 0),
        zonas=", ".join(stats.get("zonas", [])),
        criticos=stats.get("criticos", 0),
        warnings=stats.get("warnings", 0),
        oks=stats.get("oks", 0),
        fill_rate_medio=stats.get("fill_rate_medio", 0),
        issues_frequentes=issues_freq_str,
    )
    return chamar_gemini(prompt)


def gerar_recomendacoes(stats: dict, contexto_historico: str) -> str:
    problemas_str = json.dumps(
        stats.get("todos_issues", [])[:20], 
        ensure_ascii=False,
        indent=2
    )
    prompt = PROMPT_RECOMENDACOES.format(
        problemas=problemas_str,
        contexto_historico=contexto_historico,
    )
    return chamar_gemini(prompt)


def construir_secao_problemas_por_zona(por_zona: dict) -> str:
    linhas = ["## 2. Problemas por Zona\n"]

    if not por_zona:
        linhas.append("_Nenhum problema detetado nesta sessão._\n")
        return "\n".join(linhas)

    for zona_id, inspecoes in sorted(por_zona.items()):
        tem_problemas = any(
            i.get("overall_status") in ("warning", "critical") or i.get("issues")
            for i in inspecoes
        )
        if not tem_problemas:
            continue

        linhas.append(f"### Zona {zona_id}\n")

        for insp in inspecoes:
            status = insp.get("overall_status", "?")
            fill = insp.get("shelf_fill_rate", 0)
            if fill > 1.0:
                fill = fill / 100.0
                
            insp_id = insp.get("inspection_id", "?")
            ts = insp.get("timestamp", "")[:16]

            emoji_status = {"ok": "[OK]", "warning": "[WARNING]", "critical": "[CRITICAL]"}.get(status, "[?]")

            linhas.append(f"**Inspeção** `{insp_id}` — {ts}")
            linhas.append(f"- Estado: {emoji_status} {status.upper()}")
            linhas.append(f"- Fill rate: {fill:.0%}")

            issues = insp.get("issues", [])
            if issues:
                linhas.append("- Issues detetados:")
                for issue in issues:
                    sev_emoji = {"low": "[LOW]", "medium": "[MEDIUM]", "high": "[CRITICAL]"}.get(
                        issue.get("severity", "low"), "[?]"
                    )
                    linhas.append(
                        f"  - {sev_emoji} `{issue.get('type')}` em _{issue.get('location', '?')}_: "
                        f"{issue.get('description', '')} "
                        f"(confiança: {issue.get('confidence', 0):.0%})"
                    )
            else:
                linhas.append("- Sem issues registados")

            contexto = obter_contexto_historico_zona(zona_id, status)
            if contexto and "Sem histórico" not in contexto and "não disponível" not in contexto:
                linhas.append(f"\n> **Contexto histórico:** {contexto}")

            linhas.append("")

    return "\n".join(linhas)


def construir_secao_regras(notificacoes: list[dict]) -> str:
    linhas = ["## 3. Regras Disparadas\n"]

    if not notificacoes:
        linhas.append("_Nenhuma regra disparou nesta sessão._\n")
        return "\n".join(linhas)

    linhas.append(f"**Total de alertas gerados:** {len(notificacoes)}\n")

    por_nivel = {}
    for notif in notificacoes:
        nivel = notif.get("alert_level", "info")
        por_nivel.setdefault(nivel, []).append(notif)

    for nivel in ["critical", "warning", "info"]:
        if nivel not in por_nivel:
            continue
        emoji = {"critical": "[CRITICAL]", "warning": "[WARNING]", "info": "[INFO]"}.get(nivel, "")
        linhas.append(f"### {emoji} {nivel.upper()} ({len(por_nivel[nivel])} alertas)\n")

        for notif in por_nivel[nivel]:
            linhas.append(f"**{notif.get('rule_id')}** — Zona {notif.get('zone_id')}")
            linhas.append(f"- Mensagem: {notif.get('mensagem')}")
            linhas.append(f"- Motivo: _{notif.get('motivo_disparo')}_")
            linhas.append(f"- Inspeção: `{notif.get('inspection_id')}`")
            linhas.append("")

    return "\n".join(linhas)


def construir_secao_historico(contexto_geral: str, stats: dict) -> str:
    linhas = ["## 4. Contexto Histórico Relevante\n"]

    if not contexto_geral or "não disponível" in contexto_geral.lower():
        linhas.append("_Sistema RAG sem dados históricos suficientes para esta sessão._\n")
        return "\n".join(linhas)

    linhas.append("Padrões históricos recuperados da base de dados de inspeções anteriores:\n")
    linhas.append(contexto_geral)
    linhas.append("")

    return "\n".join(linhas)


def construir_secao_trajetoria() -> str:
    linhas = ["## 6. Integração com Dados de Trajectória\n"]

    try:
        from trajectory_integration import obter_correlacao
        linhas.append(obter_correlacao())
    except ImportError:
        linhas.append(
            "_Integração com dados de trajectória do Projeto 1 não implementada._\n\n"
            "Se implementada, esta secção correlacionaria os issues detetados com padrões "
            "de afluência (dwell time, número de visitantes) nas zonas correspondentes, "
            "permitindo distinguir entre falha de reposição e esvaziamento por procura elevada."
        )

    return "\n".join(linhas)


def gerar_relatorio(
    inspecoes: list[dict],
    titulo: str = "Relatório de Inspeção",
    guardar: bool = True,
) -> str:
    if not inspecoes:
        return "# Relatório de Inspeção\n\n_Nenhuma inspeção para reportar._"

    log.info(f"A gerar relatório para {len(inspecoes)} inspeções...")

    stats = calcular_estatisticas(inspecoes)
    por_zona = agrupar_por_zona(inspecoes)
    notificacoes = executing_notifs = executar_regras_sessao(inspecoes)
    contexto_geral = obter_contexto_historico_geral(stats)

    log.info("A gerar sumário executivo...")
    sumario = gerar_sumario_executivo(stats)

    log.info("A gerar recomendações...")
    recomendacoes = gerar_recomendacoes(stats, contexto_geral)

    agora = datetime.now().strftime("%Y-%m-%d %H:%M")
    zonas_str = ", ".join(sorted(stats.get("zonas", [])))

    secoes = [
        f"# {titulo}",
        f"**Gerado em:** {agora} | **Zonas:** {zonas_str} | "
        f"**Total inspeções:** {stats['total_inspecoes']}",
        "",
        "---",
        "",

        "## 1. Sumário Executivo\n",
        sumario,
        "",
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| Zonas inspecionadas | {len(stats['zonas'])} |",
        f"| Issues críticos | {stats['criticos']} |",
        f"| Issues warning | {stats['warnings']} |",
        f"| Inspeções OK | {stats['oks']} |",
        f"| Fill rate médio | {stats['fill_rate_medio']:.0%} |",
        "",

        construir_secao_problemas_por_zona(por_zona),
        construir_secao_regras(notificacoes),
        construir_secao_historico(contexto_geral, stats),

        "## 5. Recomendações\n",
        recomendacoes,
        "",

        construir_secao_trajetoria(),
        "",
        "---",
        f"_Relatório gerado automaticamente pelo Retail Vision Intelligence System — TP2 LIACD_",
    ]

    conteudo = "\n".join(secoes)

    if guardar:
        nome_ficheiro = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        caminho = REPORTS_DIR / nome_ficheiro
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
        log.info(f"✓ Relatório guardado em {caminho}")

    return conteudo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Report Generator — Relatórios de inspeção")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", metavar="PASTA", help="Gera relatório de todas as inspeções da pasta (últimas 24h)")
    group.add_argument("--inspection", metavar="FICHEIRO", help="Gera relatório de uma inspeção específica")
    group.add_argument("--zone", metavar="ZONA_ID", help="Gera relatório de uma zona nos últimos N dias")

    parser.add_argument("--period", type=int, default=14, help="Período em dias para --zone (padrão: 14)")
    parser.add_argument("--hours", type=int, default=24, help="Período em horas para --session (padrão: 24)")
    parser.add_argument("--no-save", action="store_true", help="Não guarda o relatório em disco")

    args = parser.parse_args()

    if args.session:
        inspecoes = carregar_inspecoes_sessao(args.session, horas=args.hours)
        if not inspecoes:
            print(f"Nenhuma inspeção encontrada nas últimas {args.hours}h em {args.session}")
        else:
            relatorio = gerar_relatorio(
                inspecoes,
                titulo=f"Relatório de Sessão — {datetime.now().strftime('%Y-%m-%d')}",
                guardar=not args.no_save,
            )
            print(relatorio)

    elif args.inspection:
        if not Path(args.inspection).exists():
            print(f"Erro: O ficheiro de inspeção '{args.inspection}' não existe.")
        else:
            with open(args.inspection, encoding="utf-8") as f:
                inspecao = json.load(f)
            relatorio = gerar_relatorio(
                [inspecao],
                titulo=f"Relatório — {inspecao.get('inspection_id')}",
                guardar=not args.no_save,
            )
            print(relatorio)

    elif args.zone:
        inspecoes = carregar_inspecoes_zona(args.zone, dias=args.period)
        if not inspecoes:
            print(f"Nenhuma inspeção encontrada para zona {args.zone} nos últimos {args.period} dias")
        else:
            relatorio = gerar_relatorio(
                inspecoes,
                titulo=f"Relatório Zona {args.zone} — Últimos {args.period} dias",
                guardar=not args.no_save,
            )
            print(relatorio)