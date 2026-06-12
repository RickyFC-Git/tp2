import os
import json
import sys
import shlex
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def importar_componentes():
    componentes = {}

    try:
        import shelf_inspector as si
        componentes["inspector"] = si
    except ImportError as e:
        print(f"[AVISO] shelf_inspector nao disponivel: {e}")

    try:
        import rule_engine as re_mod
        componentes["rules"] = re_mod
    except ImportError as e:
        print(f"[AVISO] rule_engine nao disponivel: {e}")

    try:
        import rag_memory as rag
        componentes["rag"] = rag
    except ImportError as e:
        print(f"[AVISO] rag_memory nao disponivel: {e}")

    try:
        import report_generator as rg
        componentes["report"] = rg
    except ImportError as e:
        print(f"[AVISO] report_generator nao disponivel: {e}")

    return componentes


class Sessao:
    def __init__(self):
        self.inspecoes_sessao: list[dict] = []
        self.notificacoes_sessao: list[dict] = []
        self.inicio = datetime.now(timezone.utc)
        self.ultima_inspecao: str = ""

    def adicionar_inspecao(self, inspecao: dict):
        self.inspecoes_sessao.append(inspecao)
        self.ultima_inspecao = inspecao.get("inspection_id", "")

    def resumo(self) -> str:
        total = len(self.inspecoes_sessao)
        if total == 0:
            return "Nenhuma inspecao realizada nesta sessao."
        criticos = sum(1 for i in self.inspecoes_sessao if i.get("overall_status") == "critical")
        warnings = sum(1 for i in self.inspecoes_sessao if i.get("overall_status") == "warning")
        return (
            f"{total} inspecao(oes) | "
            f"{criticos} critico(s) | "
            f"{warnings} warning(s) | "
            f"Inicio: {self.inicio.strftime('%H:%M')}"
        )


SEPARADOR = "-" * 60

def imprimir_titulo(texto: str):
    print(f"\n{SEPARADOR}")
    print(f"  {texto}")
    print(SEPARADOR)

def imprimir_sucesso(texto: str):
    print(f"[OK] {texto}")

def imprimir_erro(texto: str):
    print(f"[ERRO] {texto}")

def imprimir_aviso(texto: str):
    print(f"[AVISO] {texto}")

def imprimir_info(texto: str):
    print(f"  {texto}")

def imprimir_resultado_inspecao(resultado: dict):
    status = resultado.get("overall_status", "?")
    status_label = {"ok": "[OK]", "warning": "[WARNING]", "critical": "[CRITICAL]"}.get(status, "[?]")

    print(f"\n{status_label} {resultado.get('inspection_id')}")
    print(f"   Zona: {resultado.get('zone_id')} | "
          f"Fill rate: {resultado.get('shelf_fill_rate', 0):.0%} | "
          f"Timestamp: {resultado.get('timestamp', '')[:16]}")

    issues = resultado.get("issues", [])
    if issues:
        print(f"   Issues ({len(issues)}):")
        for issue in issues:
            sev = issue.get("severity", "?")
            print(f"     [{sev.upper()}] [{issue.get('type')}] "
                  f"{issue.get('location', '?')} -- {issue.get('description', '')[:60]}")
    else:
        print("   Sem issues detetados.")

    produtos = resultado.get("products_detected", [])
    if produtos:
        print(f"   Produtos: {', '.join(produtos[:5])}")



def handle_inspect(args: list, sessao: Sessao, comp: dict) -> bool:
    if "inspector" not in comp:
        imprimir_erro("shelf_inspector nao esta disponivel.")
        return False

    if not args:
        imprimir_erro("Uso: inspect <zona> --image <ficheiro>")
        imprimir_info("      inspect all --images-dir <pasta>")
        return False

    zona = args[0]
    estrategia = "B"
    image_path = None
    images_dir = None
    i = 1
    while i < len(args):
        if args[i] == "--image" and i + 1 < len(args):
            image_path = args[i + 1]; i += 2
        elif args[i] == "--images-dir" and i + 1 < len(args):
            images_dir = args[i + 1]; i += 2
        elif args[i] == "--strategy" and i + 1 < len(args):
            estrategia = args[i + 1].upper(); i += 2
        else:
            i += 1

    try:
        if image_path:
            if not Path(image_path).exists():
                imprimir_erro(f"Ficheiro nao encontrado: {image_path}")
                return False
            imprimir_info(f"A inspecionar {image_path} (zona={zona}, estrategia={estrategia})...")
            resultado = comp["inspector"].inspect_shelf(image_path, strategy=estrategia, zone_id=zona)
            sessao.adicionar_inspecao(resultado)
            imprimir_resultado_inspecao(resultado)

            if "rules" in comp:
                notifs = comp["rules"].executar_regras(resultado)
                sessao.notificacoes_sessao.extend(notifs)
                if notifs:
                    print(f"\n   {len(notifs)} alerta(s) de regras:")
                    for n in notifs:
                        nivel = n.get("alert_level", "info").upper()
                        print(f"     [{nivel}] [{n.get('rule_id')}] {n.get('mensagem')}")

            if "rag" in comp:
                comp["rag"].indexar_inspecao(resultado)
                imprimir_info("Inspecao indexada na memoria RAG.")

        elif images_dir or zona.lower() == "all":
            pasta = images_dir or "data/images/"
            if not Path(pasta).exists():
                imprimir_erro(f"Pasta nao encontrada: {pasta}")
                return False
            imprimir_info(f"A inspecionar todas as imagens em {pasta}...")
            resultados = comp["inspector"].inspect_batch(
                pasta, strategy=estrategia,
                zone_id="Z_UNKNOWN" if zona.lower() == "all" else zona
            )
            for res in resultados:
                sessao.adicionar_inspecao(res)
                imprimir_resultado_inspecao(res)
                if "rules" in comp:
                    notifs = comp["rules"].executar_regras(res)
                    sessao.notificacoes_sessao.extend(notifs)
                if "rag" in comp:
                    comp["rag"].indexar_inspecao(res)
            imprimir_sucesso(f"{len(resultados)} inspecoes concluidas.")
        else:
            imprimir_erro("Especifica --image <ficheiro> ou --images-dir <pasta>")
            return False

    except RuntimeError as e:
        if "QUOTA_EXCEEDED" in str(e):
            imprimir_aviso("Quota da API esgotada. O sistema continua a funcionar para imagens em cache.")
        else:
            imprimir_erro(f"Erro na inspecao: {e}")
        return False
    except Exception as e:
        imprimir_erro("Erro inesperado na inspecao.")
        log.exception("Detalhe:")
        return False

    return True


def handle_add_rule(args: list, comp: dict) -> bool:
    if "rules" not in comp:
        imprimir_erro("rule_engine nao esta disponivel.")
        return False
    if not args:
        imprimir_erro('Uso: add rule "descricao da regra em portugues"')
        return False
    texto = " ".join(args)
    try:
        regra = comp["rules"].cmd_add(texto)
        return regra.get("validation", {}).get("is_valid", False)
    except Exception as e:
        imprimir_erro("Erro ao adicionar regra.")
        log.exception("Detalhe:")
        return False


def handle_list_rules(comp: dict) -> bool:
    if "rules" not in comp:
        imprimir_erro("rule_engine nao esta disponivel.")
        return False
    try:
        comp["rules"].cmd_list()
        return True
    except Exception as e:
        imprimir_erro("Erro ao listar regras.")
        return False


def handle_delete_rule(args: list, comp: dict) -> bool:
    if "rules" not in comp:
        imprimir_erro("rule_engine nao esta disponivel.")
        return False
    if not args:
        imprimir_erro("Uso: delete rule RULE_001")
        return False
    try:
        comp["rules"].cmd_delete(args[0])
        return True
    except Exception as e:
        imprimir_erro("Erro ao eliminar regra.")
        return False


def handle_test_rule(args: list, comp: dict) -> bool:
    if "rules" not in comp:
        imprimir_erro("rule_engine nao esta disponivel.")
        return False
    if not args:
        imprimir_erro("Uso: test rule RULE_001 --image foto.jpg")
        return False

    rule_id = args[0]
    image_path = None
    inspection_path = None
    i = 1
    while i < len(args):
        if args[i] == "--image" and i + 1 < len(args):
            image_path = args[i + 1]; i += 2
        elif args[i] == "--inspection" and i + 1 < len(args):
            inspection_path = args[i + 1]; i += 2
        else:
            i += 1

    try:
        if image_path and "inspector" in comp:
            imprimir_info("A inspecionar imagem para teste...")
            resultado = comp["inspector"].inspect_shelf(image_path, strategy="B")
            inspection_path = f"data/inspections/{resultado.get('inspection_id')}_strategyB.json"

        if inspection_path:
            comp["rules"].cmd_test(rule_id, inspection_path)
        else:
            imprimir_erro("Especifica --image ou --inspection")
            return False
        return True
    except Exception as e:
        imprimir_erro("Erro ao testar regra.")
        log.exception("Detalhe:")
        return False


def handle_history(args: list, comp: dict) -> bool:
    if "rag" not in comp:
        imprimir_erro("rag_memory nao esta disponivel.")
        return False
    if not args:
        imprimir_erro('Uso: history "query em linguagem natural"')
        return False
    query = " ".join(args)
    try:
        imprimir_info(f"A pesquisar: {query}")
        resultado = comp["rag"].responder_query(query)
        imprimir_titulo("Resposta do Sistema de Memoria")
        print(resultado.get("resposta", ""))
        ids = resultado.get("inspection_ids_referenciados", [])
        if ids:
            print(f"\n  Inspecoes referenciadas: {', '.join(ids)}")
        return True
    except Exception as e:
        imprimir_erro("Erro na consulta historica.")
        log.exception("Detalhe:")
        return False


def handle_compare(args: list, comp: dict) -> bool:
    if "rag" not in comp:
        imprimir_erro("rag_memory nao esta disponivel.")
        return False
    if len(args) < 2:
        imprimir_erro("Uso: compare Z_S1 Z_S3 --period <dias>")
        return False

    zona1, zona2 = args[0], args[1]
    period = 7
    if "--period" in args:
        idx = args.index("--period")
        if idx + 1 < len(args):
            try:
                period = int(args[idx + 1])
            except ValueError:
                pass

    try:
        for zona in [zona1, zona2]:
            query = f"problemas na zona {zona} nos ultimos {period} dias"
            resultado = comp["rag"].responder_query(query)
            imprimir_titulo(f"Zona {zona} -- Ultimos {period} dias")
            print(resultado.get("resposta", ""))
            print()
        return True
    except Exception as e:
        imprimir_erro("Erro na comparacao.")
        return False


def handle_report(args: list, sessao: Sessao, comp: dict) -> bool:
    if "report" not in comp:
        imprimir_erro("report_generator nao esta disponivel.")
        return False

    zone = None
    period = 14
    session_mode = False
    i = 0
    while i < len(args):
        if args[i] == "--session":
            session_mode = True; i += 1
        elif args[i] == "--zone" and i + 1 < len(args):
            zone = args[i + 1]; i += 2
        elif args[i] == "--period" and i + 1 < len(args):
            try:
                period = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    try:
        if session_mode:
            if sessao.inspecoes_sessao:
                imprimir_info(f"A gerar relatorio da sessao ({len(sessao.inspecoes_sessao)} inspecoes)...")
                relatorio = comp["report"].gerar_relatorio(
                    sessao.inspecoes_sessao,
                    titulo=f"Relatorio de Sessao -- {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                )
            else:
                imprimir_info("A carregar inspecoes das ultimas 24h...")
                inspecoes = comp["report"].carregar_inspecoes_sessao("data/inspections/", horas=24)
                if not inspecoes:
                    imprimir_aviso("Nenhuma inspecao nas ultimas 24h.")
                    return False
                relatorio = comp["report"].gerar_relatorio(inspecoes)
            print(relatorio)

        elif zone:
            imprimir_info(f"A gerar relatorio da zona {zone} (ultimos {period} dias)...")
            inspecoes = comp["report"].carregar_inspecoes_zona(zone, dias=period)
            if not inspecoes:
                imprimir_aviso(f"Nenhuma inspecao para zona {zone} nos ultimos {period} dias.")
                return False
            relatorio = comp["report"].gerar_relatorio(
                inspecoes,
                titulo=f"Relatorio Zona {zone} -- Ultimos {period} dias",
            )
            print(relatorio)

        else:
            imprimir_erro("Uso: report --session [today] | report --zone Z_S1 --period 14")
            return False

        return True
    except Exception as e:
        imprimir_erro("Erro ao gerar relatorio.")
        log.exception("Detalhe:")
        return False


def handle_status(sessao: Sessao):
    imprimir_titulo("Estado da Sessao")
    imprimir_info(sessao.resumo())
    rules_dir = Path("data/rules")
    n_regras = len(list(rules_dir.glob("RULE_*.json"))) if rules_dir.exists() else 0
    imprimir_info(f"Regras ativas: {n_regras}")
    insp_dir = Path("data/inspections")
    n_insp = len(list(insp_dir.glob("*.json"))) if insp_dir.exists() else 0
    imprimir_info(f"Total de inspecoes em disco: {n_insp}")


def handle_help():
    print(f"""
{SEPARADOR}
  Retail Vision Intelligence System -- Comandos disponiveis
{SEPARADOR}

  INSPECAO:
    inspect <zona> --image <foto.jpg>
    inspect <zona> --image <foto.jpg> --strategy A|B|C
    inspect all --images-dir <pasta/>

  REGRAS:
    add rule "<regra em portugues>"
    list rules
    delete rule <RULE_ID>
    test rule <RULE_ID> --image <foto.jpg>
    show rule <RULE_ID>

  HISTORICO:
    history "<query>"
    compare <Z_S1> <Z_S2> --period <dias>

  RELATORIOS:
    report --session [today]
    report --zone <Z_S1> --period <dias>

  SISTEMA:
    status
    help
    exit / quit

{SEPARADOR}
  Exemplos:
    inspect Z_S1 --image data/images/normal/normal_0001.jpg
    add rule "Avisa-me quando a prateleira inferior estiver mais de 40% vazia"
    history "quando foi a ultima vez que Z_S1 teve problemas?"
    report --session today
{SEPARADOR}
""")


def processar_comando(linha: str, sessao: Sessao, comp: dict) -> bool:
    linha = linha.strip()
    if not linha:
        return True

    try:
        tokens = shlex.split(linha)
    except ValueError:
        tokens = linha.split()

    if not tokens:
        return True

    cmd = tokens[0].lower()
    args = tokens[1:]

    if cmd in ("exit", "quit", "sair"):
        print("\nAte logo!")
        return False
    elif cmd == "help":
        handle_help()
    elif cmd == "status":
        handle_status(sessao)
    elif cmd == "inspect":
        handle_inspect(args, sessao, comp)
    elif cmd == "add" and args and args[0].lower() == "rule":
        handle_add_rule(args[1:], comp)
    elif cmd == "list" and args and args[0].lower() == "rules":
        handle_list_rules(comp)
    elif cmd == "delete" and args and args[0].lower() == "rule":
        handle_delete_rule(args[1:], comp)
    elif cmd == "test" and args and args[0].lower() == "rule":
        handle_test_rule(args[1:], comp)
    elif cmd == "show" and args and args[0].lower() == "rule":
        if "rules" in comp and len(args) > 1:
            comp["rules"].cmd_show(args[1])
        else:
            imprimir_erro("Uso: show rule RULE_001")
    elif cmd == "history":
        handle_history(args, comp)
    elif cmd == "compare":
        handle_compare(args, comp)
    elif cmd == "report":
        handle_report(args, sessao, comp)
    else:
        imprimir_erro(f"Comando nao reconhecido: '{cmd}'")
        imprimir_info("Escreve 'help' para ver os comandos disponiveis.")

    return True


def main():
    print(""" Retail Vision Intelligence System -- TP2 LIACD
              Escreve 'help' para ver os comandos disponiveis""")

    if not os.getenv("GEMINI_API_KEY"):
        print("[AVISO] GEMINI_API_KEY nao definida no .env")
        print("        Algumas funcionalidades nao estarao disponiveis.\n")

    print("A carregar componentes...")
    comp = importar_componentes()
    print(f"[OK] Componentes carregados: {', '.join(comp.keys()) or 'nenhum'}\n")

    sessao = Sessao()

    while True:
        try:
            linha = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAte logo!")
            break

        continuar = processar_comando(linha, sessao, comp)
        if not continuar:
            break


if __name__ == "__main__":
    main()