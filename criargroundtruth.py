"""
criar_ground_truth.py
=====================
Este script ajuda-te a criar o ground_truth.json com 15 imagens anotadas.

O que faz:
  1. Seleciona 3 imagens de cada categoria (normal, vazia, planograma, desordenada, ambigua)
  2. Mostra o nome de cada imagem e pede-te para confirmar/ajustar a anotacao
  3. Gera o ficheiro ground_truth.json pronto a usar no evaluate.py

Uso:
  python criar_ground_truth.py
  python criar_ground_truth.py --auto   (gera com valores padrao sem perguntar)
"""

import json
import argparse
from pathlib import Path


DEFAULTS = {
    "normal": {
        "overall_status": "ok",
        "issues": [],
        "shelf_fill_rate_min": 0.85,
        "shelf_fill_rate_max": 1.0,
        "zone_id": "Z_S1",
    },
    "vazia": {
        "overall_status": "critical",
        "issues": [
            {"type": "empty_shelf", "location": "prateleira inferior", "severity": "high"}
        ],
        "shelf_fill_rate_min": 0.0,
        "shelf_fill_rate_max": 0.35,
        "zone_id": "Z_S2",
    },
    "planograma": {
        "overall_status": "warning",
        "issues": [
            {"type": "wrong_product", "location": "prateleira central", "severity": "medium"}
        ],
        "shelf_fill_rate_min": 0.5,
        "shelf_fill_rate_max": 0.85,
        "zone_id": "Z_S3",
    },
    "desordenada": {
        "overall_status": "warning",
        "issues": [
            {"type": "damaged", "location": "prateleira superior", "severity": "medium"},
            {"type": "misaligned", "location": "prateleira inferior", "severity": "low"},
        ],
        "shelf_fill_rate_min": 0.55,
        "shelf_fill_rate_max": 0.85,
        "zone_id": "Z_S4",
    },
    "ambigua": {
        "overall_status": "warning",
        "issues": [
            {"type": "other", "location": "prateleira central", "severity": "low"}
        ],
        "shelf_fill_rate_min": 0.4,
        "shelf_fill_rate_max": 0.8,
        "zone_id": "Z_S1",
    },
}

TIPOS_VALIDOS = ["empty_shelf", "wrong_product", "damaged", "misaligned", "label_missing", "other"]
SEVERIDADES_VALIDAS = ["low", "medium", "high"]
STATUSES_VALIDOS = ["ok", "warning", "critical"]


def selecionar_imagens(n_por_categoria: int = 3) -> list[dict]:
    """Seleciona N imagens de cada categoria e devolve lista com caminho e categoria."""
    selecionadas = []
    base = Path("data/images")

    for categoria in ["normal", "vazia", "planograma", "desordenada", "ambigua"]:
        pasta = base / categoria
        if not pasta.exists():
            print(f"[AVISO] Pasta nao encontrada: {pasta}")
            continue

        imagens = sorted(pasta.glob("*.jpg"))[:n_por_categoria]
        if len(imagens) < n_por_categoria:
            print(f"[AVISO] {categoria}: so tem {len(imagens)} imagens (precisas de {n_por_categoria})")

        for img in imagens:
            selecionadas.append({"path": str(img), "categoria": categoria})

    return selecionadas


def input_com_default(mensagem: str, default: str) -> str:
    """Pede input ao utilizador. Se Enter, usa o valor default."""
    resposta = input(f"{mensagem} [{default}]: ").strip()
    return resposta if resposta else default


def anotar_imagem_interativo(img_path: str, categoria: str, indice: int, total: int) -> dict:
    """Modo interativo: mostra a imagem e pede confirmacao das anotacoes."""
    defaults = DEFAULTS[categoria]
    print(f"\n{'='*60}")
    print(f"  Imagem {indice}/{total}: {img_path}")
    print(f"  Categoria: {categoria}")
    print(f"{'='*60}")
    print(f"  Abre esta imagem no teu computador e observa-a.")
    print(f"  Depois confirma ou altera os valores abaixo.")
    print(f"  (Carrega Enter para aceitar o valor entre [ ])\n")

    # Overall status
    while True:
        status = input_com_default(
            f"  overall_status (ok/warning/critical)",
            defaults["overall_status"]
        )
        if status in STATUSES_VALIDOS:
            break
        print(f"  [ERRO] Valor invalido. Usa: {', '.join(STATUSES_VALIDOS)}")

    # Fill rate
    while True:
        try:
            fill_min_str = input_com_default(
                f"  shelf_fill_rate minimo (0.0 a 1.0)",
                str(defaults["shelf_fill_rate_min"])
            )
            fill_max_str = input_com_default(
                f"  shelf_fill_rate maximo (0.0 a 1.0)",
                str(defaults["shelf_fill_rate_max"])
            )
            fill_min = float(fill_min_str)
            fill_max = float(fill_max_str)
            if 0.0 <= fill_min <= fill_max <= 1.0:
                break
            print("  [ERRO] fill_min deve ser <= fill_max e ambos entre 0.0 e 1.0")
        except ValueError:
            print("  [ERRO] Introduz um numero decimal (ex: 0.75)")

    # Issues
    print(f"\n  Issues predefinidos para '{categoria}':")
    for i, issue in enumerate(defaults["issues"]):
        print(f"    {i+1}. type={issue['type']}, location={issue['location']}, severity={issue['severity']}")

    usar_defaults = input_com_default(
        "  Usar estes issues? (s/n)",
        "s"
    ).lower()

    if usar_defaults == "s":
        issues = defaults["issues"]
    else:
        issues = []
        print("  Tipos validos:", ", ".join(TIPOS_VALIDOS))
        print("  Severidades validas:", ", ".join(SEVERIDADES_VALIDAS))
        print("  (Deixa em branco para terminar)\n")

        while True:
            tipo = input("  Tipo do issue (ou Enter para terminar): ").strip()
            if not tipo:
                break
            if tipo not in TIPOS_VALIDOS:
                print(f"  [ERRO] Tipo invalido. Usa: {', '.join(TIPOS_VALIDOS)}")
                continue

            location = input("  Localizacao (ex: prateleira inferior): ").strip()
            if not location:
                location = "prateleira central"

            while True:
                severity = input("  Severidade (low/medium/high): ").strip()
                if severity in SEVERIDADES_VALIDAS:
                    break
                print(f"  [ERRO] Usa: {', '.join(SEVERIDADES_VALIDAS)}")

            issues.append({"type": tipo, "location": location, "severity": severity})

    return {
        "image_path": img_path,
        "zone_id": defaults["zone_id"],
        "overall_status": status,
        "issues": issues,
        "shelf_fill_rate_min": fill_min,
        "shelf_fill_rate_max": fill_max,
    }


def anotar_imagem_auto(img_path: str, categoria: str) -> dict:
    """Modo automatico: usa os valores padrao sem perguntar."""
    defaults = DEFAULTS[categoria]
    return {
        "image_path": img_path,
        "zone_id": defaults["zone_id"],
        "overall_status": defaults["overall_status"],
        "issues": defaults["issues"],
        "shelf_fill_rate_min": defaults["shelf_fill_rate_min"],
        "shelf_fill_rate_max": defaults["shelf_fill_rate_max"],
    }


def main():
    parser = argparse.ArgumentParser(description="Cria o ground_truth.json para avaliacao")
    parser.add_argument("--auto", action="store_true",
                        help="Gera com valores padrao sem modo interativo")
    parser.add_argument("--n", type=int, default=3,
                        help="Numero de imagens por categoria (default: 3)")
    parser.add_argument("--output", default="ground_truth.json",
                        help="Ficheiro de saida (default: ground_truth.json)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Criacao do Ground Truth para Avaliacao")
    print(f"  {args.n} imagens por categoria = {args.n * 5} imagens total")
    print("=" * 60)

    imagens = selecionar_imagens(n_por_categoria=args.n)

    if not imagens:
        print("[ERRO] Nenhuma imagem encontrada em data/images/")
        print("       Verifica se o download_dataset.py correu com sucesso.")
        return

    print(f"\n[OK] {len(imagens)} imagens selecionadas\n")

    anotacoes = []

    for i, item in enumerate(imagens, start=1):
        if args.auto:
            anotacao = anotar_imagem_auto(item["path"], item["categoria"])
        else:
            anotacao = anotar_imagem_interativo(
                item["path"], item["categoria"], i, len(imagens)
            )
        anotacoes.append(anotacao)

    # Guarda o ficheiro
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  [OK] {args.output} criado com {len(anotacoes)} anotacoes")
    print(f"{'='*60}")
    print(f"\nDistribuicao:")
    contagem = {}
    for a in anotacoes:
        s = a["overall_status"]
        contagem[s] = contagem.get(s, 0) + 1
    for status, count in sorted(contagem.items()):
        print(f"  {status:<10}: {count}")

    print(f"\nProximo passo:")
    print(f"  python evaluate.py --images-dir data/images/ --ground-truth {args.output} --strategy all --output evaluation_report.json")


if __name__ == "__main__":
    main()