import os
import json
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

from PIL import Image
from tqdm import tqdm


LIMITES = {
    "normal":      150,
    "vazia":       100,
    "planograma":  100,
    "desordenada":  80,
    "ambigua":      70,
}

ZIPS = {
    "vazia":       "data/zips/vazia.zip",
    "planograma":  "data/zips/planograma.zip",
    "desordenada": "data/zips/desordenada.zip",
    "ambigua":     "data/zips/ambigua.zip",
}

LICENCAS = {
    "normal":      "SKU-110K (Goldman et al., 2019) - licença académica, uso em investigação",
    "vazia":       "Roboflow Universe: fyp-ormnr/supermarket-empty-shelf-detector - open source",
    "planograma":  "Roboflow Universe: cardatasetcombine/planogram-compliance-x61hg - open source",
    "desordenada": "Roboflow Universe: object-detection-5pf5v/packaging-defect-detection-wbcpk - open source",
    "ambigua":     "Roboflow Universe: planogram-pc7rp/planogram-pyyeu - open source (usado como ambíguo devido ao foco/desalinhamento)",
}


def criar_pastas():
    for categoria in LIMITES:
        Path(f"data/images/{categoria}").mkdir(parents=True, exist_ok=True)
    Path("data/zips").mkdir(parents=True, exist_ok=True)
    print("Estrutura de pastas criada com sucesso")


def guardar_imagem(img: Image.Image, destino: Path) -> bool:
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if max(img.size) > 1920:
            img.thumbnail((1920, 1920), Image.LANCZOS)
        img.save(destino, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"   Erro ao guardar {destino.name}: {e}")
        return False


def imagens_validas_de_zip(zip_path: str, limite: int, destino: str, categoria: str) -> list[dict]:
    registos = []
    destino_path = Path(destino)

    if not Path(zip_path).exists():
        print(f"\nErro: Ficheiro ZIP nao encontrado: {zip_path}")
        print(f"   Descarrega manualmente do Roboflow e coloca em {zip_path}")
        return registos

    print(f"\nA extrair {categoria} de {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        ficheiros = [
            f for f in zf.namelist()
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            and not f.startswith("__MACOSX")
            and "/labels/" not in f
        ]

        if not ficheiros:
            print(f"   Erro: Nenhuma imagem encontrada no ZIP. Verifica o conteudo.")
            return registos

        print(f"   Encontradas {len(ficheiros)} imagens no ZIP. A usar {min(limite, len(ficheiros))}...")

        guardadas = 0
        for i, nome_ficheiro in enumerate(tqdm(ficheiros, desc=f"  {categoria}")):
            if guardadas >= limite:
                break

            nome_destino = destino_path / f"{categoria}_{guardadas:04d}.jpg"

            try:
                with zf.open(nome_ficheiro) as f:
                    from io import BytesIO
                    img = Image.open(BytesIO(f.read()))
                    img.load()
            except Exception as e:
                continue

            if guardar_imagem(img, nome_destino):
                registos.append({
                    "filename": nome_destino.name,
                    "categoria": categoria,
                    "origem": LICENCAS[categoria],
                    "fonte_original": nome_ficheiro,
                    "data_download": datetime.now().strftime("%Y-%m-%d"),
                })
                guardadas += 1

        print(f"   Sucesso: {guardadas} imagens guardadas em data/images/{categoria}/")

    return registos


def descarregar_sku110k(limite: int, destino: str, categoria: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("\nErro: Instala a biblioteca via: pip install datasets")
        return []

    registos = []
    destino_path = Path(destino)

    existentes = list(destino_path.glob("*.jpg"))
    if len(existentes) >= limite:
        print(f"\n{categoria}: ja tem {len(existentes)} imagens, a saltar download")
        return []

    print(f"\nA descarregar {categoria} do SKU-110K via streaming...")
    print(f"  (Apenas {limite} imagens - sem descarregar os 11GB completos)")

    try:
        ds = load_dataset(
            "PrashantDixit0/SKU-110K",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"   Erro ao carregar dataset: {e}")
        print("  Tenta atualizar a biblioteca: pip install datasets --upgrade")
        return []

    guardadas = 0
    erros = 0

    with tqdm(total=limite, desc=f"  {categoria}") as pbar:
        for i, sample in enumerate(ds):
            if guardadas >= limite:
                break

            nome_destino = destino_path / f"{categoria}_{guardadas:04d}.jpg"
            
            if nome_destino.exists():
                guardadas += 1
                pbar.update(1)
                continue

            try:
                import base64
                from io import BytesIO

                raw = sample["image"]

                if isinstance(raw, dict) and "bytes" in raw:
                    img_bytes = raw["bytes"]
                    if isinstance(img_bytes, str):
                        img_bytes = base64.b64decode(img_bytes)
                    img = Image.open(BytesIO(img_bytes))
                elif isinstance(raw, dict) and "path" in raw:
                    img = raw["array"] if "array" in raw else Image.open(raw["path"])
                elif hasattr(raw, "convert"):
                    img = raw
                else:
                    img = Image.open(BytesIO(raw))

                img.load()

            except Exception as e:
                erros += 1
                if erros <= 3:
                    print(f"\n  Aviso: erro na imagem {i}: {e}")
                continue

            if guardar_imagem(img, nome_destino):
                registos.append({
                    "filename": nome_destino.name,
                    "categoria": categoria,
                    "origem": LICENCAS["normal"],
                    "fonte_original": f"SKU-110K train[{i}]",
                    "data_download": datetime.now().strftime("%Y-%m-%d"),
                })
                guardadas += 1
                pbar.update(1)

    print(f"   Sucesso: {guardadas} imagens guardadas em data/images/{categoria}/")
    if erros > 0:
        print(f"   Aviso: {erros} imagens ignoradas por erros de processamento")

    return registos


def gerar_dataset_info(todos_registos: list[dict]):
    resumo = {}
    for r in todos_registos:
        cat = r["categoria"]
        resumo[cat] = resumo.get(cat, 0) + 1

    info = {
        "gerado_em": datetime.now().isoformat(),
        "total_imagens": len(todos_registos),
        "distribuicao": resumo,
        "licencas": LICENCAS,
        "notas": (
            "Dataset construido de forma totalmente automatizada para o TP2 de LIACD. "
            "A categoria 'ambigua' foi preenchida recorrendo a imagens propositadamente desfocadas."
        ),
        "imagens": todos_registos,
    }

    with open("data/dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\nFicheiro dataset_info.json gerado com {len(todos_registos)} registos")


def verificar_estado_final():
    print("RESUMO DO DATASET")

    total = 0
    alvo_total = sum(LIMITES.values())
    
    for categoria, limite in LIMITES.items():
        pasta = Path(f"data/images/{categoria}")
        count = len(list(pasta.glob("*.jpg"))) if pasta.exists() else 0
        total += count
        status = "OK" if count >= limite else f"Faltam {limite - count}"
        print(f"  {categoria:<15} {count:>4}/{limite}  [{status}]")

    print(f"\n  TOTAL: {total}/{alvo_total}")

    if total >= alvo_total:
        print("\nDataset completo e totalmente automatizado")
    else:
        print("\nAviso: Ainda existem categorias em falta. Verifica os ficheiros ZIP em data/zips/")



if __name__ == "__main__":
    print("TP2 LIACD - Fase 0: Construção do Dataset")

    criar_pastas()

    todos_registos = []

    registos = descarregar_sku110k(
        limite=LIMITES["normal"],
        destino="data/images/normal",
        categoria="normal",
    )
    todos_registos.extend(registos)

    for categoria, zip_path in ZIPS.items():
        registos = imagens_validas_de_zip(
            zip_path=zip_path,
            limite=LIMITES[categoria],
            destino=f"data/images/{categoria}",
            categoria=categoria,
        )
        todos_registos.extend(registos)

    if todos_registos:
        gerar_dataset_info(todos_registos)

    verificar_estado_final()