import os
import json
import time
import hashlib
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

class RateLimiter:
    """Garante máximo de 15 req/min com backoff exponencial em erro 429."""

    def __init__(self, max_per_minute: int = 14):
        self.max_per_minute = max_per_minute
        self.timestamps: list[float] = []

    def wait_if_needed(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < 60]
        if len(self.timestamps) >= self.max_per_minute:
            wait = 60 - (now - self.timestamps[0]) + 1
            log.info(f"Rate limit: aguardando {wait:.1f}s...")
            time.sleep(wait)
        self.timestamps.append(time.time())

rate_limiter = RateLimiter()

def get_cache_key(image_path: str, strategy: str) -> str:
    with open(image_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return f"{md5}_{strategy}"

def load_from_cache(cache_key: str) -> Optional[dict]:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        log.info(f"Cache hit: {cache_key}")
        with open(cache_file) as f:
            return json.load(f)
    return None

def save_to_cache(cache_key: str, result: dict):
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

PROMPT_A_ZERO_SHOT = """Analisa esta imagem de uma prateleira de supermercado e devolve APENAS um JSON válido com o seguinte schema, sem texto adicional antes ou depois:

{
  "inspection_id": "INS_<TIMESTAMP>_<ID>",
  "timestamp": "<ISO8601>",
  "image_path": "<PATH>",
  "zone_id": "<ZONE>",
  "overall_status": "ok|warning|critical",
  "issues": [
    {
      "issue_id": "ISS_001",
      "type": "empty_shelf|wrong_product|damaged|misaligned|label_missing|other",
      "location": "descrição da localização na prateleira",
      "severity": "low|medium|high",
      "description": "descrição do problema",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": ["categorias de produto visíveis"],
  "model_reasoning": "raciocínio explícito antes da classificação"
}

Preenche todos os campos. Se não houver problemas, "issues" deve ser lista vazia e "overall_status" deve ser "ok"."""


PROMPT_B_COT = """Vais analisar uma imagem de prateleira de supermercado passo a passo.

PASSO 1 — DESCRIÇÃO GERAL:
Descreve o que vês na imagem: tipo de produtos, número de prateleiras visíveis, iluminação, ângulo da câmara.

PASSO 2 — ANÁLISE ZONA A ZONA:
Divide mentalmente a prateleira em secções (esquerda/centro/direita, superior/médio/inferior) e descreve o estado de cada uma.

PASSO 3 — IDENTIFICAÇÃO DE ANOMALIAS:
Para cada anomalia encontrada, indica: localização exata, tipo de problema, gravidade aparente.

PASSO 4 — CLASSIFICAÇÃO:
Com base nos passos anteriores, classifica o estado geral: ok / warning / critical.

PASSO 5 — OUTPUT JSON:
Devolve APENAS o JSON abaixo, sem texto adicional, com o teu raciocínio dos passos anteriores no campo "model_reasoning":

{
  "inspection_id": "INS_<TIMESTAMP>_<ID>",
  "timestamp": "<ISO8601>",
  "image_path": "<PATH>",
  "zone_id": "<ZONE>",
  "overall_status": "ok|warning|critical",
  "issues": [
    {
      "issue_id": "ISS_001",
      "type": "empty_shelf|wrong_product|damaged|misaligned|label_missing|other",
      "location": "descrição da localização na prateleira",
      "severity": "low|medium|high",
      "description": "descrição do problema",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": ["categorias de produto visíveis"],
  "model_reasoning": "resumo do raciocínio dos 4 passos acima"
}"""


PROMPT_C_FEW_SHOT = """Vais analisar uma imagem de prateleira de supermercado. Aqui estão dois exemplos de análises corretas anteriores para guiar o teu raciocínio:

EXEMPLO 1 (prateleira normal):
Imagem: prateleira de bebidas com garrafas alinhadas, todas as posições preenchidas.
Análise: Fill rate ~95%. Produtos bem posicionados. Sem anomalias visíveis.
JSON resultado: {{"overall_status": "ok", "issues": [], "shelf_fill_rate": 0.95, "model_reasoning": "Prateleira com bebidas em bom estado. Todas as posições preenchidas. Produtos alinhados e com etiquetas visíveis."}}

EXEMPLO 2 (prateleira com problemas):
Imagem: prateleira de snacks com 3 posições vazias no lado esquerdo e um produto tombado ao centro.
Análise: Fill rate ~65%. Dois problemas identificados: lacuna no lado esquerdo (empty_shelf, severity high) e produto tombado ao centro (misaligned, severity medium).
JSON resultado: {{"overall_status": "warning", "issues": [{{"type": "empty_shelf", "location": "prateleira central, lado esquerdo", "severity": "high", "description": "3 posições consecutivas sem produto", "confidence": 0.92, "affected_area_pct": 0.25}}, {{"type": "misaligned", "location": "prateleira central, centro", "severity": "medium", "description": "produto tombado bloqueando etiqueta", "confidence": 0.85, "affected_area_pct": 0.05}}], "shelf_fill_rate": 0.65, "model_reasoning": "Lado esquerdo com lacuna significativa. Produto tombado ao centro. Estado geral warning."}}

Agora analisa a imagem fornecida e devolve APENAS o JSON completo, sem texto adicional:

{
  "inspection_id": "INS_<TIMESTAMP>_<ID>",
  "timestamp": "<ISO8601>",
  "image_path": "<PATH>",
  "zone_id": "<ZONE>",
  "overall_status": "ok|warning|critical",
  "issues": [
    {
      "issue_id": "ISS_001",
      "type": "empty_shelf|wrong_product|damaged|misaligned|label_missing|other",
      "location": "descrição da localização na prateleira",
      "severity": "low|medium|high",
      "description": "descrição do problema",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": ["categorias de produto visíveis"],
  "model_reasoning": "raciocínio explícito antes da classificação"
}"""

PROMPTS = {
    "A": PROMPT_A_ZERO_SHOT,
    "B": PROMPT_B_COT,
    "C": PROMPT_C_FEW_SHOT,
}

def load_image_base64(image_path: str) -> tuple[str, str]:
    """Carrega imagem e devolve (base64_data, mime_type)."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64, "image/jpeg"

def generate_inspection_id(index: int = 0) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"INS_{ts}_{index:03d}"

def parse_json_response(raw: str) -> dict:
    """Extrai JSON da resposta do modelo, removendo markdown se necessário."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(raw)

def fill_metadata(result: dict, image_path: str, zone_id: str, index: int) -> dict:
    """Preenche campos de metadados que o modelo pode ter deixado como placeholder."""
    now = datetime.now(timezone.utc).isoformat()
    result.setdefault("inspection_id", generate_inspection_id(index))
    result["timestamp"] = now
    result["image_path"] = str(image_path)
    result["zone_id"] = zone_id
    for i, issue in enumerate(result.get("issues", []), start=1):
        issue.setdefault("issue_id", f"ISS_{i:03d}")
    return result

def inspect_shelf(
    image_path: str,
    strategy: str = "B",
    zone_id: str = "Z_UNKNOWN",
    index: int = 0,
    save_result: bool = True,
) -> dict:
    """
    Analisa uma imagem de prateleira com o Gemini Flash.

    Args:
        image_path: Caminho para a imagem.
        strategy: "A" (zero-shot), "B" (chain-of-thought), "C" (few-shot).
        zone_id: Identificador da zona da loja (e.g. "Z_S3").
        index: Índice para o inspection_id.
        save_result: Se True, guarda o resultado em data/inspections/.

    Returns:
        Dict com o resultado da inspeção no schema definido.
    """
    if strategy not in PROMPTS:
        raise ValueError(f"Estratégia inválida: {strategy}. Usa 'A', 'B' ou 'C'.")

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    cache_key = get_cache_key(image_path, strategy)
    cached = load_from_cache(cache_key)
    if cached:
        return cached

    prompt = PROMPTS[strategy]
    model = genai.GenerativeModel("gemini-1.5-flash")
    img_b64, mime_type = load_image_base64(image_path)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            log.info(f"[{strategy}] Inspecionando {Path(image_path).name} (tentativa {attempt+1})")

            response = model.generate_content(
                [
                    {"mime_type": mime_type, "data": img_b64},
                    prompt,
                ],
                generation_config={"temperature": 0},
            )
            raw_text = response.text
            break

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) * 10  
                    log.warning(f"Quota esgotada (429). Aguardando {wait}s...")
                    time.sleep(wait)
                else:
                    log.error("Quota diária esgotada. Sistema em modo cache-only.")
                    raise RuntimeError(
                        "QUOTA_EXCEEDED: Quota diária da API esgotada. "
                        "O sistema continua a funcionar para imagens em cache. "
                        "Tenta novamente amanhã."
                    )
            else:
                log.error(f"Erro na API: {e}")
                raise

    try:
        result = parse_json_response(raw_text)
    except json.JSONDecodeError as e:
        log.error(f"Resposta não é JSON válido: {raw_text[:200]}...")
        result = {
            "inspection_id": generate_inspection_id(index),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_path": str(image_path),
            "zone_id": zone_id,
            "overall_status": "critical",
            "issues": [],
            "shelf_fill_rate": 0.0,
            "products_detected": [],
            "model_reasoning": f"PARSE_ERROR: {str(e)} | Raw: {raw_text[:300]}",
            "_parse_error": True,
        }

    result = fill_metadata(result, image_path, zone_id, index)

    save_to_cache(cache_key, result)

    if save_result:
        inspections_dir = Path("./data/inspections")
        inspections_dir.mkdir(parents=True, exist_ok=True)
        out_file = inspections_dir / f"{result['inspection_id']}_strategy{strategy}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log.info(f"Resultado guardado em {out_file}")

    return result


def inspect_batch(
    images_dir: str,
    strategy: str = "B",
    zone_id: str = "Z_UNKNOWN",
    extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> list[dict]:
    """
    Inspecciona todas as imagens de uma pasta.

    Returns:
        Lista de resultados de inspeção.
    """
    images_dir = Path(images_dir)
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in extensions
    ]
    image_files.sort()

    if not image_files:
        log.warning(f"Nenhuma imagem encontrada em {images_dir}")
        return []

    log.info(f"Inspecionando {len(image_files)} imagens com estratégia {strategy}...")
    results = []

    for i, img_path in enumerate(image_files):
        try:
            result = inspect_shelf(
                str(img_path),
                strategy=strategy,
                zone_id=zone_id,
                index=i,
            )
            results.append(result)
        except RuntimeError as e:
            if "QUOTA_EXCEEDED" in str(e):
                log.error(str(e))
                log.info(f"Parando após {i} imagens. Processadas em cache continuam disponíveis.")
                break
            else:
                log.error(f"Erro ao processar {img_path}: {e}")
                continue
        except Exception as e:
            log.error(f"Erro inesperado em {img_path}: {e}")
            continue

    log.info(f"Batch concluído: {len(results)}/{len(image_files)} imagens processadas.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shelf Inspector — análise visual de prateleiras")
    parser.add_argument("image", help="Caminho para a imagem ou pasta de imagens")
    parser.add_argument("--strategy", choices=["A", "B", "C"], default="B",
                        help="Estratégia de prompting: A=zero-shot, B=CoT, C=few-shot")
    parser.add_argument("--zone", default="Z_UNKNOWN", help="ID da zona (ex: Z_S3)")
    parser.add_argument("--all-strategies", action="store_true",
                        help="Corre as 3 estratégias e compara resultados")
    args = parser.parse_args()

    target = Path(args.image)

    if args.all_strategies:
        print(f"\n{'='*60}")
        print(f"Comparação das 3 estratégias para: {target.name}")
        print(f"{'='*60}")
        for s in ["A", "B", "C"]:
            print(f"\n--- Estratégia {s} ---")
            try:
                res = inspect_shelf(str(target), strategy=s, zone_id=args.zone)
                print(f"Status: {res['overall_status']}")
                print(f"Fill rate: {res.get('shelf_fill_rate', 'N/A')}")
                print(f"Issues: {len(res.get('issues', []))}")
                print(f"Raciocínio: {res.get('model_reasoning', '')[:150]}...")
            except Exception as e:
                print(f"Erro: {e}")

    elif target.is_dir():
        results = inspect_batch(str(target), strategy=args.strategy, zone_id=args.zone)
        print(f"\nResultados: {len(results)} inspeções")
        status_counts = {}
        for r in results:
            s = r.get("overall_status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
    else:
        result = inspect_shelf(str(target), strategy=args.strategy, zone_id=args.zone)
        print(json.dumps(result, ensure_ascii=False, indent=2))