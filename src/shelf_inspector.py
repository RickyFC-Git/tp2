import os
import json
import time
import hashlib
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("A variável de ambiente OPENROUTER_API_KEY não foi encontrada.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/teu-utilizador/tp2-liacd",
        "X-Title": "Retail Vision Intelligence System",
    }
)

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class RateLimiter:
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
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return None


def save_to_cache(cache_key: str, result: dict):
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

PROMPT_A_ZERO_SHOT = """Analisa esta imagem de uma prateleira de supermercado e devolve um JSON válido com o seguinte schema obrigatório:

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
      "location": "TEXTO EM PORTUGUÊS (ex: 'prateleira do meio, lado esquerdo')",
      "severity": "low|medium|high",
      "description": "TEXTO EM PORTUGUÊS (ex: 'várias falhas de stock visíveis')",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": ["TEXTO EM PORTUGUÊS (ex: 'medicamentos', 'higiene')"],
  "model_reasoning": "TEXTO EM PORTUGUÊS (raciocínio detalhado antes da classificação)"
}

REGRAS DE IDIOMA CRÍTICAS:
1. As chaves do JSON e os valores de 'overall_status', 'type' e 'severity' TÊM de manter o formato em inglês do schema acima.
2. Os campos de texto livre ('location', 'description', 'products_detected' e 'model_reasoning') têm de ser OBRIGATORIAMENTE preenchidos em Português de Portugal. Não uses nenhuma palavra em inglês nestes campos."""


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
Devolve o JSON estruturado abaixo. Nota que o campo 'model_reasoning' deve conter o resumo em português do teu raciocínio dos passos anteriores:

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
      "location": "TEXTO EM PORTUGUÊS (ex: 'secção superior, lado direito')",
      "severity": "low|medium|high",
      "description": "TEXTO EM PORTUGUÊS (ex: 'produto desalinhado e a tapar o preço')",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": ["TEXTO EM PORTUGUÊS"],
  "model_reasoning": "TEXTO EM PORTUGUÊS (resumo dos passos 1 a 4)"
}

REGRAS DE IDIOMA CRÍTICAS:
1. Mantém as chaves e os enums ('overall_status', 'type', 'severity') em inglês exatamente como solicitado no schema.
2. Escreve obrigatoriamente todo o conteúdo dos campos 'location', 'description', 'products_detected' e 'model_reasoning' em Português de Portugal."""


PROMPT_C_FEW_SHOT = """Vais analisar uma imagem de prateleira de supermercado. Aqui estão dois exemplos de análises corretas anteriores para guiar o teu raciocínio:

EXEMPLO 1:
Imagem: prateleira de bebidas com garrafas alinhadas, todas as posições preenchidas.
JSON resultado: {{"overall_status": "ok", "issues": [], "shelf_fill_rate": 95.0, "products_detected": ["bebidas", "sumos"], "model_reasoning": "Prateleira com bebidas em bom estado. Todas as posições preenchidas. Produtos alinhados e com etiquetas visíveis."}}

EXEMPLO 2:
Imagem: prateleira de snacks com 3 posições vazias no lado esquerdo e um produto tombado ao centro.
JSON resultado: {{"overall_status": "warning", "issues": [{{"type": "empty_shelf", "location": "prateleira central, lado esquerdo", "severity": "high", "description": "3 posições consecutivas sem produto", "confidence": 0.92, "affected_area_pct": 25.0}}, {{"type": "misaligned", "location": "prateleira central, centro", "severity": "medium", "description": "produto tombado bloqueando etiqueta", "confidence": 0.85, "affected_area_pct": 5.0}}], "shelf_fill_rate": 65.0, "products_detected": ["snacks", "batatas fritas"], "model_reasoning": "Lado esquerdo com lacuna significativa. Produto tombado ao centro. Estado geral warning."}}

Agora analisa a imagem fornecida e devolve o JSON completo com o seguinte formato:

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

REGRAS DE IDIOMA CRÍTICAS:
1. Mantém as chaves e os enums ('overall_status', 'type', 'severity') em inglês exatamente como solicitado no schema.
2. Escreve obrigatoriamente todo o conteúdo dos campos 'location', 'description', 'products_detected' e 'model_reasoning' em Português de Portugal."""

PROMPTS = {
    "A": PROMPT_A_ZERO_SHOT,
    "B": PROMPT_B_COT,
    "C": PROMPT_C_FEW_SHOT,
}

def load_image_base64(image_path: str, max_size: int = 1024) -> tuple[str, str]:
    img = Image.open(image_path)
    
    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((max_size, max_size))

    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64, "image/jpeg"

    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64, "image/jpeg"

def parse_json_response(raw: str) -> dict:
    """Limpa blocos de código Markdown de forma segura se o modelo os incluir."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return json.loads(raw)

def generate_inspection_id(index: int = 0) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"INS_{ts}_{index:03d}"

def fill_metadata(result: dict, image_path: str, zone_id: str, index: int) -> dict:
    result["inspection_id"] = generate_inspection_id(index)
    
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
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

    if strategy not in PROMPTS:
        raise ValueError("Estratégia inválida. Usa A, B ou C.")

    if not Path(image_path).exists():
        raise FileNotFoundError(image_path)

    cache_key = get_cache_key(image_path, strategy)
    cached = load_from_cache(cache_key)
    if cached:
        return cached

    prompt = PROMPTS[strategy]
    
    model = "openai/gpt-4o-mini"

    img_b64, mime_type = load_image_base64(image_path)

    for attempt in range(5):
        try:
            rate_limiter.wait_if_needed()

            log.info(f"[{strategy}] {Path(image_path).name} tentativa {attempt+1}")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=2000,
                timeout=60.0
            )

            raw_text = response.choices[0].message.content
            break

        except Exception as e:
            log.error(f"Erro API OpenRouter: {e}")
            time.sleep(2 ** attempt)

    else:
        raise RuntimeError("Falha após várias tentativas no OpenRouter")

    try:
        result = parse_json_response(raw_text)
    except Exception as e:
        result = {
            "inspection_id": generate_inspection_id(index),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_path": image_path,
            "zone_id": zone_id,
            "overall_status": "critical",
            "issues": [],
            "shelf_fill_rate": 0.0,
            "products_detected": [],
            "model_reasoning": f"PARSE_ERROR: {str(e)}. Raw text: {raw_text[:200]}",
        }

    result = fill_metadata(result, image_path, zone_id, index)

    save_to_cache(cache_key, result)

    if save_result:
        out_dir = Path("./data/inspections")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"{result['inspection_id']}_strategy{strategy}.json"
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        log.info(f"Guardado: {out_file}")

    return result


def inspect_batch(
    images_dir: str,
    strategy: str = "B",
    zone_id: str = "Z_UNKNOWN",
    extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> list[dict]:
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