import os
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "openai/gpt-4o-mini"

VECTORSTORE_DIR = "./vectorstore"
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "hybrid")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROMPT_SUMMARY = """Tens os seguintes dados de uma inspeção de prateleira de supermercado.
Gera um summary rico em português, com todos os detalhes relevantes para recuperação futura.

O summary deve mencionar explicitamente:
- A zona inspecionada e a data/hora
- O fill rate exato
- Cada problema encontrado com localização específica e severidade
- Os tipos de produto visíveis
- O estado geral (ok/warning/critical)

DADOS DA INSPEÇÃO:
{dados_inspecao}

EXEMPLO DE BOM SUMMARY:
"Inspeção da zona Z_S3 em terça-feira às 15h. Fill rate de 72%. Produto de limpeza 
(detergente líquido) fora de posição na secção central, severidade média. Embalagem 
danificada detetada no lado direito da prateleira inferior, severidade alta. Produtos 
visíveis: detergentes, amaciadores. Estado geral: warning."

EXEMPLO DE MAU SUMMARY (evitar):
"Prateleira com problemas."

Gera APENAS o summary, sem texto adicional:"""

PROMPT_RESPOSTA_RAG = """És um assistente de gestão de loja de retalho com acesso ao histórico de inspeções.

QUERY DO GESTOR:
"{query}"

INSPEÇÕES RELEVANTES RECUPERADAS (ordenadas por relevância):
{contexto}

Com base nestas inspeções históricas, responde à query do gestor em português.
Obrigatório:
- Referencia explicitamente os inspection_id e datas das inspeções que suportam a resposta
- Sê específico e direto
- Se não houver informação suficiente no histórico, diz-o claramente
- Identifica padrões se existirem (ex: problema recorrente na mesma zona/horário)

Resposta:"""

def chamar_openrouter(prompt: str, temperature: float = 0.2, max_retries: int = 3) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("A variável de ambiente OPENROUTER_API_KEY não está configurada no ficheiro .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }

    url = "https://openrouter.ai/api/v1/chat/completions"

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
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

def inicializar_vectorstore(strategy: str = CHUNK_STRATEGY):
    try:
        import chromadb
    except ImportError:
        raise ImportError("Instala: pip install chromadb")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Instala: pip install sentence-transformers")

    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection_name = f"inspecoes_{strategy}"

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    log.info(f"ChromaDB inicializado: {collection_name} ({collection.count()} documentos)")
    return client, collection, embedding_model

def gerar_summary(inspecao: dict) -> str:
    dados_simplificados = {
        "inspection_id": inspecao.get("inspection_id"),
        "timestamp": inspecao.get("timestamp"),
        "zone_id": inspecao.get("zone_id"),
        "overall_status": inspecao.get("overall_status"),
        "shelf_fill_rate": inspecao.get("shelf_fill_rate"),
        "products_detected": inspecao.get("products_detected", []),
        "issues": inspecao.get("issues", []),
    }

    prompt = PROMPT_SUMMARY.format(
        dados_inspecao=json.dumps(dados_simplificados, ensure_ascii=False, indent=2)
    )

    try:
        return chamar_openrouter(prompt, temperature=0.2)
    except Exception as e:
        log.warning(f"Erro ao gerar summary via OpenRouter: {e}. A usar fallback.")
        return _summary_fallback(inspecao)

def _summary_fallback(inspecao: dict) -> str:
    ts = inspecao.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        data_str = dt.strftime("%A às %Hh")
    except Exception:
        data_str = ts[:16] if ts else "data desconhecida"

    issues = inspecao.get("issues", [])
    issues_str = ""
    if issues:
        descricoes = [f"{i.get('type')} em {i.get('location', '?')} ({i.get('severity')})"
                      for i in issues]
        issues_str = ". Problemas: " + "; ".join(descricoes)

    fill = inspecao.get("shelf_fill_rate", 0)
    if fill <= 1.0:
        fill_str = f"{fill:.0%}"
    else:
        fill_str = f"{fill:.0f}%"

    produtos = ", ".join(inspecao.get("products_detected", []))

    return (
        f"Inspeção {inspecao.get('inspection_id')} da zona {inspecao.get('zone_id')} "
        f"em {data_str}. Fill rate de {fill_str}. "
        f"Estado: {inspecao.get('overall_status')}{issues_str}. "
        f"Produtos: {produtos or 'não identificados'}."
    )

def chunks_hibrido(inspecao: dict, summary: str) -> list[dict]:
    ts = inspecao.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        dia_semana = dt.strftime("%A")
        hora = dt.hour
    except Exception:
        dia_semana = ""
        hora = -1

    return [{
        "id": inspecao.get("inspection_id", f"INS_{hash(summary)}"),
        "texto": summary,
        "metadata": {
            "inspection_id": inspecao.get("inspection_id", ""),
            "zone_id": inspecao.get("zone_id", ""),
            "timestamp": ts,
            "dia_semana": dia_semana,
            "hora": hora,
            "overall_status": inspecao.get("overall_status", ""),
            "shelf_fill_rate": float(inspecao.get("shelf_fill_rate", 0)),
            "issue_count": len(inspecao.get("issues", [])),
            "issue_types": json.dumps([i.get("type") for i in inspecao.get("issues", [])]),
            "strategy": "hybrid",
        }
    }]

def chunks_por_issue(inspecao: dict, summary: str) -> list[dict]:
    inspection_id = inspecao.get("inspection_id", "")
    ts = inspecao.get("timestamp", "")
    zone_id = inspecao.get("zone_id", "")

    try:
        dt = datetime.fromisoformat(ts)
        dia_semana = dt.strftime("%A")
        hora = dt.hour
    except Exception:
        dia_semana = ""
        hora = -1

    metadata_base = {
        "inspection_id": inspection_id,
        "zone_id": zone_id,
        "timestamp": ts,
        "dia_semana": dia_semana,
        "hora": hora,
        "overall_status": inspecao.get("overall_status", ""),
        "shelf_fill_rate": float(inspecao.get("shelf_fill_rate", 0)),
        "strategy": "by_issue",
    }

    chunks = []

    chunks.append({
        "id": f"{inspection_id}_summary",
        "texto": summary,
        "metadata": {**metadata_base, "chunk_type": "summary", "issue_type": "none"},
    })

    for i, issue in enumerate(inspecao.get("issues", [])):
        fill = inspecao.get("shelf_fill_rate", 0)
        fill_str = f"{fill:.0%}" if fill <= 1.0 else f"{fill:.0f}%"

        texto_issue = (
            f"Issue na inspeção {inspection_id} da zona {zone_id}: "
            f"{issue.get('type')} em {issue.get('location', '?')}, "
            f"severidade {issue.get('severity')}. "
            f"{issue.get('description', '')}. "
            f"Fill rate geral: {fill_str}. "
            f"Data: {ts[:10] if ts else '?'}."
        )
        chunks.append({
            "id": f"{inspection_id}_issue_{i:02d}",
            "texto": texto_issue,
            "metadata": {
                **metadata_base,
                "chunk_type": "issue",
                "issue_type": issue.get("type", "other"),
                "issue_severity": issue.get("severity", "low"),
                "issue_location": issue.get("location", ""),
            },
        })

    return chunks

def indexar_inspecao(
    inspecao: dict,
    strategy: str = CHUNK_STRATEGY,
    gerar_summary_api: bool = True,
) -> int:
    _, collection, embedding_model = inicializar_vectorstore(strategy)
    inspection_id = inspecao.get("inspection_id", "")

    existentes = collection.get(where={"inspection_id": inspection_id})
    if existentes and existentes["ids"]:
        log.info(f"Inspeção {inspection_id} já indexada ({len(existentes['ids'])} chunks). A saltar.")
        return 0

    if gerar_summary_api:
        summary = gerar_summary(inspecao)
    else:
        summary = _summary_fallback(inspecao)

    log.info(f"Summary gerado: {summary[:100]}...")

    if strategy == "by_issue":
        chunks = chunks_por_issue(inspecao, summary)
    else:
        chunks = chunks_hibrido(inspecao, summary)

    if not chunks:
        log.warning(f"Nenhum chunk gerado para {inspection_id}")
        return 0

    textos = [c["texto"] for c in chunks]
    embeddings = embedding_model.encode(textos, show_progress_bar=False).tolist()

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=textos,
        metadatas=[c["metadata"] for c in chunks],
    )

    log.info(f"Inspeção {inspection_id} indexada com {len(chunks)} chunks (strategy={strategy})")
    return len(chunks)

def indexar_pasta(pasta: str, strategy: str = CHUNK_STRATEGY) -> dict:
    pasta_path = Path(pasta)
    ficheiros = list(pasta_path.glob("*.json"))

    if not ficheiros:
        log.warning(f"Nenhum ficheiro JSON encontrado em {pasta}")
        return {"total": 0, "indexados": 0, "erros": 0}

    total_chunks = 0
    indexados = 0
    erros = 0

    for ficheiro in sorted(ficheiros):
        try:
            with open(ficheiro, encoding="utf-8") as f:
                inspecao = json.load(f)

            if "inspection_id" not in inspecao:
                continue

            chunks_adicionados = indexar_inspecao(inspecao, strategy=strategy)
            total_chunks += chunks_adicionados
            indexados += 1

        except Exception as e:
            log.error(f"Erro ao indexar {ficheiro.name}: {e}")
            erros += 1

    log.info(f"Indexação concluída: {indexados} inspeções, {total_chunks} chunks, {erros} erros")
    return {"total": len(ficheiros), "indexados": indexados, "chunks": total_chunks, "erros": erros}

def recuperar_contexto(
    query: str,
    strategy: str = CHUNK_STRATEGY,
    top_k: int = TOP_K,
    filtros: Optional[dict] = None,
) -> list[dict]:
    _, collection, embedding_model = inicializar_vectorstore(strategy)

    if collection.count() == 0:
        log.warning("Vector store vazia. Indexa inspeções primeiro.")
        return []

    query_embedding = embedding_model.encode([query], show_progress_bar=False).tolist()[0]

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if filtros:
        kwargs["where"] = filtros

    resultados = collection.query(**kwargs)

    chunks = []
    for i in range(len(resultados["ids"][0])):
        chunks.append({
            "id": resultados["ids"][0][i],
            "texto": resultados["documents"][0][i],
            "metadata": resultados["metadatas"][0][i],
            "distancia": resultados["distances"][0][i],
            "similaridade": 1 - resultados["distances"][0][i],
        })

    return chunks

def responder_query(
    query: str,
    strategy: str = CHUNK_STRATEGY,
    top_k: int = TOP_K,
) -> dict:
    log.info(f"Query: {query}")

    chunks = recuperar_contexto(query, strategy=strategy, top_k=top_k)

    if not chunks:
        return {
            "query": query,
            "resposta": "Não há inspeções indexadas no histórico. Indexa inspeções primeiro.",
            "chunks_usados": [],
            "inspection_ids_referenciados": [],
        }

    contexto_partes = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        contexto_partes.append(
            f"[{i}] inspection_id={meta.get('inspection_id')} | "
            f"zona={meta.get('zone_id')} | "
            f"data={meta.get('timestamp', '')[:16]} | "
            f"similaridade={chunk['similaridade']:.2f}\n"
            f"{chunk['texto']}"
        )

    contexto = "\n\n".join(contexto_partes)
    prompt = PROMPT_RESPOSTA_RAG.format(query=query, contexto=contexto)

    try:
        resposta = chamar_openrouter(prompt, temperature=0.3)
    except Exception as e:
        log.error(f"Erro ao gerar resposta RAG: {e}")
        resposta = (
            f"Não foi possível gerar resposta via API. "
            f"Chunks recuperados: {len(chunks)}. "
            f"Inspection IDs: {[c['metadata'].get('inspection_id') for c in chunks]}"
        )

    inspection_ids = list({
        c["metadata"].get("inspection_id")
        for c in chunks
        if c["metadata"].get("inspection_id")
    })

    return {
        "query": query,
        "resposta": resposta,
        "chunks_usados": chunks,
        "inspection_ids_referenciados": inspection_ids,
        "strategy": strategy,
        "top_k": top_k,
    }

def avaliar_recall_at_k(
    queries_ground_truth: list[dict],
    strategy: str = CHUNK_STRATEGY,
    k: int = 3,
) -> dict:
    acertos = 0
    total = len(queries_ground_truth)

    detalhes = []
    for item in queries_ground_truth:
        query = item["query"]
        relevantes = set(item["inspection_ids_relevantes"])

        chunks = recuperar_contexto(query, strategy=strategy, top_k=k)
        recuperados = {c["metadata"].get("inspection_id") for c in chunks}

        acertou = bool(relevantes & recuperados)
        acertos += int(acertou)

        detalhes.append({
            "query": query,
            "relevantes": list(relevantes),
            "recuperados": list(recuperados),
            "acertou": acertou,
        })

    recall = acertos / total if total > 0 else 0

    return {
        "strategy": strategy,
        "k": k,
        f"recall_at_{k}": recall,
        "acertos": acertos,
        "total_queries": total,
        "detalhes": detalhes,
    }

def estatisticas(strategy: str = CHUNK_STRATEGY) -> dict:
    _, collection, _ = inicializar_vectorstore(strategy)

    count = collection.count()
    if count == 0:
        return {"total_chunks": 0, "message": "Vector store vazia"}

    sample = collection.get(limit=min(count, 1000), include=["metadatas"])
    metadatas = sample["metadatas"]

    zonas = {}
    statuses = {}
    for m in metadatas:
        z = m.get("zone_id", "?")
        zonas[z] = zonas.get(z, 0) + 1
        s = m.get("overall_status", "?")
        statuses[s] = statuses.get(s, 0) + 1

    return {
        "strategy": strategy,
        "total_chunks": count,
        "chunks_por_zona": zonas,
        "chunks_por_status": statuses,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Memory — Memória de inspeções histórica")
    parser.add_argument("--strategy", choices=["hybrid", "by_issue"], default=CHUNK_STRATEGY,
                        help="Estratégia de chunking a utilizar")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    sub = parser.add_subparsers(dest="cmd")

    p_idx = sub.add_parser("index", help="Indexa um ficheiro individual de inspeção")
    p_idx.add_argument("ficheiro", help="Caminho para o JSON de inspeção")
    p_idx.add_argument("--no-api", action="store_true", help="Usa o sumário fallback local")

    p_idxa = sub.add_parser("index-all", help="Indexa de forma massiva uma diretoria completa")
    p_idxa.add_argument("pasta", help="Pasta com os ficheiros de inspeção")
    p_idxa.add_argument("--no-api", action="store_true")

    p_q = sub.add_parser("query", help="Efetua uma pesquisa contextual assistida por RAG")
    p_q.add_argument("texto", help="Query do gestor em linguagem natural")

    sub.add_parser("stats", help="Mostra estatísticas detalhadas do ChromaDB")

    args = parser.parse_args()
    strategy = args.strategy

    if args.cmd == "index":
        with open(args.ficheiro, encoding="utf-8") as f:
            inspecao = json.load(f)
        n = indexar_inspecao(inspecao, strategy=strategy, gerar_summary_api=not args.no_api)
        print(f"Sucesso: {n} chunks adicionados.")

    elif args.cmd == "index-all":
        resultado = indexar_pasta(args.pasta, strategy=strategy)
        print(json.dumps(resultado, indent=2))

    elif args.cmd == "query":
        resultado = responder_query(args.texto, strategy=strategy, top_k=args.top_k)
        print(f"\n{'='*60}")
        print(f"Query: {resultado['query']}")
        print(f"{'='*60}")
        print(f"\nResposta:\n{resultado['resposta']}")
        print(f"\nInspeções referenciadas: {', '.join(resultado['inspection_ids_referenciados'])}")
        print(f"\nChunks usados ({len(resultado['chunks_usados'])}):")
        for c in resultado["chunks_usados"]:
            print(f"  [{c['similaridade']:.2f}] {c['id']} — {c['texto'][:80]}...")

    elif args.cmd == "stats":
        stats = estatisticas(strategy=strategy)
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    else:
        parser.print_help()