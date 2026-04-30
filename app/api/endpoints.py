from fastapi import APIRouter, HTTPException, Query, Body, Request
from typing import List, Dict, Any, Optional

from app.services.vector_db import VectorDB
from app.services.document_db import DocumentDB
from app.services.llm_client import LLMClient
from app.models.schemas import InsertReq, DocInsertReq, DocSearchReq
from app.utils.chunker import chunk_text
from app.utils.init_demo import load_demo

router = APIRouter()

DIMS = 16
db = VectorDB(DIMS)
doc_db = DocumentDB()
llm = LLMClient()

# Initialize demo data
load_demo(db)

@router.get("/search")
def search(v: str, k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    try:
        q = [float(x) for x in v.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid vector format")

    if len(q) != DIMS:
        return {"error": f"need {DIMS}D vector"}

    out = db.search(q, k, metric, algo)
    return out

@router.post("/insert")
def insert(req: InsertReq):
    if len(req.embedding) != DIMS:
        return {"error": "invalid body"}

    dist_metric = "cosine"
    id_ = db.insert(req.metadata, req.category, req.embedding, dist_metric)
    return {"id": id_}

@router.delete("/delete/{id}")
def delete(id: int):
    ok = db.remove(id)
    return {"ok": ok}

@router.get("/items")
def items():
    all_items = db.all()
    return [{"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.emb} for v in all_items]

@router.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    try:
        q = [float(x) for x in v.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid vector format")

    if len(q) != DIMS:
        return {"error": f"need {DIMS}D vector"}

    return db.benchmark(q, k, metric)

@router.get("/hnsw-info")
def hnsw_info():
    return db.hnsw_info()

@router.post("/doc/insert")
def doc_insert(req: DocInsertReq):
    if not req.title or not req.text:
        return {"error": "need title and text"}

    chunks = chunk_text(req.text, 250, 30)
    ids = []

    for i, c in enumerate(chunks):
        emb = llm.embed(c)
        if not emb:
            return {"error": "HF Embedding unavailable. Check HF API reachability"}

        chunk_title = f"{req.title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else req.title
        id_ = doc_db.insert(chunk_title, c, emb)
        ids.append(id_)

    return {
        "ids": ids,
        "chunks": len(chunks),
        "dims": doc_db.get_dims()
    }

@router.delete("/doc/delete/{id}")
def doc_delete(id: int):
    ok = doc_db.remove(id)
    return {"ok": ok}

@router.get("/doc/list")
def doc_list():
    docs = doc_db.all()
    out = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        words = len(d.text.split())
        out.append({
            "id": d.id,
            "title": d.title,
            "preview": preview,
            "words": words
        })
    return out

@router.post("/doc/search")
def doc_search(req: DocSearchReq):
    if not req.question:
        return {"error": "need question"}

    q_emb = llm.embed(req.question)
    if not q_emb:
        return {"error": "Embedding unavailable"}

    hits = doc_db.search(q_emb, req.k)
    return {
        "contexts": [
            {
                "id": doc.id,
                "title": doc.title,
                "distance": round(d, 4)
            }
            for d, doc in hits
        ]
    }

@router.post("/doc/ask")
def doc_ask(req: DocSearchReq):
    if not req.question:
        return {"error": "need question"}

    # Step 1: embed question
    q_emb = llm.embed(req.question)
    if not q_emb:
        return {"error": "Embedding unavailable"}

    # Step 2: retrieve
    hits = doc_db.search(q_emb, req.k)

    # Step 3: build prompt
    ctx_str = ""
    for i, (d, doc) in enumerate(hits):
        ctx_str += f"[{i+1}] {doc.title}:\n{doc.text}\n\n"

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        f"Context:\n{ctx_str}"
        f"Question: {req.question}\n\n"
        "Answer:"
    )

    # Step 4: generate with Groq
    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "model": llm.gen_model,
        "contexts": [
            {
                "id": doc.id,
                "title": doc.title,
                "text": doc.text,
                "distance": round(d, 4)
            }
            for d, doc in hits
        ],
        "docCount": doc_db.size()
    }

@router.get("/status")
def status():
    return {
        "ollamaAvailable": llm.is_available(), 
        "embedModel": llm.embed_model,
        "genModel": llm.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size()
    }

@router.get("/stats")
def stats():
    return {
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"]
    }
