#!/usr/bin/env python3
"""
Servidor RAG MCP Simple para Cloud Run
Versión mínima sin dependencias externas complejas
"""

import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Modelos simples
class QueryRequest(BaseModel):
    query: str
    max_results: int = 5

class RAGResponse(BaseModel):
    answer: str
    sources: list
    context_used: dict

# App FastAPI
app = FastAPI(title="Simple RAG MCP Server", version="1.0.0")

# Datos en memoria simple
knowledge_base = [
    {"id": "1", "content": "Vino tinto Tempranillo de Rioja, ideal para carnes rojas", "metadata": {"type": "vino", "region": "Rioja"}},
    {"id": "2", "content": "Vino blanco Albariño de Rías Baixas, perfecto para mariscos", "metadata": {"type": "vino", "region": "Rías Baixas"}},
    {"id": "3", "content": "Cava Brut Nature, excelente para aperitivos", "metadata": {"type": "vino", "region": "Penedès"}},
]

@app.get("/")
async def root():
    return {"message": "Simple RAG MCP Server running", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Endpoint de consulta simple"""
    
    # Búsqueda simple por palabras clave
    query_lower = request.query.lower()
    matching_items = []
    
    for item in knowledge_base:
        content_lower = item["content"].lower()
        if any(word in content_lower for word in query_lower.split()):
            matching_items.append({
                "content": item["content"],
                "metadata": item["metadata"],
                "relevance_score": 0.8,
                "rank": len(matching_items) + 1
            })
    
    # Limitar resultados
    matching_items = matching_items[:request.max_results]
    
    # Respuesta simple
    if matching_items:
        answer = f"Encontré {len(matching_items)} resultado(s) para '{request.query}':\n\n"
        for i, item in enumerate(matching_items, 1):
            answer += f"{i}. {item['content']}\n"
    else:
        answer = f"No encontré resultados específicos para '{request.query}'. Puedes preguntar sobre vinos de Rioja, Rías Baixas o Cava."
    
    return RAGResponse(
        answer=answer,
        sources=matching_items,
        context_used={"query": request.query, "total_found": len(matching_items)}
    )

@app.post("/documents")
async def add_document(content: str, metadata: dict = None):
    """Añadir documento simple"""
    new_id = str(len(knowledge_base) + 1)
    new_doc = {
        "id": new_id,
        "content": content,
        "metadata": metadata or {}
    }
    knowledge_base.append(new_doc)
    return {"doc_id": new_id, "status": "added", "total_docs": len(knowledge_base)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 