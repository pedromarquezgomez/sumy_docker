#!/usr/bin/env python3
"""
Servidor RAG MCP Mejorado para Cloud Run
Versión híbrida: simple en dependencias, completo en datos
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pathlib import Path

# Modelos
class QueryRequest(BaseModel):
    query: str
    max_results: int = 5

class RAGResponse(BaseModel):
    answer: str
    sources: list
    context_used: dict

# App FastAPI
app = FastAPI(title="Enhanced RAG MCP Server", version="2.0.0")

# Base de conocimientos mejorada
knowledge_base = []

def load_wine_data():
    """Cargar datos de vinos desde JSON"""
    wine_file = Path("knowledge_base/vinos.json")
    if wine_file.exists():
        try:
            with open(wine_file, 'r', encoding='utf-8') as f:
                wines = json.load(f)
                print(f"✅ Cargados {len(wines)} vinos desde {wine_file}")
                
                for i, wine in enumerate(wines):
                    content = f"""Vino: {wine.get('name', 'Sin nombre')}
Tipo: {wine.get('type', 'Sin tipo')}
Región: {wine.get('region', 'Sin región')}
Año: {wine.get('vintage', 'Sin año')}
Precio: {wine.get('price', 'Sin precio')}€
Stock: {wine.get('stock', 'Sin stock')} unidades
Maridaje: {wine.get('pairing', 'Sin maridaje')}
Descripción: {wine.get('description', 'Sin descripción')}
Puntuación: {wine.get('rating', 'Sin puntuación')}/100"""
                    
                    knowledge_base.append({
                        "id": f"wine_{i}",
                        "content": content,
                        "metadata": {
                            "type": "vino",
                            "name": wine.get('name', ''),
                            "wine_type": wine.get('type', ''),
                            "region": wine.get('region', ''),
                            "vintage": wine.get('vintage', ''),
                            "price": wine.get('price', 0),
                            "rating": wine.get('rating', 0),
                            "pairing": wine.get('pairing', ''),
                            "description": wine.get('description', ''),
                            "index": i
                        }
                    })
        except Exception as e:
            print(f"❌ Error cargando vinos: {e}")

def load_special_message():
    """Cargar mensaje especial"""
    message_file = Path("knowledge_base/mensaje_especial.txt")
    if message_file.exists():
        try:
            with open(message_file, 'r', encoding='utf-8') as f:
                message_content = f.read()
                print(f"✅ Cargado mensaje especial desde {message_file}")
                
                knowledge_base.append({
                    "id": "mensaje_especial",
                    "content": message_content,
                    "metadata": {
                        "type": "mensaje_especial",
                        "name": "Mensaje para Vicky",
                        "keywords": ["mensaje", "especial", "vicky", "pedro", "amor", "secreto", "romántico"]
                    }
                })
        except Exception as e:
            print(f"❌ Error cargando mensaje especial: {e}")

def load_wine_theory():
    """Cargar teoría de sumillería"""
    theory_file = Path("knowledge_base/teoria_sumiller.txt")
    if theory_file.exists():
        try:
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory_content = f.read()
                print(f"✅ Cargada teoría de sumillería desde {theory_file}")
                
                # Dividir la teoría en secciones
                sections = theory_content.split('\n\n')
                for i, section in enumerate(sections):
                    if section.strip():
                        knowledge_base.append({
                            "id": f"theory_{i}",
                            "content": section.strip(),
                            "metadata": {
                                "type": "teoria",
                                "section": i,
                                "keywords": ["sumiller", "vino", "cata", "teoría"]
                            }
                        })
        except Exception as e:
            print(f"❌ Error cargando teoría: {e}")

def search_knowledge_base(query: str, max_results: int = 5):
    """Búsqueda mejorada en la base de conocimientos"""
    query_lower = query.lower()
    matching_items = []
    
    # Palabras clave especiales para mensaje secreto
    secret_keywords = ["mensaje", "secreto", "vicky", "pedro", "especial", "amor", "romántico"]
    has_secret_keyword = any(keyword in query_lower for keyword in secret_keywords)
    
    for item in knowledge_base:
        content_lower = item["content"].lower()
        metadata = item.get("metadata", {})
        
        score = 0
        
        # Prioridad máxima para mensaje especial
        if has_secret_keyword and metadata.get("type") == "mensaje_especial":
            score = 1.0
        else:
            # Búsqueda por palabras clave en contenido
            query_words = query_lower.split()
            for word in query_words:
                if word in content_lower:
                    score += 0.2
                
                # Búsqueda en metadatos específicos
                if word in str(metadata.get("name", "")).lower():
                    score += 0.3
                if word in str(metadata.get("wine_type", "")).lower():
                    score += 0.3
                if word in str(metadata.get("region", "")).lower():
                    score += 0.3
                if word in str(metadata.get("pairing", "")).lower():
                    score += 0.2
        
        if score > 0:
            matching_items.append({
                "content": item["content"],
                "metadata": metadata,
                "relevance_score": min(score, 1.0),
                "rank": 0  # Se asignará después
            })
    
    # Ordenar por relevancia
    matching_items.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Asignar ranks
    for i, item in enumerate(matching_items):
        item["rank"] = i + 1
    
    return matching_items[:max_results]

async def load_data_async():
    """Cargar datos de forma asíncrona"""
    print("📦 Cargando base de conocimientos...")
    
    load_wine_data()
    load_special_message()
    load_wine_theory()
    
    print(f"✅ Base de conocimientos cargada: {len(knowledge_base)} documentos")

# Variable para controlar si los datos están cargados
data_loaded = False

@app.on_event("startup")
async def startup_event():
    """Iniciar servidor inmediatamente y cargar datos en background"""
    global data_loaded
    print("🚀 Iniciando Enhanced RAG MCP Server...")
    
    # Cargar datos en background
    import asyncio
    asyncio.create_task(load_data_background())

async def load_data_background():
    """Cargar datos en background"""
    global data_loaded
    await load_data_async()
    data_loaded = True

@app.get("/")
async def root():
    return {
        "message": "Enhanced RAG MCP Server running", 
        "status": "ok",
        "documents_loaded": len(knowledge_base)
    }

@app.get("/health")
async def health():
    global data_loaded
    return {
        "status": "healthy", 
        "version": "2.0.0",
        "documents_loaded": len(knowledge_base),
        "data_ready": data_loaded
    }

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Endpoint de consulta mejorado"""
    
    global data_loaded
    
    # Si los datos no están listos, responder temporalmente
    if not data_loaded:
        return RAGResponse(
            answer="El sistema está cargando la base de conocimientos. Por favor, inténtalo en unos segundos.",
            sources=[],
            context_used={"query": request.query, "total_found": 0, "status": "loading"}
        )
    
    # Búsqueda en base de conocimientos
    matching_items = search_knowledge_base(request.query, request.max_results)
    
    # Generar respuesta
    if matching_items:
        # Verificar si es mensaje especial
        if matching_items[0].get("metadata", {}).get("type") == "mensaje_especial":
            answer = f"🍷💕 **Mensaje Especial Descubierto** 💕🍷\n\n{matching_items[0]['content']}\n\n---\n*Como sumiller, debo decir que este es el maridaje más hermoso que he visto. ¡Felicidades a los dos! 🥂*"
        else:
            answer = f"Encontré {len(matching_items)} resultado(s) para '{request.query}':\n\n"
            for i, item in enumerate(matching_items, 1):
                if item.get("metadata", {}).get("type") == "vino":
                    # Formato especial para vinos
                    metadata = item.get("metadata", {})
                    answer += f"{i}. **{metadata.get('name', 'Vino desconocido')}**\n"
                    answer += f"   • Tipo: {metadata.get('wine_type', 'N/A')}\n"
                    answer += f"   • Región: {metadata.get('region', 'N/A')}\n"
                    answer += f"   • Precio: {metadata.get('price', 'N/A')}€\n"
                    answer += f"   • Puntuación: {metadata.get('rating', 'N/A')}/100\n"
                    answer += f"   • Maridaje: {metadata.get('pairing', 'N/A')}\n\n"
                else:
                    # Formato general
                    answer += f"{i}. {item['content'][:200]}{'...' if len(item['content']) > 200 else ''}\n\n"
    else:
        answer = f"No encontré resultados específicos para '{request.query}'. Puedes preguntar sobre vinos específicos, regiones, maridajes, o el mensaje especial."
    
    return RAGResponse(
        answer=answer,
        sources=matching_items,
        context_used={"query": request.query, "total_found": len(matching_items)}
    )

@app.post("/documents")
async def add_document(content: str, metadata: dict = None):
    """Añadir documento dinámicamente"""
    new_id = f"doc_{len(knowledge_base)}"
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