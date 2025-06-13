#!/usr/bin/env python3
"""
Servidor RAG MCP Agéntico - Versión Cloud
Mantiene arquitectura original pero optimizado para Cloud Run
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Vector DB imports (solo in-memory)
import chromadb
from chromadb.config import Settings
import openai

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración global (mantener igual)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")

# Modelos de datos (mantener iguales)
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    max_results: int = 5

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    doc_id: Optional[str] = None

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_used: Dict[str, Any]

class AgenticRAGEngine:
    """Motor de RAG Agéntico con capacidades avanzadas - Versión Cloud"""
    
    def __init__(self):
        # Usar base de conocimientos en memoria para cloud
        self.knowledge_base = []
        self.vector_db = None
        self.collection = None
        self.openai_client = None
        
        if OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL
            )
    
    async def initialize(self):
        """Inicializar conexiones a bases de datos vectoriales"""
        try:
            # Siempre usar in-memory para cloud
            logger.info("Usando ChromaDB en modo in-memory para Cloud Run")
            self.vector_db = chromadb.Client(settings=Settings(is_persistent=False))

            # Crear o obtener colección
            self.collection = self.vector_db.get_or_create_collection(
                "rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Colección 'rag_documents' asegurada (creada u obtenida).")

        except Exception as e:
            logger.error(f"Error fatal inicializando vector DB: {e}", exc_info=True)
            raise
    
    def _simple_embedding(self, text: str) -> List[float]:
        """Embeddings simples usando hash + normalización"""
        # Versión ultra-simple para evitar dependencias ML pesadas
        words = text.lower().split()
        # Crear vector de 384 dimensiones (como sentence-transformers)
        vector = [0.0] * 384
        
        for i, word in enumerate(words[:100]):  # Limitar a 100 palabras
            hash_val = hash(word) % 384
            vector[hash_val] += 1.0 / (i + 1)  # Peso decreciente por posición
        
        # Normalizar
        norm = sum(x*x for x in vector) ** 0.5
        if norm > 0:
            vector = [x/norm for x in vector]
        
        return vector
    
    async def add_document(self, content: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> str:
        """Agregar documento a la base de conocimiento"""
        try:
            if not doc_id:
                doc_id = f"doc_{len(self.knowledge_base) + 1}"
            
            # Agregar a base de conocimientos en memoria
            doc_data = {
                "id": doc_id,
                "content": content,
                "metadata": metadata or {}
            }
            self.knowledge_base.append(doc_data)
            
            # Generar embedding simple
            embedding = self._simple_embedding(content)
            
            # Agregar a ChromaDB
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
            
            logger.info(f"Documento agregado: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error agregando documento: {e}")
            raise
    
    async def semantic_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Búsqueda semántica en la base de conocimiento"""
        try:
            # Búsqueda híbrida: keyword + vector
            query_lower = query.lower()
            query_embedding = self._simple_embedding(query)
            
            # Primero búsqueda por palabras clave para casos especiales
            keyword_matches = []
            for doc in self.knowledge_base:
                content_lower = doc["content"].lower()
                metadata = doc.get("metadata", {})
                
                # Detectar mensaje especial
                secret_keywords = ["mensaje", "secreto", "vicky", "pedro", "especial", "amor"]
                if any(keyword in query_lower for keyword in secret_keywords):
                    if metadata.get("type") == "mensaje_especial" or any(keyword in content_lower for keyword in secret_keywords):
                        keyword_matches.append({
                            'content': doc["content"],
                            'metadata': metadata,
                            'relevance_score': 1.0,
                            'rank': 1
                        })
                        break
            
            if keyword_matches:
                return keyword_matches[:max_results]
            
            # Si no hay matches especiales, usar ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            if results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'relevance_score': max(0, 1 - distance),
                        'rank': i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return []
    
    async def agentic_query_expansion(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Expansión agéntica de consultas usando LLM - MANTENER FUNCIONALIDAD"""
        if not self.openai_client:
            return [query]
        
        try:
            system_prompt = """Eres un experto en expandir consultas para mejorar la recuperación de información sobre vinos.
            Genera 2-3 variaciones de la consulta original."""
            
            user_prompt = f"Consulta: {query}\nGenera variaciones para búsqueda de vinos."
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            expanded_text = response.choices[0].message.content
            # Extraer consultas de la respuesta
            lines = [line.strip() for line in expanded_text.split('\n') if line.strip()]
            expanded_queries = [query] + lines[:3]  # Original + máximo 3 expansiones
            
            return expanded_queries[:4]
            
        except Exception as e:
            logger.error(f"Error en expansión de consulta: {e}")
            return [query]
    
    async def generate_answer(self, query: str, sources: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Generar respuesta usando LLM con fuentes recuperadas"""
        if not self.openai_client:
            return f"Basado en {len(sources)} fuentes encontradas para: '{query}'"
        
        try:
            sources_text = "\n\n".join([
                f"Fuente {i+1}:\n{source['content']}"
                for i, source in enumerate(sources[:3])
            ])
            
            system_prompt = """Eres Sumy, un sumiller profesional. Responde basándote SOLO en las fuentes proporcionadas.
            Si no hay información suficiente, dilo claramente."""
            
            user_prompt = f"""Pregunta: {query}

Fuentes:
{sources_text}

Responde como sumiller profesional basándote únicamente en las fuentes."""
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return f"Error generando respuesta basada en {len(sources)} fuentes para: '{query}'"
    
    async def agentic_rag_query(self, query: str, context: Dict[str, Any] = None, max_results: int = 5) -> RAGResponse:
        """Consulta RAG agéntica completa - MANTENER ARQUITECTURA"""
        try:
            # 1. Expansión agéntica de consulta
            expanded_queries = await self.agentic_query_expansion(query, context)
            
            # 2. Búsqueda semántica con múltiples queries
            all_sources = []
            for exp_query in expanded_queries:
                sources = await self.semantic_search(exp_query, max_results // len(expanded_queries) + 1)
                all_sources.extend(sources)
            
            # 3. Deduplicar y ordenar por relevancia
            unique_sources = {}
            for source in all_sources:
                content_hash = hash(source['content'])
                if content_hash not in unique_sources or source['relevance_score'] > unique_sources[content_hash]['relevance_score']:
                    unique_sources[content_hash] = source
            
            final_sources = sorted(unique_sources.values(), key=lambda x: x['relevance_score'], reverse=True)[:max_results]
            
            # 4. Generar respuesta contextual
            answer = await self.generate_answer(query, final_sources, context)
            
            return RAGResponse(
                answer=answer,
                sources=final_sources,
                context_used={
                    "original_query": query,
                    "expanded_queries": expanded_queries,
                    "context": context or {}
                }
            )
            
        except Exception as e:
            logger.error(f"Error en consulta RAG agéntica: {e}")
            return RAGResponse(
                answer=f"Error procesando consulta: {str(e)}",
                sources=[],
                context_used={"query": query, "error": str(e)}
            )

# Instancia global del motor RAG
rag_engine = AgenticRAGEngine()

# FastAPI para HTTP
app = FastAPI(title="Agentic RAG MCP Server - Cloud", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Inicializar al arrancar - optimizado para Cloud Run"""
    logger.info("🚀 Iniciando servidor RAG Cloud...")
    
    # Inicializar motor RAG
    await rag_engine.initialize()
    
    # Cargar datos en background para evitar timeout de startup
    asyncio.create_task(load_knowledge_base_background())

async def load_knowledge_base_background():
    """Cargar base de conocimientos en background"""
    try:
        knowledge_dir = Path("knowledge_base")
        if knowledge_dir.exists():
            await load_documents_from_directory(knowledge_dir)
        else:
            logger.warning("Directorio knowledge_base no encontrado")
    except Exception as e:
        logger.error(f"Error cargando base de conocimientos: {e}")

async def load_documents_from_directory(knowledge_dir: Path):
    """Función auxiliar para cargar documentos - mantener lógica original"""
    # Cargar archivos de texto
    for file_path in knowledge_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "mensaje" in file_path.name.lower() or "especial" in file_path.name.lower():
                metadata = {"source": file_path.name, "type": "mensaje_especial"}
            else:
                metadata = {"source": file_path.name, "type": "text"}
                
            await rag_engine.add_document(content, metadata, file_path.stem)
            logger.info(f"Documento de texto cargado: {file_path.name}")
        except Exception as e:
            logger.error(f"Error cargando archivo de texto {file_path}: {e}")
    
    # Cargar archivos JSON (vinos) - MANTENER LÓGICA ORIGINAL
    for file_path in knowledge_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Si es el archivo de vinos
            if file_path.name == "vinos.json" and isinstance(data, list):
                logger.info(f"Cargando {len(data)} vinos desde {file_path.name}")
                for i, vino in enumerate(data):
                    # Crear contenido estructurado para cada vino - MANTENER IGUAL
                    content = f"""Vino: {vino.get('name', 'Sin nombre')}
Tipo: {vino.get('type', 'Sin tipo')}
Región: {vino.get('region', 'Sin región')}
Año: {vino.get('vintage', 'Sin año')}
Precio: {vino.get('price', 'Sin precio')}€
Stock: {vino.get('stock', 'Sin stock')} unidades
Maridaje: {vino.get('pairing', 'Sin maridaje')}
Descripción: {vino.get('description', 'Sin descripción')}
Puntuación: {vino.get('rating', 'Sin puntuación')}/100"""
                    
                    # Metadata rica para búsquedas - MANTENER IGUAL
                    metadata = {
                        "source": file_path.name,
                        "type": "vino",
                        "name": vino.get('name', ''),
                        "wine_type": vino.get('type', ''),
                        "region": vino.get('region', ''),
                        "vintage": vino.get('vintage', ''),
                        "price": vino.get('price', ''),
                        "rating": vino.get('rating', ''),
                        "pairing": vino.get('pairing', ''),
                        "index": i
                    }
                    
                    doc_id = f"vino_{i}_{vino.get('name', 'sin_nombre').replace(' ', '_')}"
                    await rag_engine.add_document(content, metadata, doc_id)
                
                logger.info(f"✅ {len(data)} vinos cargados exitosamente desde {file_path.name}")
                
        except Exception as e:
            logger.error(f"Error cargando archivo JSON {file_path}: {e}")

@app.get("/")
async def root():
    return {"message": "Agentic RAG MCP Server - Cloud", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Verificación de salud"""
    return {
        "status": "healthy", 
        "vector_db": "chroma_in_memory", 
        "version": "cloud",
        "documents_loaded": len(rag_engine.knowledge_base)
    }

@app.post("/query")
async def query_rag_mcp(query_data: QueryRequest):
    """Endpoint de consulta RAG - MANTENER FUNCIONALIDAD ORIGINAL"""
    start_total = time.time()
    logger.info(f"Received query: {query_data.query}")

    try:
        # Asegurar que rag_engine esté inicializado
        if rag_engine.collection is None:
            await rag_engine.initialize()

        # Realizar consulta RAG agéntica completa
        result = await rag_engine.agentic_rag_query(
            query_data.query,
            query_data.context,
            query_data.max_results
        )

        end_total = time.time()
        logger.info(f"Tiempo total de la solicitud /query: {end_total - start_total:.4f}s")

        return {
            "answer": result.answer,
            "sources": result.sources,
            "context_used": result.context_used
        }

    except Exception as e:
        logger.error(f"Error en /query: {e}")
        return {
            "answer": f"Error procesando consulta: {str(e)}",
            "sources": [],
            "context_used": {"query": query_data.query, "error": str(e)}
        }

@app.post("/documents")
async def add_document_endpoint(request: DocumentRequest):
    """Endpoint HTTP para agregar documentos"""
    doc_id = await rag_engine.add_document(
        request.content,
        request.metadata,
        request.doc_id
    )
    return {"doc_id": doc_id, "status": "added"}

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", "8080"))  # Cloud Run usa 8080 por defecto
    uvicorn.run(app, host="0.0.0.0", port=PORT) 