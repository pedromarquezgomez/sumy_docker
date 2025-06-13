#!/usr/bin/env python3
"""
Servidor RAG MCP Agéntico
Implementa un sistema completo de RAG con capacidades agénticas usando MCP
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# MCP SDK imports
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Vector DB imports
import chromadb
from chromadb.config import Settings
import openai
from sentence_transformers import SentenceTransformer

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración global
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

# Modelos de datos
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
    """Motor de RAG Agéntico con capacidades avanzadas"""
    
    def __init__(self):
        # Cargar el modelo desde la caché local
        model_name = 'all-MiniLM-L6-v2'
        model_path = f'./model_cache/sentence-transformers_{model_name.replace("/", "_")}'
        
        if os.path.exists(model_path):
            logger.info(f"Cargando modelo desde la caché local: {model_path}")
            # Usar 'use_auth_token=False' y 'trust_remote_code=True' puede ayudar a evitar llamadas de red
            self.embedding_model = SentenceTransformer(model_path, use_auth_token=False, trust_remote_code=True)
        else:
            logger.warning(f"El modelo no se encontró en {model_path}, intentando cargar por nombre.")
            # Fallback para desarrollo local
            self.embedding_model = SentenceTransformer(model_name)
            
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
            if os.getenv("ENVIRONMENT") == "cloud":
                logger.info("Usando ChromaDB en modo in-memory para Cloud Run")
                self.vector_db = chromadb.Client(settings=Settings(is_persistent=False))
            else: # Entorno local
                logger.info(f"Conectando a ChromaDB en {CHROMA_HOST}:{CHROMA_PORT}")
                self.vector_db = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

            # Crear o obtener colección
            self.collection = self.vector_db.get_or_create_collection(
                "rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Colección 'rag_documents' asegurada (creada u obtenida).")

        except Exception as e:
            logger.error(f"Error fatal inicializando vector DB: {e}", exc_info=True)
            raise
    
    def _embed_text(self, text: str) -> List[float]:
        """Generar embeddings para texto"""
        return self.embedding_model.encode(text).tolist()
    
    async def add_document(self, content: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> str:
        """Agregar documento a la base de conocimiento"""
        try:
            if not doc_id:
                doc_id = f"doc_{len(self.collection.get()['ids']) + 1}"
            
            embedding = self._embed_text(content)
            
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
            query_embedding = self._embed_text(query)
            
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
                        'relevance_score': 1 - distance,  # Convertir distancia a score
                        'rank': i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return []
    
    async def agentic_query_expansion(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Expansión agéntica de consultas usando LLM"""
        if not self.openai_client:
            return [query]  # Fallback si no hay OpenAI
        
        try:
            system_prompt = """Eres un experto en expandir consultas para mejorar la recuperación de información.
            Dado una consulta del usuario, genera variaciones y reformulaciones que puedan ayudar a encontrar información relevante.
            Incluye sinónimos, términos relacionados y diferentes formas de expresar la misma pregunta.
            
            Responde con un JSON que contenga una lista de consultas expandidas."""
            
            user_prompt = f"""
            Consulta original: "{query}"
            Contexto adicional: {json.dumps(context or {}, indent=2)}
            
            Genera 3-5 variaciones de esta consulta para mejorar la búsqueda.
            """
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            
            # Intentar parsear JSON
            try:
                expanded_queries = json.loads(result)
                if isinstance(expanded_queries, list):
                    return [query] + expanded_queries[:4]  # Original + 4 expansiones max
                elif isinstance(expanded_queries, dict) and 'queries' in expanded_queries:
                    return [query] + expanded_queries['queries'][:4]
            except:
                # Si falla el parsing, usar la consulta original
                pass
            
            return [query]
            
        except Exception as e:
            logger.error(f"Error en expansión de consulta: {e}")
            return [query]
    
    async def generate_answer(self, query: str, sources: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Generar respuesta usando LLM con fuentes recuperadas"""
        if not self.openai_client:
            # Fallback sin LLM
            return f"Basado en {len(sources)} fuentes encontradas para: '{query}'"
        
        try:
            # Construir contexto de fuentes
            sources_text = "\n\n".join([
                f"Fuente {i+1} (relevancia: {source.get('relevance_score', 0):.2f}):\n{source['content']}"
                for i, source in enumerate(sources[:3])  # Máximo 3 fuentes
            ])
            
            system_prompt = """Eres un asistente inteligente que responde preguntas basándose en fuentes proporcionadas.
            Usa SOLAMENTE la información de las fuentes para responder. Si la información no está en las fuentes, dilo claramente.
            Cita las fuentes relevantes en tu respuesta."""
            
            user_prompt = f"""
            Pregunta: {query}
            
            Contexto adicional: {json.dumps(context or {}, indent=2)}
            
            Fuentes disponibles:
            {sources_text}
            
            Responde la pregunta basándote únicamente en las fuentes proporcionadas.
            """
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return f"Error generando respuesta basada en {len(sources)} fuentes para: '{query}'"
    
    async def agentic_rag_query(self, query: str, context: Dict[str, Any] = None, max_results: int = 5) -> RAGResponse:
        """Consulta RAG agéntica completa"""
        try:
            # 1. Expansión agéntica de consulta
            expanded_queries = await self.agentic_query_expansion(query, context)
            logger.info(f"Consultas expandidas: {expanded_queries}")
            
            # 2. Búsqueda semántica multi-consulta
            all_sources = []
            for exp_query in expanded_queries:
                sources = await self.semantic_search(exp_query, max_results=3)
                all_sources.extend(sources)
            
            # 3. Deduplicación y ranking
            seen_content = set()
            unique_sources = []
            for source in all_sources:
                content_hash = hash(source['content'][:100])  # Hash de los primeros 100 chars
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_sources.append(source)
            
            # Ordenar por relevancia
            unique_sources.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            top_sources = unique_sources[:max_results]
            
            # 4. Generación de respuesta
            answer = await self.generate_answer(query, top_sources, context)
            
            return RAGResponse(
                answer=answer,
                sources=top_sources,
                context_used=context or {}
            )
            
        except Exception as e:
            logger.error(f"Error en consulta RAG agéntica: {e}")
            return RAGResponse(
                answer=f"Error procesando consulta: {str(e)}",
                sources=[],
                context_used=context or {}
            )

# Instancia global del motor RAG
rag_engine = AgenticRAGEngine()

# Servidor MCP
mcp_server = Server("agentic-rag-server")

@mcp_server.list_tools()
async def list_tools() -> List[types.Tool]:
    """Lista todas las herramientas disponibles"""
    return [
        # === HERRAMIENTAS RAG BÁSICAS ===
        types.Tool(
            name="buscar_vinos",
            description="Busca vinos en la base de datos usando búsqueda semántica avanzada",
            inputSchema={
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "Consulta para buscar vinos (ej: 'vino tinto para asado')"
                    },
                    "max_resultados": {
                        "type": "integer",
                        "description": "Número máximo de vinos a devolver (1-20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["consulta"]
            }
        ),
        types.Tool(
            name="agregar_documento",
            description="Agrega un nuevo documento (vino, teoría, etc.) a la base de conocimientos",
            inputSchema={
                "type": "object",
                "properties": {
                    "contenido": {
                        "type": "string",
                        "description": "Contenido del documento a agregar"
                    },
                    "metadatos": {
                        "type": "object",
                        "description": "Metadatos del documento (tipo, fuente, etc.)",
                        "default": {}
                    },
                    "id_documento": {
                        "type": "string",
                        "description": "ID único para el documento (opcional)"
                    }
                },
                "required": ["contenido"]
            }
        ),
        
        # === HERRAMIENTAS DE MARIDAJE ===
        types.Tool(
            name="sugerir_maridaje",
            description="Sugiere vinos ideales para maridar con platos específicos",
            inputSchema={
                "type": "object",
                "properties": {
                    "plato": {
                        "type": "string",
                        "description": "Descripción del plato o tipo de comida (ej: 'paella', 'asado', 'pescado al horno')"
                    },
                    "ocasion": {
                        "type": "string",
                        "description": "Tipo de ocasión (casual, formal, romántica, celebración)",
                        "enum": ["casual", "formal", "romantica", "celebracion", "familiar", "negocios"],
                        "default": "casual"
                    },
                    "presupuesto_max": {
                        "type": "number",
                        "description": "Presupuesto máximo en euros (opcional)"
                    }
                },
                "required": ["plato"]
            }
        ),
        types.Tool(
            name="crear_menu_maridaje",
            description="Crea un menú completo con maridajes de vinos para cada plato",
            inputSchema={
                "type": "object",
                "properties": {
                    "platos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de platos para el menú (mínimo 2, máximo 6)"
                    },
                    "estilo_evento": {
                        "type": "string",
                        "description": "Estilo del evento",
                        "enum": ["elegante", "casual", "rustico", "moderno", "tradicional"],
                        "default": "casual"
                    },
                    "num_comensales": {
                        "type": "integer",
                        "description": "Número de comensales",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 4
                    }
                },
                "required": ["platos"]
            }
        ),
        
        # === HERRAMIENTAS DE ANÁLISIS DE VINOS ===
        types.Tool(
            name="analizar_vino",
            description="Proporciona análisis detallado de las características de un vino específico",
            inputSchema={
                "type": "object",
                "properties": {
                    "nombre_vino": {
                        "type": "string",
                        "description": "Nombre exacto del vino a analizar"
                    },
                    "aspectos": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["aromas", "sabores", "estructura", "maridajes", "temperatura", "decantacion", "guarda"]
                        },
                        "description": "Aspectos específicos a analizar",
                        "default": ["aromas", "sabores", "maridajes"]
                    }
                },
                "required": ["nombre_vino"]
            }
        ),
        types.Tool(
            name="comparar_vinos",
            description="Compara las características entre dos o más vinos",
            inputSchema={
                "type": "object",
                "properties": {
                    "vinos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de nombres de vinos a comparar (2-4 vinos)",
                        "minItems": 2,
                        "maxItems": 4
                    },
                    "criterios": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["precio", "calidad", "region", "tipo", "añada", "maridajes", "puntuacion"]
                        },
                        "description": "Criterios de comparación",
                        "default": ["precio", "calidad", "puntuacion"]
                    }
                },
                "required": ["vinos"]
            }
        ),
        
        # === HERRAMIENTAS DE RECOMENDACIÓN ===
        types.Tool(
            name="recomendar_por_presupuesto",
            description="Recomienda vinos dentro de un rango de presupuesto específico",
            inputSchema={
                "type": "object",
                "properties": {
                    "presupuesto_min": {
                        "type": "number",
                        "description": "Presupuesto mínimo en euros",
                        "minimum": 0
                    },
                    "presupuesto_max": {
                        "type": "number",
                        "description": "Presupuesto máximo en euros"
                    },
                    "tipo_vino": {
                        "type": "string",
                        "description": "Tipo de vino preferido",
                        "enum": ["tinto", "blanco", "rosado", "espumoso", "cualquiera"],
                        "default": "cualquiera"
                    },
                    "num_recomendaciones": {
                        "type": "integer",
                        "description": "Número de recomendaciones",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3
                    }
                },
                "required": ["presupuesto_max"]
            }
        ),
        types.Tool(
            name="recomendar_por_region",
            description="Recomienda vinos de una región vitivinícola específica",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "Región vitivinícola (ej: 'Rioja', 'Ribera del Duero', 'Rías Baixas')"
                    },
                    "estilo": {
                        "type": "string",
                        "description": "Estilo de vino buscado",
                        "enum": ["joven", "crianza", "reserva", "gran_reserva", "cualquiera"],
                        "default": "cualquiera"
                    },
                    "max_resultados": {
                        "type": "integer",
                        "description": "Máximo número de recomendaciones",
                        "minimum": 1,
                        "maximum": 15,
                        "default": 5
                    }
                },
                "required": ["region"]
            }
        ),
        
        # === HERRAMIENTAS DE EDUCACIÓN VINÍCOLA ===
        types.Tool(
            name="explicar_concepto",
            description="Explica conceptos técnicos de sumillería y viticultura en español",
            inputSchema={
                "type": "object",
                "properties": {
                    "concepto": {
                        "type": "string",
                        "description": "Concepto a explicar (ej: 'taninos', 'maloláctico', 'terroir', 'decantación')"
                    },
                    "nivel_detalle": {
                        "type": "string",
                        "description": "Nivel de explicación",
                        "enum": ["basico", "intermedio", "avanzado"],
                        "default": "intermedio"
                    }
                },
                "required": ["concepto"]
            }
        ),
        types.Tool(
            name="guia_cata",
            description="Proporciona guía paso a paso para catar vinos profesionalmente",
            inputSchema={
                        "type": "object",
                "properties": {
                    "tipo_vino": {
                        "type": "string",
                        "description": "Tipo de vino a catar",
                        "enum": ["tinto", "blanco", "rosado", "espumoso"],
                        "default": "tinto"
                    },
                    "experiencia": {
                        "type": "string",
                        "description": "Nivel de experiencia del catador",
                        "enum": ["principiante", "intermedio", "avanzado"],
                        "default": "principiante"
                    }
                },
                "required": ["tipo_vino"]
            }
        ),
        
        # === HERRAMIENTAS DE GESTIÓN DE BODEGA ===
        types.Tool(
            name="calcular_inventario",
            description="Calcula estadísticas del inventario de vinos disponible",
            inputSchema={
                "type": "object",
                "properties": {
                    "filtros": {
                        "type": "object",
                        "properties": {
                            "tipo": {"type": "string"},
                            "region": {"type": "string"},
                            "precio_min": {"type": "number"},
                            "precio_max": {"type": "number"}
                        },
                        "description": "Filtros opcionales para el cálculo"
                    },
                    "incluir_estadisticas": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["stock_total", "valor_total", "precio_promedio", "por_region", "por_tipo", "por_añada"]
                        },
                        "description": "Estadísticas a incluir",
                        "default": ["stock_total", "valor_total", "precio_promedio"]
                    }
                }
            }
        ),
        types.Tool(
            name="temperaturas_servicio",
            description="Proporciona temperaturas óptimas de servicio para diferentes tipos de vino",
            inputSchema={
                "type": "object",
                "properties": {
                    "tipo_vino": {
                        "type": "string",
                        "description": "Tipo de vino o nombre específico",
                        "enum": ["tinto_joven", "tinto_crianza", "tinto_reserva", "blanco_joven", "blanco_crianza", "rosado", "espumoso", "dulce", "fortificado"]
                    },
                    "contexto": {
                        "type": "string",
                        "description": "Contexto de consumo",
                        "enum": ["aperitivo", "comida", "postre", "degustacion"],
                        "default": "comida"
                    }
                },
                "required": ["tipo_vino"]
            }
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Ejecutar herramientas"""
    try:
        # === HERRAMIENTAS RAG BÁSICAS ===
        if name == "buscar_vinos":
            consulta = arguments.get("consulta", "")
            max_resultados = arguments.get("max_resultados", 5)
            
            response = await rag_engine.agentic_rag_query(consulta, max_results=max_resultados)
            
            # Filtrar solo vinos
            vinos = [source for source in response.sources if source.get('metadata', {}).get('type') == 'vino']
            
            result = f"🍷 **Búsqueda de vinos**: '{consulta}'\n\n"
            result += f"**Encontrados**: {len(vinos)} vinos\n\n"
            
            for i, vino in enumerate(vinos[:max_resultados], 1):
                metadata = vino.get('metadata', {})
                result += f"**{i}. {metadata.get('name', 'Sin nombre')}**\n"
                result += f"   • Tipo: {metadata.get('wine_type', 'N/A')}\n"
                result += f"   • Región: {metadata.get('region', 'N/A')}\n"
                result += f"   • Precio: {metadata.get('price', 'N/A')}€\n"
                result += f"   • Puntuación: {metadata.get('rating', 'N/A')}/100\n"
                result += f"   • Maridaje: {metadata.get('pairing', 'N/A')}\n\n"
            
            return [types.TextContent(type="text", text=result)]
            
        elif name == "agregar_documento":
            contenido = arguments.get("contenido", "")
            metadatos = arguments.get("metadatos", {})
            id_documento = arguments.get("id_documento")
            
            doc_id = await rag_engine.add_document(contenido, metadatos, id_documento)
            
            result = f"✅ **Documento agregado exitosamente**\n\n"
            result += f"**ID del documento**: {doc_id}\n"
            result += f"**Contenido**: {contenido[:100]}{'...' if len(contenido) > 100 else ''}\n"
            result += f"**Metadatos**: {metadatos}\n"
            
            return [types.TextContent(type="text", text=result)]
        
        # === HERRAMIENTAS DE MARIDAJE ===
        elif name == "sugerir_maridaje":
            plato = arguments.get("plato", "")
            ocasion = arguments.get("ocasion", "casual")
            presupuesto_max = arguments.get("presupuesto_max")
            
            # Expandir la consulta para maridaje
            consulta_maridaje = f"vino maridaje {plato} {ocasion}"
            response = await rag_engine.agentic_rag_query(consulta_maridaje, max_results=5)
            
            # Filtrar vinos y aplicar filtro de presupuesto si existe
            vinos_sugeridos = []
            for source in response.sources:
                if source.get('metadata', {}).get('type') == 'vino':
                    precio = source.get('metadata', {}).get('price', 0)
                    if presupuesto_max is None or (precio and precio <= presupuesto_max):
                        vinos_sugeridos.append(source)
            
            result = f"🍽️ **Sugerencias de maridaje para**: {plato}\n"
            result += f"**Ocasión**: {ocasion.title()}\n"
            if presupuesto_max:
                result += f"**Presupuesto máximo**: {presupuesto_max}€\n"
            result += f"\n**Vinos recomendados** ({len(vinos_sugeridos)} encontrados):\n\n"
            
            for i, vino in enumerate(vinos_sugeridos[:3], 1):
                metadata = vino.get('metadata', {})
                result += f"**{i}. {metadata.get('name', 'Sin nombre')}**\n"
                result += f"   • Tipo: {metadata.get('wine_type', 'N/A')}\n"
                result += f"   • Región: {metadata.get('region', 'N/A')}\n"
                result += f"   • Precio: {metadata.get('price', 'N/A')}€\n"
                result += f"   • ¿Por qué funciona?: {metadata.get('pairing', 'Maridaje versátil')}\n\n"
            
            if not vinos_sugeridos:
                result += "No se encontraron vinos específicos, pero puedes buscar vinos de estas características:\n"
                result += "• Para carnes: vinos tintos con cuerpo\n"
                result += "• Para pescados: vinos blancos frescos\n"
                result += "• Para postres: vinos dulces o espumosos\n"
            
            return [types.TextContent(type="text", text=result)]

        elif name == "explicar_concepto":
            concepto = arguments.get("concepto", "")
            nivel_detalle = arguments.get("nivel_detalle", "intermedio")
            
            # Buscar información del concepto en la base de conocimientos
            response = await rag_engine.agentic_rag_query(f"concepto {concepto} sumilleria viticultura", max_results=3)
            
            result = f"📚 **Concepto**: {concepto.title()}\n"
            result += f"**Nivel**: {nivel_detalle.title()}\n\n"
            
            # Si hay información en la base de conocimientos
            if response.sources and any('teoria' in s.get('metadata', {}).get('source', '') for s in response.sources):
                teoria_source = next((s for s in response.sources if 'teoria' in s.get('metadata', {}).get('source', '')), None)
                if teoria_source:
                    result += f"**Explicación desde la base de conocimientos:**\n"
                    result += f"{teoria_source.get('content', '')[:500]}...\n\n"
            
            # Explicaciones básicas predefinidas
            explicaciones = {
                "taninos": {
                    "basico": "Compuestos que dan sensación de sequedad y astringencia en boca. Vienen de las pieles de la uva.",
                    "intermedio": "Polifenoles que aportan estructura, color y capacidad de guarda a los vinos tintos. Se extraen de pieles, pepitas y madera.",
                    "avanzado": "Compuestos fenólicos que incluyen taninos condensados (proantocianidinas) y taninos hidrolizables, fundamentales en la estructura tánica y evolución organoléptica del vino."
                },
                "terroir": {
                    "basico": "El conjunto de factores ambientales que influyen en el carácter del vino: suelo, clima y tradición.",
                    "intermedio": "Concepto francés que engloba suelo, microclima, topografía y factor humano, creando la personalidad única de cada viñedo.",
                    "avanzado": "Sistema complejo de interacciones entre geología, pedología, climatología, hidrología y prácticas vitivinícolas que confieren tipicidad."
                },
                "decantacion": {
                    "basico": "Separar el vino de los sedimentos y oxigenarlo antes de servir.",
                    "intermedio": "Proceso de trasiego que permite separar sedimentos y favorecer la oxigenación para desarrollar aromas y suavizar taninos.",
                    "avanzado": "Técnica de oxigenación controlada que acelera procesos evolutivos, permitiendo la volatilización de compuestos reductivos y la polimerización tánica."
                }
            }
            
            if concepto.lower() in explicaciones:
                result += f"**Explicación ({nivel_detalle}):**\n"
                result += f"{explicaciones[concepto.lower()][nivel_detalle]}\n\n"
            
            result += f"**💡 Consejo práctico:**\n"
            if concepto.lower() == "taninos":
                result += "Para apreciar los taninos, prueba vinos jóvenes vs. vinos con crianza del mismo viñedo.\n"
            elif concepto.lower() == "terroir":
                result += "Compara vinos de la misma varietal de diferentes regiones para entender el terroir.\n"
            elif concepto.lower() == "decantacion":
                result += "Decanta vinos tintos jóvenes 30-60 min antes, vinos maduros solo para separar sedimentos.\n"
            else:
                result += "Profundiza practicando cata y consultando literatura especializada.\n"
            
            return [types.TextContent(type="text", text=result)]

        elif name == "temperaturas_servicio":
            tipo_vino = arguments.get("tipo_vino", "")
            contexto = arguments.get("contexto", "comida")
            
            temperaturas = {
                "tinto_joven": "14-16°C",
                "tinto_crianza": "16-18°C", 
                "tinto_reserva": "17-19°C",
                "blanco_joven": "8-10°C",
                "blanco_crianza": "10-12°C",
                "rosado": "8-10°C",
                "espumoso": "6-8°C",
                "dulce": "6-8°C",
                "fortificado": "12-16°C"
            }
            
            result = f"🌡️ **Temperaturas de Servicio**\n\n"
            
            if tipo_vino in temperaturas:
                result += f"**Vino**: {tipo_vino.replace('_', ' ').title()}\n"
                result += f"**Temperatura óptima**: {temperaturas[tipo_vino]}\n"
                result += f"**Contexto**: {contexto.title()}\n\n"
                
                # Consejos específicos
                result += f"**💡 Consejos prácticos:**\n"
                
                if "tinto" in tipo_vino:
                    result += "• Sacar de la bodega 1-2 horas antes\n"
                    result += "• En verano, puede necesitar refrigeración ligera\n"
                    result += "• Evitar calentamiento excesivo en la mano\n"
                elif "blanco" in tipo_vino or tipo_vino in ["rosado", "espumoso"]:
                    result += "• Refrigerar 2-3 horas antes del servicio\n"
                    result += "• Usar cubitera con agua y hielo durante la comida\n"
                    result += "• No sobre-enfriar (pierde aromas)\n"
                
                result += f"\n**⚠️ Importante:**\n"
                result += "• Termómetro de vino para precisión\n"
                result += "• La temperatura afecta percepción de aromas y sabores\n"
                result += "• Ajustar según estación del año\n"
                
            else:
                result += f"**Todas las temperaturas de servicio:**\n\n"
                for tipo, temp in temperaturas.items():
                    result += f"• **{tipo.replace('_', ' ').title()}**: {temp}\n"
                
                result += f"\n**Regla general:**\n"
                result += "• Más ligero el vino → más frío\n"
                result += "• Más estructura/crianza → más cálido\n"
                result += "• Espumosos siempre muy fríos\n"
            
            return [types.TextContent(type="text", text=result)]
        
        # Herramienta no encontrada
        else:
            return [types.TextContent(type="text", text=f"❌ Herramienta '{name}' no implementada")]
            
    except Exception as e:
        logger.error(f"Error ejecutando herramienta {name}: {e}")
        return [types.TextContent(type="text", text=f"❌ Error: {str(e)}")]

@mcp_server.list_resources()
async def list_resources() -> List[types.Resource]:
    """Listar recursos disponibles"""
    return [
        types.Resource(
            uri="knowledge://stats",
            name="Knowledge Base Statistics",
            description="Estadísticas de la base de conocimiento",
            mimeType="application/json"
        ),
        types.Resource(
            uri="knowledge://documents",
            name="Documents Collection",
            description="Colección completa de documentos",
            mimeType="application/json"
        )
    ]

@mcp_server.read_resource()
async def read_resource(uri: str) -> str:
    """Leer recurso solicitado"""
    if uri == "knowledge://stats":
        if rag_engine.collection:
            stats = rag_engine.collection.get()
            response = {
                "total_documents": len(stats['ids']),
                "collection_name": "rag_documents",
                "vector_db_type": VECTOR_DB_TYPE,
                "embedding_model": "all-MiniLM-L6-v2"
            }
            return json.dumps(response, indent=2)
        else:
            return json.dumps({"error": "Colección no inicializada"})
            
    elif uri == "knowledge://documents":
        if rag_engine.collection:
            docs = rag_engine.collection.get()
            response = {
                "documents": [
                    {
                        "id": doc_id,
                        "content": content[:200] + "..." if len(content) > 200 else content,
                        "metadata": metadata
                    }
                    for doc_id, content, metadata in zip(
                        docs['ids'],
                        docs['documents'],
                        docs['metadatas']
                    )
                ]
            }
            return json.dumps(response, indent=2, ensure_ascii=False)
        else:
            return json.dumps({"error": "Colección no inicializada"})
    else:
        raise ValueError(f"Recurso desconocido: {uri}")

# FastAPI para HTTP (opcional)
app = FastAPI(title="Agentic RAG MCP Server", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Inicializar al arrancar"""
    await rag_engine.initialize()
    
    # Cargar documentos de ejemplo si existen
    knowledge_dir = Path("/app/knowledge_base")
    if knowledge_dir.exists():
        # Cargar archivos de texto
        for file_path in knowledge_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    metadata = {"source": file_path.name, "type": "text"}
                    await rag_engine.add_document(content, metadata, file_path.stem)
                logger.info(f"Documento de texto cargado: {file_path.name}")
            except Exception as e:
                logger.error(f"Error cargando archivo de texto {file_path}: {e}")
        
        # Cargar archivos JSON (vinos)
        for file_path in knowledge_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Si es el archivo de vinos
                if file_path.name == "vinos.json" and isinstance(data, list):
                    logger.info(f"Cargando {len(data)} vinos desde {file_path.name}")
                    for i, vino in enumerate(data):
                        # Crear contenido estructurado para cada vino
                        content = f"""Vino: {vino.get('name', 'Sin nombre')}
Tipo: {vino.get('type', 'Sin tipo')}
Región: {vino.get('region', 'Sin región')}
Año: {vino.get('vintage', 'Sin año')}
Precio: {vino.get('price', 'Sin precio')}€
Stock: {vino.get('stock', 'Sin stock')} unidades
Maridaje: {vino.get('pairing', 'Sin maridaje')}
Descripción: {vino.get('description', 'Sin descripción')}
Puntuación: {vino.get('rating', 'Sin puntuación')}/100"""
                        
                        # Metadata rica para búsquedas
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
                else:
                    # Para otros archivos JSON, cargar como documento único
                    content = json.dumps(data, indent=2, ensure_ascii=False)
                    metadata = {"source": file_path.name, "type": "json"}
                    await rag_engine.add_document(content, metadata, file_path.stem)
                    logger.info(f"Documento JSON cargado: {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Error cargando archivo JSON {file_path}: {e}")

@app.get("/health")
async def health_check():
    """Verificación de salud"""
    return {"status": "healthy", "vector_db": VECTOR_DB_TYPE}

@app.post("/query")
async def query_rag_mcp(query_data: QueryRequest):
    start_total = time.time()
    logger.info(f"Received query: {query_data.query}")

    try:
        # Asegurar que rag_engine esté inicializado
        if rag_engine.collection is None:
            await rag_engine.initialize()

        # Paso 1: Obtener embeddings de la consulta
        start_embedding = time.time()
        query_embedding = rag_engine._embed_text(query_data.query)
        end_embedding = time.time()
        logger.info(f"Tiempo para obtener embedding de la consulta: {end_embedding - start_embedding:.4f}s")

        # Paso 2: Buscar documentos relevantes en ChromaDB
        start_chroma_search = time.time()
        results = rag_engine.collection.query(
            query_embeddings=[query_embedding],
            n_results=query_data.max_results,
            include=['documents', 'metadatas', 'distances']
        )
        end_chroma_search = time.time()
        logger.info(f"Tiempo para buscar en ChromaDB: {end_chroma_search - start_chroma_search:.4f}s")

        # Procesar resultados
        documents = results['documents'][0] if results and 'documents' in results else []
        metadatas = results['metadatas'][0] if results and 'metadatas' in results else []
        distances = results['distances'][0] if results and 'distances' in results else []

        # Construir fuentes
        sources = []
        context_str = ""
        for i, (doc_content, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            source_entry = {
                "content": doc_content,
                "metadata": metadata,
                "relevance_score": 1 - distance,
                "rank": i + 1
            }
            sources.append(source_entry)
            context_str += f"Fuente {i+1}:\n{doc_content}\n\n"

        # Paso 3: Generar respuesta usando OpenAI
        start_openai_call = time.time()
        
        if rag_engine.openai_client:
            messages = [
                {"role": "system", "content": "Eres un asistente sumiller experto. Usa el contexto proporcionado para responder a las preguntas sobre vinos. Si no puedes encontrar la respuesta en el contexto, indica que no tienes esa información. No alucines."},
                {"role": "user", "content": f"Contexto:\n{context_str}\n\nPregunta: {query_data.query}"}
            ]
            
            response = rag_engine.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )
            
            llm_answer = response.choices[0].message.content if response.choices else "No se pudo obtener una respuesta del modelo."
        else:
            llm_answer = f"Basado en {len(sources)} fuentes encontradas para: '{query_data.query}'"
        
        end_openai_call = time.time()
        logger.info(f"Tiempo para la llamada a OpenAI: {end_openai_call - start_openai_call:.4f}s")

        end_total = time.time()
        logger.info(f"Tiempo total de la solicitud /query: {end_total - start_total:.4f}s")

        return {
            "answer": llm_answer,
            "sources": sources,
            "context_used": {"query": query_data.query, "context": context_str}
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

async def main():
    """Función principal para ejecutar como servidor MCP"""
    import sys
    from mcp.server.stdio import stdio_server
    
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        # Modo HTTP con FastAPI
        import uvicorn
        # uvicorn.run(app, host="0.0.0.0", port=8000)
        PORT = int(os.getenv("PORT", "8000")) # Usar 8000 como fallback para desarrollo local
        uvicorn.run(app, host="0.0.0.0", port=PORT) 
    else:
        # Modo MCP stdio por defecto
        await rag_engine.initialize()
        
        # Cargar documentos desde directorio local si existe
        knowledge_dir = Path("knowledge_base")
        if knowledge_dir.exists():
            await startup_event()
        
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options()
            )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())