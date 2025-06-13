#!/usr/bin/env python3
"""
Servidor de Memoria MCP
Maneja la memoria persistente y contexto para el sistema RAG agéntico
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid

import redis.asyncio as redis
from fastapi import FastAPI
import uvicorn

# MCP SDK imports
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

class MemoryManager:
    """Gestor de memoria persistente para el sistema RAG agéntico"""
    
    def __init__(self):
        self.redis_client = None
    
    async def initialize(self):
        """Inicializar conexión a Redis"""
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Conexión a Redis establecida")
        except Exception as e:
            logger.error(f"Error conectando a Redis: {e}")
            raise
    
    async def store_conversation(self, session_id: str, user_query: str, response: str, context: Dict[str, Any] = None):
        """Almacenar conversación en memoria"""
        try:
            conversation_entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "response": response,
                "context": context or {},
                "session_id": session_id
            }
            
            # Almacenar en lista de conversación de la sesión
            await self.redis_client.lpush(
                f"conversation:{session_id}",
                json.dumps(conversation_entry)
            )
            
            # Mantener solo las últimas 50 entradas por sesión
            await self.redis_client.ltrim(f"conversation:{session_id}", 0, 49)
            
            # Índice global por timestamp
            await self.redis_client.zadd(
                "conversations:timeline",
                {json.dumps(conversation_entry): datetime.now().timestamp()}
            )
            
            return conversation_entry["id"]
            
        except Exception as e:
            logger.error(f"Error almacenando conversación: {e}")
            raise
    
    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener historial de conversación"""
        try:
            entries = await self.redis_client.lrange(f"conversation:{session_id}", 0, limit-1)
            return [json.loads(entry) for entry in entries]
        except Exception as e:
            logger.error(f"Error obteniendo historial: {e}")
            return []
    
    async def store_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Almacenar preferencias de usuario"""
        try:
            await self.redis_client.hset(
                f"user:{user_id}:preferences",
                mapping={k: json.dumps(v) for k, v in preferences.items()}
            )
            return True
        except Exception as e:
            logger.error(f"Error almacenando preferencias: {e}")
            return False
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Obtener preferencias de usuario"""
        try:
            prefs = await self.redis_client.hgetall(f"user:{user_id}:preferences")
            return {k: json.loads(v) for k, v in prefs.items()}
        except Exception as e:
            logger.error(f"Error obteniendo preferencias: {e}")
            return {}
    
    async def store_domain_knowledge(self, domain: str, key: str, knowledge: Dict[str, Any]):
        """Almacenar conocimiento específico del dominio"""
        try:
            knowledge_entry = {
                "key": key,
                "domain": domain,
                "knowledge": knowledge,
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4())
            }
            
            await self.redis_client.hset(
                f"domain:{domain}:knowledge",
                key,
                json.dumps(knowledge_entry)
            )
            
            return knowledge_entry["id"]
            
        except Exception as e:
            logger.error(f"Error almacenando conocimiento de dominio: {e}")
            raise
    
    async def get_domain_knowledge(self, domain: str, key: str = None) -> Dict[str, Any]:
        """Obtener conocimiento del dominio"""
        try:
            if key:
                knowledge = await self.redis_client.hget(f"domain:{domain}:knowledge", key)
                return json.loads(knowledge) if knowledge else {}
            else:
                all_knowledge = await self.redis_client.hgetall(f"domain:{domain}:knowledge")
                return {k: json.loads(v) for k, v in all_knowledge.items()}
        except Exception as e:
            logger.error(f"Error obteniendo conocimiento de dominio: {e}")
            return {}
    
    async def store_query_pattern(self, pattern: str, responses: List[str], metadata: Dict[str, Any] = None):
        """Almacenar patrones de consulta frecuentes"""
        try:
            pattern_entry = {
                "pattern": pattern,
                "responses": responses,
                "metadata": metadata or {},
                "count": 1,
                "last_used": datetime.now().isoformat(),
                "created": datetime.now().isoformat()
            }
            
            # Verificar si el patrón ya existe
            existing = await self.redis_client.get(f"pattern:{pattern}")
            if existing:
                existing_data = json.loads(existing)
                pattern_entry["count"] = existing_data.get("count", 0) + 1
                pattern_entry["created"] = existing_data.get("created", pattern_entry["created"])
            
            await self.redis_client.set(
                f"pattern:{pattern}",
                json.dumps(pattern_entry)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error almacenando patrón: {e}")
            return False
    
    async def get_similar_patterns(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Buscar patrones similares (búsqueda simple por ahora)"""
        try:
            # Obtener todos los patrones
            pattern_keys = await self.redis_client.keys("pattern:*")
            patterns = []
            
            for key in pattern_keys:
                pattern_data = await self.redis_client.get(key)
                if pattern_data:
                    data = json.loads(pattern_data)
                    # Búsqueda simple por palabras clave
                    pattern = data["pattern"].lower()
                    query_lower = query.lower()
                    
                    # Calcular similaridad básica
                    common_words = set(pattern.split()) & set(query_lower.split())
                    similarity = len(common_words) / max(len(pattern.split()), len(query_lower.split()))
                    
                    if similarity > 0.2:  # Umbral mínimo
                        data["similarity"] = similarity
                        patterns.append(data)
            
            # Ordenar por similaridad y frecuencia
            patterns.sort(key=lambda x: (x["similarity"], x["count"]), reverse=True)
            return patterns[:limit]
            
        except Exception as e:
            logger.error(f"Error buscando patrones similares: {e}")
            return []
    
    async def cleanup_old_data(self, days: int = 30):
        """Limpiar datos antiguos"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            # Limpiar conversaciones antiguas del timeline
            await self.redis_client.zremrangebyscore(
                "conversations:timeline",
                0,
                cutoff_timestamp
            )
            
            logger.info(f"Datos anteriores a {days} días limpiados")
            return True
            
        except Exception as e:
            logger.error(f"Error limpiando datos antiguos: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de memoria"""
        try:
            info = await self.redis_client.info()
            
            # Contar elementos
            conversation_sessions = len(await self.redis_client.keys("conversation:*"))
            user_prefs = len(await self.redis_client.keys("user:*:preferences"))
            domain_knowledge = len(await self.redis_client.keys("domain:*:knowledge"))
            patterns = len(await self.redis_client.keys("pattern:*"))
            
            return {
                "redis_info": {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                },
                "data_counts": {
                    "conversation_sessions": conversation_sessions,
                    "user_preferences": user_prefs,
                    "domain_knowledge_domains": domain_knowledge,
                    "query_patterns": patterns
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}

# Instancia global del gestor de memoria
memory_manager = MemoryManager()

# Servidor MCP
mcp_server = Server("memory-server")

@mcp_server.list_tools()
async def list_tools() -> List[types.Tool]:
    """Listar herramientas de memoria disponibles"""
    return [
        types.Tool(
            name="store_conversation",
            description="Almacenar una conversación en la memoria del sistema",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "ID de la sesión"},
                    "user_query": {"type": "string", "description": "Consulta del usuario"},
                    "response": {"type": "string", "description": "Respuesta del sistema"},
                    "context": {"type": "object", "description": "Contexto adicional"}
                },
                "required": ["session_id", "user_query", "response"]
            }
        ),
        types.Tool(
            name="get_conversation_history",
            description="Obtener historial de conversación de una sesión",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "ID de la sesión"},
                    "limit": {"type": "integer", "description": "Número máximo de entradas", "default": 10}
                },
                "required": ["session_id"]
            }
        ),
        types.Tool(
            name="store_user_preferences",
            description="Almacenar preferencias de usuario",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "ID del usuario"},
                    "preferences": {"type": "object", "description": "Preferencias del usuario"}
                },
                "required": ["user_id", "preferences"]
            }
        ),
        types.Tool(
            name="get_user_preferences",
            description="Obtener preferencias de usuario",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "ID del usuario"}
                },
                "required": ["user_id"]
            }
        ),
        types.Tool(
            name="store_domain_knowledge",
            description="Almacenar conocimiento específico del dominio",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Dominio del conocimiento"},
                    "key": {"type": "string", "description": "Clave del conocimiento"},
                    "knowledge": {"type": "object", "description": "Datos del conocimiento"}
                },
                "required": ["domain", "key", "knowledge"]
            }
        ),
        types.Tool(
            name="get_domain_knowledge",
            description="Obtener conocimiento del dominio",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Dominio del conocimiento"},
                    "key": {"type": "string", "description": "Clave específica (opcional)"}
                },
                "required": ["domain"]
            }
        ),
        types.Tool(
            name="find_similar_patterns",
            description="Buscar patrones de consulta similares",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta para buscar patrones similares"},
                    "limit": {"type": "integer", "description": "Número máximo de patrones", "default": 5}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_memory_stats",
            description="Obtener estadísticas del sistema de memoria",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Ejecutar herramienta de memoria"""
    try:
        if name == "store_conversation":
            session_id = arguments["session_id"]
            user_query = arguments["user_query"]
            response = arguments["response"]
            context = arguments.get("context", {})
            
            conv_id = await memory_manager.store_conversation(session_id, user_query, response, context)
            
            return [types.TextContent(
                type="text",
                text=f"Conversación almacenada con ID: {conv_id}"
            )]
            
        elif name == "get_conversation_history":
            session_id = arguments["session_id"]
            limit = arguments.get("limit", 10)
            
            history = await memory_manager.get_conversation_history(session_id, limit)
            
            return [types.TextContent(
                type="text",
                text=json.dumps(history, indent=2, ensure_ascii=False)
            )]
            
        elif name == "store_user_preferences":
            user_id = arguments["user_id"]
            preferences = arguments["preferences"]
            
            success = await memory_manager.store_user_preferences(user_id, preferences)
            
            return [types.TextContent(
                type="text",
                text=f"Preferencias {'almacenadas' if success else 'no pudieron ser almacenadas'}"
            )]
            
        elif name == "get_user_preferences":
            user_id = arguments["user_id"]
            
            preferences = await memory_manager.get_user_preferences(user_id)
            
            return [types.TextContent(
                type="text",
                text=json.dumps(preferences, indent=2, ensure_ascii=False)
            )]
            
        elif name == "store_domain_knowledge":
            domain = arguments["domain"]
            key = arguments["key"]
            knowledge = arguments["knowledge"]
            
            knowledge_id = await memory_manager.store_domain_knowledge(domain, key, knowledge)
            
            return [types.TextContent(
                type="text",
                text=f"Conocimiento de dominio almacenado con ID: {knowledge_id}"
            )]
            
        elif name == "get_domain_knowledge":
            domain = arguments["domain"]
            key = arguments.get("key")
            
            knowledge = await memory_manager.get_domain_knowledge(domain, key)
            
            return [types.TextContent(
                type="text",
                text=json.dumps(knowledge, indent=2, ensure_ascii=False)
            )]
            
        elif name == "find_similar_patterns":
            query = arguments["query"]
            limit = arguments.get("limit", 5)
            
            patterns = await memory_manager.get_similar_patterns(query, limit)
            
            return [types.TextContent(
                type="text",
                text=json.dumps(patterns, indent=2, ensure_ascii=False)
            )]
            
        elif name == "get_memory_stats":
            stats = await memory_manager.get_memory_stats()
            
            return [types.TextContent(
                type="text",
                text=json.dumps(stats, indent=2, ensure_ascii=False)
            )]
            
        else:
            raise ValueError(f"Herramienta desconocida: {name}")
            
    except Exception as e:
        logger.error(f"Error ejecutando herramienta {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]

@mcp_server.list_resources()
async def list_resources() -> List[types.Resource]:
    """Listar recursos de memoria disponibles"""
    return [
        types.Resource(
            uri="memory://stats",
            name="Memory Statistics",
            description="Estadísticas del sistema de memoria",
            mimeType="application/json"
        ),
        types.Resource(
            uri="memory://sessions",
            name="Active Sessions",
            description="Sesiones activas en memoria",
            mimeType="application/json"
        )
    ]

@mcp_server.read_resource()
async def read_resource(uri: str) -> str:
    """Leer recurso de memoria"""
    if uri == "memory://stats":
        stats = await memory_manager.get_memory_stats()
        return json.dumps(stats, indent=2, ensure_ascii=False)
        
    elif uri == "memory://sessions":
        try:
            session_keys = await memory_manager.redis_client.keys("conversation:*")
            sessions = []
            
            for key in session_keys:
                session_id = key.split(":")[-1]
                recent_entries = await memory_manager.redis_client.lrange(key, 0, 2)
                
                sessions.append({
                    "session_id": session_id,
                    "recent_entries_count": len(recent_entries),
                    "last_activity": json.loads(recent_entries[0])["timestamp"] if recent_entries else None
                })
            
            return json.dumps({"active_sessions": sessions}, indent=2, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    else:
        raise ValueError(f"Recurso desconocido: {uri}")

# FastAPI para HTTP (opcional)
app = FastAPI(title="Memory MCP Server", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Inicializar al arrancar"""
    await memory_manager.initialize()

@app.get("/health")
async def health_check():
    """Verificación de salud"""
    try:
        await memory_manager.redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except:
        return {"status": "unhealthy", "redis": "disconnected"}

# --- Endpoints REST para el sumiller-bot ---

@app.get("/memory/{user_id}")
async def get_user_memory(user_id: str):
    """Obtener memoria del usuario"""
    try:
        # Obtener historial de conversación
        conversation_history = await memory_manager.get_conversation_history(user_id, limit=10)
        
        # Obtener preferencias del usuario
        user_preferences = await memory_manager.get_user_preferences(user_id)
        
        return {
            "user_id": user_id,
            "conversation_history": conversation_history,
            "preferences": user_preferences,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error obteniendo memoria del usuario {user_id}: {e}")
        return {
            "user_id": user_id,
            "conversation_history": [],
            "preferences": {},
            "status": "error",
            "error": str(e)
        }

@app.post("/memory/save")
async def save_user_memory(data: dict):
    """Guardar interacción del usuario"""
    try:
        user_id = data.get("user_id", "default_user")
        query = data.get("query", "")
        response = data.get("response", "")
        context = data.get("context", {})
        
        # Guardar conversación
        conversation_id = await memory_manager.store_conversation(
            session_id=user_id,
            user_query=query,
            response=response,
            context=context
        )
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error guardando memoria: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/stats")
async def get_memory_stats():
    """Obtener estadísticas de memoria"""
    try:
        stats = await memory_manager.get_memory_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def main():
    """Función principal"""
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        # Modo HTTP
        # uvicorn.run(app, host="0.0.0.0", port=8000)
        PORT = int(os.getenv("PORT", "8000")) # Usar 8000 como fallback para desarrollo local
        uvicorn.run(app, host="0.0.0.0", port=PORT) 
    else:
        # Modo MCP stdio
        await memory_manager.initialize()
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options()
            )

if __name__ == "__main__":
    asyncio.run(main())