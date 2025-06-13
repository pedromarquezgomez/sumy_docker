#!/usr/bin/env python3
"""
Cliente MCP para conectar servidores RAG y Memory con Claude Desktop
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    """Cliente MCP que conecta con los servidores RAG y Memory"""
    
    def __init__(self):
        self.rag_session = None
        self.memory_session = None
        self.current_session_id = "claude_session_001"
    
    async def connect_to_servers(self):
        """Conectar a los servidores MCP"""
        try:
            # Configuración para servidor RAG
            rag_params = StdioServerParameters(
                command="python",
                args=["/app/rag_mcp_server.py"],
                env={
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                    "VECTOR_DB_TYPE": "chroma",
                    "CHROMA_HOST": "chromadb",
                    "CHROMA_PORT": "8001"
                }
            )
            
            # Configuración para servidor Memory
            memory_params = StdioServerParameters(
                command="python",
                args=["/app/memory_mcp_server.py"],
                env={
                    "REDIS_URL": "redis://redis:6379"
                }
            )
            
            # Conectar a servidor RAG
            logger.info("Conectando a servidor RAG...")
            rag_stdio = stdio_client(rag_params)
            rag_read, rag_write = await rag_stdio.__aenter__()
            self.rag_session = ClientSession(rag_read, rag_write)
            await self.rag_session.initialize()
            
            # Conectar a servidor Memory
            logger.info("Conectando a servidor Memory...")
            memory_stdio = stdio_client(memory_params)
            memory_read, memory_write = await memory_stdio.__aenter__()
            self.memory_session = ClientSession(memory_read, memory_write)
            await self.memory_session.initialize()
            
            logger.info("✅ Conexiones establecidas exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error conectando a servidores: {e}")
            raise
    
    async def search_knowledge(self, query: str, context: Dict[str, Any] = None, max_results: int = 5) -> Dict[str, Any]:
        """Buscar en la base de conocimiento usando RAG agéntico"""
        try:
            if not self.rag_session:
                raise Exception("Servidor RAG no conectado")
            
            result = await self.rag_session.call_tool(
                "search_knowledge",
                {
                    "query": query,
                    "context": context or {},
                    "max_results": max_results
                }
            )
            
            # Parsear respuesta
            if result.content and result.content[0].text:
                response_data = json.loads(result.content[0].text)
                
                # Almacenar en memoria
                await self.store_conversation(query, response_data.get("answer", ""), context)
                
                return response_data
            else:
                return {"error": "No response from RAG server"}
                
        except Exception as e:
            logger.error(f"Error en búsqueda de conocimiento: {e}")
            return {"error": str(e)}
    
    async def add_document(self, content: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> str:
        """Agregar documento a la base de conocimiento"""
        try:
            if not self.rag_session:
                raise Exception("Servidor RAG no conectado")
            
            result = await self.rag_session.call_tool(
                "add_document",
                {
                    "content": content,
                    "metadata": metadata or {},
                    "doc_id": doc_id
                }
            )
            
            if result.content and result.content[0].text:
                return result.content[0].text
            else:
                return "Error adding document"
                
        except Exception as e:
            logger.error(f"Error agregando documento: {e}")
            return f"Error: {str(e)}"
    
    async def store_conversation(self, user_query: str, response: str, context: Dict[str, Any] = None):
        """Almacenar conversación en memoria"""
        try:
            if not self.memory_session:
                logger.warning("Servidor Memory no conectado, saltando almacenamiento")
                return
            
            await self.memory_session.call_tool(
                "store_conversation",
                {
                    "session_id": self.current_session_id,
                    "user_query": user_query,
                    "response": response,
                    "context": context or {}
                }
            )
            
        except Exception as e:
            logger.error(f"Error almacenando conversación: {e}")
    
    async def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener historial de conversación"""
        try:
            if not self.memory_session:
                return []
            
            result = await self.memory_session.call_tool(
                "get_conversation_history",
                {
                    "session_id": self.current_session_id,
                    "limit": limit
                }
            )
            
            if result.content and result.content[0].text:
                return json.loads(result.content[0].text)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error obteniendo historial: {e}")
            return []
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de conocimiento"""
        try:
            if not self.rag_session:
                return {"error": "RAG server not connected"}
            
            result = await self.rag_session.call_tool("get_collection_stats", {})
            
            if result.content and result.content[0].text:
                return json.loads(result.content[0].text)
            else:
                return {"error": "No stats available"}
                
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
    
    async def interactive_session(self):
        """Sesión interactiva para probar el cliente"""
        print("🤖 Cliente MCP Interactivo")
        print("Comandos disponibles:")
        print("  search <consulta>     - Buscar en la base de conocimiento")
        print("  add <contenido>       - Agregar documento")
        print("  history               - Ver historial de conversación")
        print("  stats                 - Ver estadísticas")
        print("  help                  - Mostrar ayuda")
        print("  quit                  - Salir")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 >> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 ¡Hasta luego!")
                    break
                
                elif user_input.lower() == 'help':
                    print("📚 Comandos disponibles:")
                    print("  search <consulta>     - Buscar información")
                    print("  add <contenido>       - Agregar documento")
                    print("  history               - Ver conversaciones")
                    print("  stats                 - Ver estadísticas")
                    print("  quit                  - Salir")
                
                elif user_input.lower() == 'history':
                    print("📜 Obteniendo historial...")
                    history = await self.get_conversation_history()
                    if history:
                        for i, entry in enumerate(history[:5]):  # Últimas 5
                            print(f"\n{i+1}. {entry.get('timestamp', 'N/A')}")
                            print(f"   👤: {entry.get('user_query', 'N/A')[:80]}...")
                            print(f"   🤖: {entry.get('response', 'N/A')[:80]}...")
                    else:
                        print("   No hay historial disponible")
                
                elif user_input.lower() == 'stats':
                    print("📊 Obteniendo estadísticas...")
                    stats = await self.get_knowledge_stats()
                    print(f"   📚 Total documentos: {stats.get('total_documents', 'N/A')}")
                    print(f"   🗃️ Colección: {stats.get('collection_name', 'N/A')}")
                    print(f"   💾 Base vectorial: {stats.get('vector_db_type', 'N/A')}")
                
                elif user_input.startswith('add '):
                    content = user_input[4:].strip()
                    if content:
                        print("📝 Agregando documento...")
                        result = await self.add_document(content, {"source": "interactive"})
                        print(f"   ✅ {result}")
                    else:
                        print("   ⚠️ Contenido requerido")
                
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        print("🔍 Buscando...")
                        result = await self.search_knowledge(query)
                        
                        if "error" in result:
                            print(f"   ❌ Error: {result['error']}")
                        else:
                            print(f"\n🎯 Respuesta:")
                            print(f"   {result.get('answer', 'No response')}")
                            
                            sources = result.get('sources', [])
                            if sources:
                                print(f"\n📚 Fuentes ({len(sources)}):")
                                for i, source in enumerate(sources[:3]):
                                    relevance = source.get('relevance_score', 0)
                                    content = source.get('content', '')[:100]
                                    print(f"   {i+1}. [{relevance:.2f}] {content}...")
                    else:
                        print("   ⚠️ Consulta requerida")
                
                else:
                    # Tratar como búsqueda por defecto
                    print("🔍 Buscando...")
                    result = await self.search_knowledge(user_input)
                    
                    if "error" in result:
                        print(f"   ❌ Error: {result['error']}")
                    else:
                        print(f"\n🎯 Respuesta:")
                        print(f"   {result.get('answer', 'No response')}")
                        
                        sources = result.get('sources', [])
                        if sources:
                            print(f"\n📚 Fuentes encontradas: {len(sources)}")
            
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

async def main():
    """Función principal"""
    client = MCPClient()
    
    try:
        await client.connect_to_servers()
        
        if len(sys.argv) > 1 and sys.argv[1] == "interactive":
            await client.interactive_session()
        else:
            # Modo de demostración
            print("🚀 Demo del Cliente MCP")
            
            # Agregar documento de prueba
            print("\n📝 Agregando documento de prueba...")
            doc_result = await client.add_document(
                "MCP (Model Context Protocol) es un protocolo estándar para conectar LLMs con herramientas externas.",
                {"topic": "MCP", "source": "demo"}
            )
            print(f"   {doc_result}")
            
            # Realizar búsqueda
            print("\n🔍 Realizando búsqueda de prueba...")
            search_result = await client.search_knowledge("¿Qué es MCP?")
            
            if "error" not in search_result:
                print(f"   🎯 Respuesta: {search_result.get('answer', 'No response')}")
                print(f"   📚 Fuentes encontradas: {len(search_result.get('sources', []))}")
            else:
                print(f"   ❌ Error: {search_result['error']}")
            
            # Mostrar estadísticas
            print("\n📊 Estadísticas:")
            stats = await client.get_knowledge_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            print("\n✅ Demo completada!")
            
    except Exception as e:
        logger.error(f"Error en cliente: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())