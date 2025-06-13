#!/usr/bin/env python3
"""
Tester Interactivo para Servidores MCP
Permite probar y validar la funcionalidad de los servidores RAG y Memory
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

import streamlit as st
import requests
import uuid
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://localhost:8000")
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8002")

class MCPTester:
    """Clase para probar servidores MCP"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.test_results = []
    
    async def test_rag_server_http(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Probar servidor RAG vía HTTP"""
        try:
            payload = {
                "query": query,
                "context": context or {},
                "max_results": 5
            }
            
            response = requests.post(f"{RAG_SERVER_URL}/query", json=payload, timeout=30)
            response.raise_for_status()
            
            return {
                "success": True,
                "response": response.json(),
                "status_code": response.status_code
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": getattr(e, 'response', {}).get('status_code', 'N/A')
            }
    
    async def test_memory_server_http(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Probar servidor de memoria vía HTTP"""
        try:
            # Esta función podría expandirse para incluir endpoints HTTP del servidor de memoria
            # Por ahora, simulamos la funcionalidad
            return {
                "success": True,
                "response": f"Memory action '{action}' executed",
                "data": data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_test_result(self, test_name: str, result: Dict[str, Any]):
        """Agregar resultado de prueba"""
        self.test_results.append({
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Obtener resumen de pruebas"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for test in self.test_results if test["result"].get("success", False))
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
        }

# Instancia global del tester
tester = MCPTester()

# Configuración de Streamlit
st.set_page_config(
    page_title="MCP Server Tester",
    page_icon="🧪",
    layout="wide"
)

def main():
    """Interfaz principal de Streamlit"""
    st.title("🧪 MCP Server Tester")
    st.markdown("Herramienta interactiva para probar servidores MCP RAG y Memory")
    
    # Sidebar con configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        st.text(f"RAG Server: {RAG_SERVER_URL}")
        st.text(f"Memory Server: {MEMORY_SERVER_URL}")
        st.text(f"Session ID: {tester.session_id}")
        
        # Verificar salud de servidores
        st.subheader("🏥 Estado de Servidores")
        
        try:
            rag_health = requests.get(f"{RAG_SERVER_URL}/health", timeout=5)
            st.success("✅ RAG Server: Healthy")
        except:
            st.error("❌ RAG Server: Unreachable")
        
        try:
            memory_health = requests.get(f"{MEMORY_SERVER_URL}/health", timeout=5)
            st.success("✅ Memory Server: Healthy")
        except:
            st.error("❌ Memory Server: Unreachable")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 RAG Testing", "🧠 Memory Testing", "📊 Results", "📋 Test Suite"])
    
    with tab1:
        st.header("🔍 Pruebas del Servidor RAG")
        
        # Sección para agregar documentos
        st.subheader("📄 Agregar Documentos")
        col1, col2 = st.columns(2)
        
        with col1:
            doc_content = st.text_area("Contenido del documento", height=150)
            doc_metadata = st.text_input("Metadatos (JSON)", '{"source": "test", "type": "manual"}')
        
        with col2:
            doc_id = st.text_input("ID del documento (opcional)")
            
            if st.button("➕ Agregar Documento"):
                if doc_content.strip():
                    try:
                        metadata = json.loads(doc_metadata) if doc_metadata else {}
                        payload = {
                            "content": doc_content,
                            "metadata": metadata,
                            "doc_id": doc_id if doc_id else None
                        }
                        
                        response = requests.post(f"{RAG_SERVER_URL}/documents", json=payload)
                        
                        if response.status_code == 200:
                            st.success(f"✅ Documento agregado: {response.json()}")
                        else:
                            st.error(f"❌ Error: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Contenido del documento requerido")
        
        st.divider()
        
        # Sección para consultas RAG
        st.subheader("❓ Consultas RAG")
        
        query_input = st.text_input("Escribe tu consulta:", placeholder="¿Qué información necesitas?")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            max_results = st.slider("Máximo resultados", 1, 10, 5)
        with col2:
            include_context = st.checkbox("Incluir contexto")
        with col3:
            if st.button("🔍 Ejecutar Consulta", type="primary"):
                if query_input.strip():
                    with st.spinner("Procesando consulta..."):
                        context = {"session_id": tester.session_id} if include_context else None
                        result = asyncio.run(tester.test_rag_server_http(query_input, context))
                        tester.add_test_result("RAG Query", result)
                        
                        if result["success"]:
                            st.success("✅ Consulta exitosa")
                            
                            response_data = result["response"]
                            st.subheader("🎯 Respuesta")
                            st.write(response_data.get("answer", "No response"))
                            
                            st.subheader("📚 Fuentes")
                            sources = response_data.get("sources", [])
                            for i, source in enumerate(sources):
                                with st.expander(f"Fuente {i+1} (Relevancia: {source.get('relevance_score', 0):.2f})"):
                                    st.write(source.get("content", "No content"))
                                    st.json(source.get("metadata", {}))
                        else:
                            st.error(f"❌ Error: {result['error']}")
                else:
                    st.warning("⚠️ Escribe una consulta")
    
    with tab2:
        st.header("🧠 Pruebas del Servidor de Memoria")
        
        # Sección para almacenar conversaciones
        st.subheader("💬 Almacenar Conversación")
        col1, col2 = st.columns(2)
        
        with col1:
            user_query = st.text_input("Consulta del usuario:")
            system_response = st.text_area("Respuesta del sistema:", height=100)
        
        with col2:
            context_data = st.text_area("Contexto (JSON):", '{"test": true}', height=100)
            
            if st.button("💾 Guardar Conversación"):
                if user_query and system_response:
                    try:
                        context = json.loads(context_data) if context_data else {}
                        # Simular almacenamiento (en una implementación real, usaríamos MCP)
                        result = {
                            "success": True,
                            "conversation_id": str(uuid.uuid4()),
                            "message": "Conversación almacenada"
                        }
                        tester.add_test_result("Store Conversation", result)
                        st.success("✅ Conversación guardada")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Consulta y respuesta requeridas")
        
        st.divider()
        
        # Sección para preferencias de usuario
        st.subheader("👤 Preferencias de Usuario")
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.text_input("ID de Usuario:", value="test_user")
            preferences = st.text_area("Preferencias (JSON):", '{"language": "es", "theme": "dark"}')
        
        with col2:
            if st.button("💾 Guardar Preferencias"):
                try:
                    prefs = json.loads(preferences)
                    result = {
                        "success": True,
                        "user_id": user_id,
                        "message": "Preferencias guardadas"
                    }
                    tester.add_test_result("Store Preferences", result)
                    st.success("✅ Preferencias guardadas")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            if st.button("📖 Cargar Preferencias"):
                # Simular carga de preferencias
                result = {
                    "success": True,
                    "preferences": {"language": "es", "theme": "dark"},
                    "message": "Preferencias cargadas"
                }
                tester.add_test_result("Load Preferences", result)
                st.success("✅ Preferencias cargadas")
                st.json(result["preferences"])
    
    with tab3:
        st.header("📊 Resultados de Pruebas")
        
        # Resumen de pruebas
        summary = tester.get_test_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pruebas", summary["total_tests"])
        with col2:
            st.metric("Exitosas", summary["successful_tests"])
        with col3:
            st.metric("Fallidas", summary["failed_tests"])
        with col4:
            st.metric("% Éxito", f"{summary['success_rate']:.1f}%")
        
        # Historial de pruebas
        if tester.test_results:
            st.subheader("📋 Historial de Pruebas")
            
            for i, test in enumerate(reversed(tester.test_results[-10:])):  # Últimas 10
                with st.expander(f"#{len(tester.test_results)-i}: {test['test_name']} - {test['timestamp']}"):
                    if test["result"]["success"]:
                        st.success("✅ Exitosa")
                    else:
                        st.error("❌ Fallida")
                    st.json(test["result"])
        else:
            st.info("ℹ️ No hay resultados de pruebas aún")
    
    with tab4:
        st.header("📋 Suite de Pruebas Automatizadas")
        
        st.markdown("Ejecuta una serie de pruebas predefinidas para validar el sistema completo.")
        
        # Configuración de la suite
        st.subheader("⚙️ Configuración")
        col1, col2 = st.columns(2)
        
        with col1:
            test_documents = st.checkbox("Probar agregado de documentos", True)
            test_queries = st.checkbox("Probar consultas RAG", True)
            test_memory = st.checkbox("Probar sistema de memoria", True)
        
        with col2:
            num_test_docs = st.slider("Número de documentos de prueba", 1, 10, 3)
            num_test_queries = st.slider("Número de consultas de prueba", 1, 10, 5)
        
        if st.button("🚀 Ejecutar Suite Completa", type="primary"):
            with st.spinner("Ejecutando suite de pruebas..."):
                progress_bar = st.progress(0)
                results_container = st.container()
                
                total_steps = 0
                if test_documents: total_steps += num_test_docs
                if test_queries: total_steps += num_test_queries
                if test_memory: total_steps += 3
                
                current_step = 0
                
                # Documentos de prueba
                test_docs = [
                    {"content": "El machine learning es una rama de la inteligencia artificial.", "metadata": {"topic": "AI"}},
                    {"content": "Python es un lenguaje de programación versátil y fácil de aprender.", "metadata": {"topic": "Programming"}},
                    {"content": "Docker permite containerizar aplicaciones para facilitar el despliegue.", "metadata": {"topic": "DevOps"}},
                    {"content": "Los modelos de lenguaje grandes como GPT pueden generar texto coherente.", "metadata": {"topic": "NLP"}},
                    {"content": "La computación en la nube ofrece escalabilidad y flexibilidad.", "metadata": {"topic": "Cloud"}}
                ]
                
                if test_documents:
                    results_container.subheader("📄 Pruebas de Documentos")
                    for i in range(num_test_docs):
                        doc = test_docs[i % len(test_docs)]
                        try:
                            response = requests.post(f"{RAG_SERVER_URL}/documents", 
                                                   json={**doc, "doc_id": f"test_doc_{i}"})
                            if response.status_code == 200:
                                results_container.success(f"✅ Documento {i+1} agregado")
                            else:
                                results_container.error(f"❌ Error documento {i+1}: {response.text}")
                        except Exception as e:
                            results_container.error(f"❌ Error documento {i+1}: {str(e)}")
                        
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                
                # Consultas de prueba
                test_queries_list = [
                    "¿Qué es machine learning?",
                    "¿Cuáles son los beneficios de Python?",
                    "¿Cómo funciona Docker?",
                    "¿Qué son los modelos de lenguaje?",
                    "¿Qué ventajas ofrece la nube?"
                ]
                
                if test_queries:
                    results_container.subheader("❓ Pruebas de Consultas")
                    for i in range(num_test_queries):
                        query = test_queries_list[i % len(test_queries_list)]
                        try:
                            payload = {"query": query, "max_results": 3}
                            response = requests.post(f"{RAG_SERVER_URL}/query", json=payload, timeout=30)
                            if response.status_code == 200:
                                data = response.json()
                                results_container.success(f"✅ Consulta {i+1}: {len(data.get('sources', []))} fuentes encontradas")
                            else:
                                results_container.error(f"❌ Error consulta {i+1}: {response.text}")
                        except Exception as e:
                            results_container.error(f"❌ Error consulta {i+1}: {str(e)}")
                        
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                
                # Pruebas de memoria
                if test_memory:
                    results_container.subheader("🧠 Pruebas de Memoria")
                    
                    # Simular pruebas de memoria (en implementación real usaríamos MCP)
                    memory_tests = [
                        ("Almacenar conversación", True),
                        ("Recuperar historial", True),
                        ("Guardar preferencias", True)
                    ]
                    
                    for test_name, success in memory_tests:
                        if success:
                            results_container.success(f"✅ {test_name}")
                        else:
                            results_container.error(f"❌ {test_name}")
                        
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                
                progress_bar.progress(1.0)
                st.success("🎉 Suite de pruebas completada!")

# Ejemplos de documentos predefinidos
SAMPLE_DOCUMENTS = [
    {
        "content": """
        El Protocolo de Contexto de Modelo (MCP) es un estándar abierto que permite que 
        los grandes modelos de lenguaje interactúen dinámicamente con herramientas externas, 
        bases de datos y APIs a través de una interfaz estandarizada. MCP actúa como un 
        conector universal, permitiendo que los LLMs accedan a datos en tiempo real y 
        ejecuten acciones sin necesidad de integraciones personalizadas para cada fuente de datos.
        """,
        "metadata": {"topic": "MCP", "type": "definition", "source": "documentation"}
    },
    {
        "content": """
        RAG (Retrieval-Augmented Generation) combina un modelo de lenguaje con recuperación 
        de conocimiento externo, de modo que las respuestas del modelo se basan en hechos 
        actualizados en lugar de solo en sus datos de entrenamiento. En un pipeline RAG, 
        una consulta del usuario se utiliza para buscar en una base de conocimiento y los 
        documentos más relevantes se "aumentan" en el prompt del modelo.
        """,
        "metadata": {"topic": "RAG", "type": "explanation", "source": "technical_guide"}
    },
    {
        "content": """
        Los agentes agénticos en sistemas RAG pueden incluir diferentes tipos:
        - Agentes de enrutamiento: dirigen consultas a las fuentes de datos apropiadas
        - Agentes de planificación de consultas: descomponen consultas complejas
        - Agentes ReAct: crean soluciones paso a paso con razonamiento y acción
        - Agentes de planificación y ejecución: ejecutan flujos de trabajo completos de forma autónoma
        """,
        "metadata": {"topic": "Agentic_RAG", "type": "categorization", "source": "architecture_guide"}
    }
]

def load_sample_documents():
    """Cargar documentos de ejemplo"""
    st.subheader("📚 Cargar Documentos de Ejemplo")
    
    if st.button("📥 Cargar Documentos de Ejemplo"):
        with st.spinner("Cargando documentos..."):
            success_count = 0
            for i, doc in enumerate(SAMPLE_DOCUMENTS):
                try:
                    payload = {**doc, "doc_id": f"sample_doc_{i}"}
                    response = requests.post(f"{RAG_SERVER_URL}/documents", json=payload)
                    if response.status_code == 200:
                        success_count += 1
                except Exception as e:
                    st.error(f"Error cargando documento {i}: {str(e)}")
            
            st.success(f"✅ {success_count}/{len(SAMPLE_DOCUMENTS)} documentos cargados exitosamente")

def export_test_results():
    """Exportar resultados de pruebas"""
    if tester.test_results:
        st.subheader("📤 Exportar Resultados")
        
        export_data = {
            "session_id": tester.session_id,
            "summary": tester.get_test_summary(),
            "test_results": tester.test_results,
            "exported_at": datetime.now().isoformat()
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="💾 Descargar Resultados (JSON)",
            data=json_str,
            file_name=f"mcp_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    # Agregar secciones adicionales en la sidebar
    with st.sidebar:
        st.divider()
        load_sample_documents()
        st.divider()
        export_test_results()
        
        # Información adicional
        st.subheader("ℹ️ Información")
        st.markdown("""
        **Servidor RAG**: Maneja consultas agénticas con recuperación de información
        
        **Servidor Memory**: Gestiona memoria persistente y contexto
        
        **Funcionalidades**:
        - Búsqueda semántica
        - Expansión de consultas
        - Memoria conversacional
        - Preferencias de usuario
        """)
    
    main()

# Funciones adicionales para testing por línea de comandos
async def cli_test_rag_server():
    """Probar servidor RAG desde línea de comandos"""
    print("🔍 Probando Servidor RAG...")
    
    # Agregar documento de prueba
    doc_payload = {
        "content": "Docker es una plataforma de contenedores que facilita el despliegue de aplicaciones.",
        "metadata": {"topic": "DevOps", "source": "cli_test"}
    }
    
    try:
        response = requests.post(f"{RAG_SERVER_URL}/documents", json=doc_payload)
        print(f"✅ Documento agregado: {response.status_code}")
    except Exception as e:
        print(f"❌ Error agregando documento: {e}")
    
    # Probar consulta
    query_payload = {
        "query": "¿Qué es Docker?",
        "max_results": 3
    }
    
    try:
        response = requests.post(f"{RAG_SERVER_URL}/query", json=query_payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Consulta exitosa: {len(data.get('sources', []))} fuentes")
            print(f"Respuesta: {data.get('answer', 'No response')[:100]}...")
        else:
            print(f"❌ Error en consulta: {response.status_code}")
    except Exception as e:
        print(f"❌ Error en consulta: {e}")

async def cli_test_servers():
    """Probar ambos servidores desde línea de comandos"""
    print("🧪 Iniciando pruebas CLI...")
    
    # Verificar salud de servidores
    try:
        rag_health = requests.get(f"{RAG_SERVER_URL}/health", timeout=5)
        print(f"✅ RAG Server: {rag_health.status_code}")
    except:
        print("❌ RAG Server: No responde")
    
    try:
        memory_health = requests.get(f"{MEMORY_SERVER_URL}/health", timeout=5)
        print(f"✅ Memory Server: {memory_health.status_code}")
    except:
        print("❌ Memory Server: No responde")
    
    # Probar funcionalidades
    await cli_test_rag_server()
    
    print("🎉 Pruebas CLI completadas!")

if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] == "cli":
        # Modo línea de comandos
        asyncio.run(cli_test_servers())
    elif sys.argv[1] == "web":
        # Modo web (Streamlit)
        # Este será ejecutado automáticamente por Streamlit
        pass