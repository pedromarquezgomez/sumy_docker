
# test_data/load_test_data.py
#!/usr/bin/env python3
"""
Script para cargar datos de prueba en el sistema MCP
"""

import json
import requests
import time
import sys
from pathlib import Path

# Configuración
RAG_SERVER_URL = "http://localhost:8000"
MEMORY_SERVER_URL = "http://localhost:8002"

def load_json_file(filename):
    """Cargar archivo JSON"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error cargando {filename}: {e}")
        return None

def wait_for_server(url, timeout=60):
    """Esperar a que el servidor esté disponible"""
    print(f"⏳ Esperando servidor {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Servidor {url} disponible")
                return True
        except:
            pass
        time.sleep(2)
    
    print(f"❌ Servidor {url} no disponible después de {timeout}s")
    return False

def load_test_documents():
    """Cargar documentos de prueba"""
    print("\n📄 Cargando documentos de prueba...")
    
    # Cargar desde test_queries.json
    queries_data = load_json_file("test_queries.json")
    if queries_data and "test_documents" in queries_data:
        for doc in queries_data["test_documents"]:
            try:
                response = requests.post(
                    f"{RAG_SERVER_URL}/documents",
                    json={
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "doc_id": doc["doc_id"]
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    print(f"✅ Documento {doc['doc_id']} cargado")
                else:
                    print(f"❌ Error cargando {doc['doc_id']}: {response.text}")
            except Exception as e:
                print(f"❌ Error con documento {doc['doc_id']}: {e}")
    
    # Cargar desde sample_documents.json 
    sample_docs = load_json_file("sample_documents.json")
    if sample_docs and "documents" in sample_docs:
        for doc in sample_docs["documents"]:
            try:
                response = requests.post(
                    f"{RAG_SERVER_URL}/documents",
                    json={
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "doc_id": doc["id"]
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    print(f"✅ Documento {doc['id']} cargado")
                else:
                    print(f"❌ Error cargando {doc['id']}: {response.text}")
            except Exception as e:
                print(f"❌ Error con documento {doc['id']}: {e}")

def run_sample_queries():
    """Ejecutar consultas de prueba"""
    print("\n🔍 Ejecutando consultas de prueba...")
    
    queries_data = load_json_file("test_queries.json")
    if not queries_data or "test_queries" not in queries_data:
        print("❌ No se encontraron consultas de prueba")
        return
    
    for query_data in queries_data["test_queries"][:3]:  # Solo primeras 3
        query = query_data["query"]
        print(f"\n🤔 Consulta: {query}")
        
        try:
            response = requests.post(
                f"{RAG_SERVER_URL}/query",
                json={"query": query, "max_results": 3},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Respuesta obtenida")
                print(f"📚 Fuentes encontradas: {len(result.get('sources', []))}")
                if result.get('answer'):
                    print(f"💬 Respuesta: {result['answer'][:100]}...")
            else:
                print(f"❌ Error en consulta: {response.text}")
        except Exception as e:
            print(f"❌ Error ejecutando consulta: {e}")

def main():
    """Función principal"""
    print("🚀 Cargador de Datos de Prueba MCP")
    print("=" * 40)
    
    # Verificar servidores
    if not wait_for_server(RAG_SERVER_URL):
        print("❌ Servidor RAG no disponible")
        sys.exit(1)
    
    # Cargar datos
    load_test_documents()
    
    # Probar consultas
    run_sample_queries()
    
    print("\n🎉 ¡Datos de prueba cargados exitosamente!")
    print("\nAhora puedes:")
    print("- 🌐 Abrir http://localhost:8003 para la interfaz web")
    print("- 🤖 Usar Claude Desktop con los servidores MCP")
    print("- 💻 Probar el cliente CLI interactivo")

if __name__ == "__main__":
    main()
