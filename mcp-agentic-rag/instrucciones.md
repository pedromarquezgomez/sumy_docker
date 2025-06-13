# 🚀 Sistema MCP Agentic RAG - Guía Completa de Uso

## 📁 Estructura del Proyecto

Crea la siguiente estructura de directorios:

```
mcp-agentic-rag/
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .env
├── rag_mcp_server.py
├── memory_mcp_server.py
├── mcp_tester.py
├── claude_client.py
├── claude_desktop_config.json
├── docker/
│   ├── Dockerfile.rag-server
│   ├── Dockerfile.memory-server
│   ├── Dockerfile.tester
│   └── Dockerfile.client
├── knowledge_base/
│   ├── sample_doc_1.txt
│   ├── sample_doc_2.txt
│   └── sample_doc_3.txt
├── test_data/
│   └── test_queries.json
├── scripts/
│   ├── setup.sh
│   ├── test.sh
│   └── claude-setup.sh
└── README.md
```

## 🏗️ Instalación Paso a Paso

### 1. Configuración Inicial

```bash
# Crear directorio del proyecto
mkdir mcp-agentic-rag
cd mcp-agentic-rag

# Crear subdirectorios
mkdir -p docker knowledge_base test_data scripts data

# Crear archivos principales (copiar contenido de los artifacts)
touch docker-compose.yml
touch requirements.txt
touch rag_mcp_server.py
touch memory_mcp_server.py
touch mcp_tester.py
touch claude_client.py
```

### 2. Configurar Variables de Entorno

```bash
# Copiar y editar archivo de configuración
cp .env.example .env

# Editar .env con tus credenciales:
# OPENAI_API_KEY=tu_clave_openai_aqui
# VECTOR_DB_TYPE=chroma
# REDIS_URL=redis://redis:6379
```

### 3. Construir y Ejecutar

```bash
# Hacer ejecutables los scripts
chmod +x scripts/*.sh

# Ejecutar configuración
./scripts/setup.sh

# Levantar servicios
docker-compose up -d

# Verificar que todo funcione
docker-compose ps
```

## 🎯 Formas de Uso

### 1. 🌐 Interfaz Web (Recomendado para Testing)

```bash
# Acceder al tester web
open http://localhost:8003

# O en tu navegador:
# http://localhost:8003
```

**Funcionalidades disponibles:**
- ✅ Agregar documentos a la base de conocimiento
- 🔍 Realizar consultas RAG agénticas
- 💾 Probar sistema de memoria
- 📊 Ver estadísticas y resultados
- 🧪 Ejecutar suite de pruebas automatizadas

### 2. 🤖 Integración con Claude Desktop

```bash
# Configurar Claude Desktop automáticamente
./scripts/claude-setup.sh

# O configurar manualmente:
# 1. Abrir Claude Desktop
# 2. Ir a Settings > Developer
# 3. Edit Config
# 4. Pegar contenido de claude_desktop_config.json
```

**En Claude Desktop podrás:**
- 🔍 Usar tool `search_knowledge` para buscar información
- 📄 Usar tool `add_document` para agregar contenido
- 📊 Usar tool `get_collection_stats` para estadísticas
- 💾 Acceder a recursos de memoria persistente

### 3. 💻 Cliente de Línea de Comandos

```bash
# Modo interactivo
docker-compose exec demo-client python claude_client.py interactive

# Comandos disponibles en modo interactivo:
# search <consulta>     - Buscar en la base de conocimiento
# add <contenido>       - Agregar nuevo documento
# history               - Ver historial de conversación
# stats                 - Ver estadísticas del sistema
# help                  - Mostrar ayuda
# quit                  - Salir
```

### 4. 🌐 API HTTP Directa

```bash
# Agregar documento
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "El machine learning es una rama de la IA",
    "metadata": {"topic": "AI", "source": "manual"}
  }'

# Realizar consulta RAG
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué es machine learning?",
    "max_results": 5
  }'

# Verificar salud del sistema
curl http://localhost:8000/health
```

## 🧪 Testing y Validación

### Pruebas Automatizadas

```bash
# Ejecutar suite completa de pruebas
./scripts/test.sh

# Pruebas específicas
docker-compose exec mcp-tester python mcp_tester.py cli
```

### Verificaciones Manuales

```bash
# Verificar servicios están corriendo
docker-compose ps

# Ver logs en tiempo real
docker-compose logs -f rag-mcp-server
docker-compose logs -f memory-mcp-server

# Verificar bases de datos
docker-compose exec chromadb curl http://localhost:8000/api/v1/heartbeat
docker-compose exec redis redis-cli ping
```

## 📝 Ejemplos de Uso con Claude

Una vez configurado en Claude Desktop, puedes usar estos ejemplos:

### Consultas Básicas
```
"Usa la herramienta search_knowledge para buscar información sobre MCP"
```

### Agregar Conocimiento
```
"Agrega este documento a la base de conocimiento usando add_document: 
'FastAPI es un framework web moderno para Python que permite crear APIs rápidamente'"
```

### Consultas Contextuales
```
"Busca información sobre RAG agéntico y explícame las diferencias con RAG tradicional"
```

### Usar Recursos
```
"Carga el recurso knowledge://stats y muéstrame las estadísticas de la base de conocimiento"
```

## 🔧 Personalización y Extensión

### Agregar Nuevos Documentos

```python
# Via API
import requests

documents = [
    {
        "content": "Tu contenido aquí",
        "metadata": {"topic": "tu_topic", "source": "tu_fuente"},
        "doc_id": "documento_unico_1"
    }
]

for doc in documents:
    response = requests.post("http://localhost:8000/documents", json=doc)
    print(f"Documento agregado: {response.json()}")
```

### Modificar Configuración

```bash
# Cambiar modelo de embeddings
# Editar en rag_mcp_server.py línea:
# self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Por:
# self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Usar Pinecone en lugar de ChromaDB
# En .env cambiar:
# VECTOR_DB_TYPE=pinecone
# PINECONE_API_KEY=tu_clave
# PINECONE_ENVIRONMENT=tu_entorno
```

### Agregar Nuevos Tools MCP

```python
# En rag_mcp_server.py, agregar en list_tools():
types.Tool(
    name="tu_nueva_herramienta",
    description="Descripción de tu herramienta",
    inputSchema={
        "type": "object",
        "properties": {
            "parametro": {"type": "string", "description": "Tu parámetro"}
        },
        "required": ["parametro"]
    }
)

# Y en call_tool(), agregar:
elif name == "tu_nueva_herramienta":
    # Tu lógica aquí
    return [types.TextContent(type="text", text="Tu respuesta")]
```

## 🚨 Solución de Problemas

### Servicios No Responden

```bash
# Verificar Docker
docker --version
docker-compose --version

# Reiniciar servicios
docker-compose down
docker-compose up -d

# Ver logs para errores
docker-compose logs
```

### Claude Desktop No Detecta Servidores

```bash
# Verificar configuración
cat ~/.config/claude/claude_desktop_config.json  # Linux
cat ~/Library/Application\ Support/claude/claude_desktop_config.json  # macOS

# Verificar rutas absolutas
pwd  # Debe coincidir con las rutas en la configuración

# Reiniciar Claude Desktop completamente
```

### Errores de API Keys

```bash
# Verificar variables de entorno
docker-compose exec rag-mcp-server env | grep OPENAI
docker-compose exec rag-mcp-server env | grep CHROMA

# Actualizar .env y reiniciar
docker-compose down
docker-compose up -d
```

### Base Vectorial No Funciona

```bash
# Verificar ChromaDB
curl http://localhost:8001/api/v1/heartbeat

# Recrear volúmenes si es necesario
docker-compose down -v
docker-compose up -d
```

## 📊 Monitoreo y Métricas

### Estadísticas del Sistema

```bash
# Via API
curl http://localhost:8000/health
curl http://localhost:8002/health

# Via cliente
docker-compose exec demo-client python -c "
import asyncio
from claude_client import MCPClient
client = MCPClient()
asyncio.run(client.connect_to_servers())
stats = asyncio.run(client.get_knowledge_stats())
print(stats)
"
```

### Logs en Tiempo Real

```bash
# Todos los servicios
docker-compose logs -f

# Servicio específico
docker-compose logs -f rag-mcp-server
docker-compose logs -f memory-mcp-server
```

## 🎉 ¡Listo para Usar!

Tu sistema MCP Agentic RAG está configurado y listo. Puedes:

1. **🌐 Explorar** la interfaz web en `http://localhost:8003`
2. **🤖 Integrar** con Claude Desktop usando los MCP tools
3. **💻 Experimentar** con el cliente CLI interactivo
4. **🔧 Personalizar** agregando tus propios documentos y herramientas
5. **📊 Monitorear** el rendimiento y estadísticas

¡Disfruta explorando las capacidades de RAG agéntico con MCP! 🚀