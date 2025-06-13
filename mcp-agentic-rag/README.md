# Sistema MCP Agentic RAG

Un sistema completo de **Retrieval-Augmented Generation (RAG) Agéntico** implementado con el **Model Context Protocol (MCP)** de Anthropic. Este proyecto demuestra cómo construir servidores MCP que exponen capacidades avanzadas de RAG a Claude Desktop y otros clientes MCP.

## 🚀 Características

### 🧠 Servidor RAG Agéntico
- **Expansión agéntica de consultas** usando LLMs
- **Búsqueda semántica** con embeddings
- **Múltiples fuentes de datos** (ChromaDB, Pinecone)
- **Generación contextual** de respuestas
- **Deduplicación inteligente** de resultados

### 💾 Servidor de Memoria
- **Memoria conversacional persistente** con Redis
- **Preferencias de usuario** personalizables
- **Conocimiento específico del dominio**
- **Patrones de consulta** frecuentes
- **Limpieza automática** de datos antiguos

### 🧪 Suite de Testing
- **Interfaz web interactiva** con Streamlit
- **Pruebas automatizadas** para ambos servidores
- **Cliente CLI** para testing rápido
- **Métricas de rendimiento** y estadísticas

### 🔌 Integración con Claude
- **Configuración automática** para Claude Desktop
- **Herramientas MCP** nativas
- **Recursos compartidos** entre sesiones
- **Contexto persistente**

## 📋 Requisitos Previos

- **Docker** y **Docker Compose**
- **Python 3.11+** (para desarrollo local)
- **Claude Desktop** (para testing con Claude)
- **OpenAI API Key** (opcional, para generación mejorada)

## 🛠️ Instalación Rápida

```bash
# 1. Clonar y configurar
git clone <repository-url>
cd mcp-agentic-rag
chmod +x scripts/*.sh
./scripts/setup.sh

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 3. Levantar servicios
docker-compose up -d

# 4. Verificar que todo funcione
./scripts/test.sh
```

## 🎯 Uso Rápido

### Con la Interfaz Web
1. Accede a `http://localhost:8003`
2. Agrega documentos en la pestaña "RAG Testing"
3. Realiza consultas y ve los resultados
4. Revisa estadísticas en "Results"

### Con Claude Desktop
1. Ejecuta: `./scripts/claude-setup.sh`
2. Reinicia Claude Desktop
3. Verifica servidores en Settings > Developer
4. ¡Haz consultas usando los tools de MCP!

### Con Cliente CLI
```bash
# Modo interactivo
docker-compose exec demo-client python claude_client.py interactive

# Comandos disponibles:
# search <consulta>     - Buscar información
# add <contenido>       - Agregar documento
# history               - Ver conversaciones
# stats                 - Ver estadísticas
```

## 📚 Arquitectura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude        │    │   Tester Web    │    │   Cliente CLI   │
│   Desktop       │    │   (Streamlit)   │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     Transport Layer       │
                    │  (MCP Protocol/HTTP)      │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
    ┌─────────▼─────────┐ ┌───────▼────────┐ ┌───────▼────────┐
    │   RAG MCP Server  │ │ Memory Server   │ │  ChromaDB      │
    │                   │ │                 │ │  Vector Store  │
    │ • Query Expansion │ │ • Conversations │ └────────────────┘
    │ • Semantic Search │ │ • User Prefs    │
    │ • Answer Gen      │ │ • Domain Know   │ ┌────────────────┐
    └───────────────────┘ └─────────────────┘ │     Redis      │
                                              │   Memory DB    │
                                              └────────────────┘
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# OpenAI (opcional, mejora respuestas)
OPENAI_API_KEY=your_key_here

# Base vectorial
VECTOR_DB_TYPE=chroma  # o 'pinecone'
CHROMA_HOST=chromadb
CHROMA_PORT=8001

# Memoria
REDIS_URL=redis://redis:6379
```

### Claude Desktop
```json
{
  "mcpServers": {
    "agentic-rag": {
      "command": "python",
      "args": ["/path/to/rag_mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your_key",
        "VECTOR_DB_TYPE": "chroma",
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8001"
      }
    }
  }
}
```

## 🧪 Testing y Validación

### Suite Automatizada
```bash
# Ejecutar todas las pruebas
./scripts/test.sh

# Pruebas específicas
docker-compose exec mcp-tester python mcp_tester.py cli
```

### Verificaciones de Salud
```bash
# Verificar servicios
curl http://localhost:8000/health  # RAG Server
curl http://localhost:8002/health  # Memory Server
curl http://localhost:8003/health  # Tester
```

### Métricas Disponibles
- **Total de documentos** en la base vectorial
- **Conversaciones almacenadas** por sesión
- **Tiempo de respuesta** promedio
- **Tasa de éxito** de consultas
- **Uso de memoria** Redis

## 🎨 Personalización

### Agregar Nuevos Tipos de Agentes
```python
# En rag_mcp_server.py
async def custom_agent_logic(self, query: str) -> List[str]:
    """Tu lógica de agente personalizada"""
    # Implementar nueva estrategia de expansión
    return expanded_queries
```

### Nuevas Fuentes de Datos
```python
# Agregar soporte para nueva DB vectorial
if VECTOR_DB_TYPE == "your_db":
    self.vector_db = YourVectorDB()
```

### Tools MCP Personalizados
```python
@mcp_server.list_tools()
async def custom_tools():
    return [
        types.Tool(
            name="your_custom_tool",
            description="Tu herramienta personalizada",
            inputSchema={...}
        )
    ]
```

## 🤝 Contribución

1. **Fork** el repositorio
2. **Crea** una rama feature (`git checkout -b feature/amazing-feature`)
3. **Commit** tus cambios (`git commit -m 'Add amazing feature'`)
4. **Push** a la rama (`git push origin feature/amazing-feature`)
5. **Abre** un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **Anthropic** por el Model Context Protocol
- **ChromaDB** por la base vectorial
- **OpenAI** por los modelos de lenguaje
- **FastAPI** por el framework web
- **Streamlit** por la interfaz de testing
