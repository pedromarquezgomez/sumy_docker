#!/bin/bash
# Script para configurar Claude Desktop

echo "🤖 Configurando Claude Desktop para MCP..."

# Detectar sistema operativo
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CONFIG_DIR="$HOME/Library/Application Support/claude"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    CONFIG_DIR="$HOME/.config/claude"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    CONFIG_DIR="$APPDATA/claude"
else
    echo "❌ Sistema operativo no soportado: $OSTYPE"
    exit 1
fi

# Crear directorio de configuración si no existe
mkdir -p "$CONFIG_DIR"

# Obtener ruta absoluta del proyecto
PROJECT_DIR=$(pwd)

# Crear configuración de Claude Desktop
cat > "$CONFIG_DIR/claude_desktop_config.json" << EOF
{
  "mcpServers": {
    "agentic-rag": {
      "command": "python",
      "args": ["$PROJECT_DIR/rag_mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "$(grep OPENAI_API_KEY .env | cut -d '=' -f2)",
        "VECTOR_DB_TYPE": "chroma",
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8001"
      }
    },
    "memory-server": {
      "command": "python",
      "args": ["$PROJECT_DIR/memory_mcp_server.py"],
      "env": {
        "REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
EOF

echo "✅ Configuración de Claude Desktop creada en: $CONFIG_DIR/claude_desktop_config.json"
echo ""
echo "Próximos pasos:"
echo "1. Asegúrate de que los servicios estén corriendo: docker-compose up -d"
echo "2. Reinicia Claude Desktop"
echo "3. Verifica que los servidores MCP aparezcan en Settings > Developer"
echo "4. ¡Prueba hacer consultas usando los tools de MCP!"
