#!/bin/bash
# Script de configuración inicial

echo "🚀 Configurando proyecto MCP Agentic RAG..."

# Crear directorios necesarios
mkdir -p data knowledge_base test_data docker

# Copiar archivo de ejemplo de configuración
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Archivo .env creado. Por favor, edita las variables de entorno."
fi

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Por favor, instala Docker primero."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado. Por favor, instala Docker Compose primero."
    exit 1
fi

echo "✅ Docker y Docker Compose encontrados"

# Construir imágenes
echo "🏗️ Construyendo imágenes Docker..."
docker-compose build

echo "📋 Configuración completa!"
echo ""
echo "Próximos pasos:"
echo "1. Edita el archivo .env con tus credenciales"
echo "2. Ejecuta: docker-compose up -d"
echo "3. Accede al tester en: http://localhost:8003"
echo "4. Configura Claude Desktop con claude_desktop_config.json"
