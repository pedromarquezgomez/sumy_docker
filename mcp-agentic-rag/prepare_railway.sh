#!/bin/bash

# 🚂 Script de preparación para Railway
# Prepara automáticamente tu proyecto MCP RAG para deployment en Railway

echo "🚂 Preparando proyecto para Railway..."
echo "======================================"

# Verificar archivos necesarios
echo "📋 Verificando archivos necesarios..."

required_files=(
    "rag_mcp_server_cloud.py"
    "requirements_cloud.txt"
    "Dockerfile.railway"
    "RAILWAY_DEPLOYMENT.md"
    "env.railway.example"
)

missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    else
        echo "✅ $file"
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Archivos faltantes:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Por favor asegúrate de tener todos los archivos necesarios."
    exit 1
fi

# Verificar directorio knowledge_base
echo ""
echo "📚 Verificando base de conocimientos..."
if [ -d "knowledge_base" ]; then
    file_count=$(find knowledge_base -type f | wc -l)
    echo "✅ Directorio knowledge_base encontrado con $file_count archivos"
    
    # Listar archivos en knowledge_base
    echo "   Archivos encontrados:"
    find knowledge_base -type f -exec basename {} \; | sort | sed 's/^/   - /'
else
    echo "⚠️  Directorio knowledge_base no encontrado"
    echo "   Creando directorio vacío..."
    mkdir -p knowledge_base
    echo "   ✅ Creado. Puedes agregar tus archivos de datos aquí."
fi

# Verificar que el archivo principal tenga el puerto correcto
echo ""
echo "🔧 Verificando configuración de puerto..."
if grep -q "PORT = int(os.getenv(\"PORT\", \"8080\"))" rag_mcp_server_cloud.py; then
    echo "✅ Configuración de puerto correcta para Railway"
else
    echo "⚠️  Configuración de puerto necesita ajuste"
fi

# Crear archivo .gitignore si no existe
echo ""
echo "📝 Verificando .gitignore..."
if [ ! -f ".gitignore" ]; then
    echo "Creando .gitignore..."
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Railway
.railway/

# Docker
.dockerignore

# Temporary files
tmp/
temp/
EOF
    echo "✅ .gitignore creado"
else
    echo "✅ .gitignore existe"
fi

# Mostrar siguiente pasos
echo ""
echo "🎉 ¡Proyecto preparado para Railway!"
echo "===================================="
echo ""
echo "📋 Próximos pasos:"
echo "1. Sube tu código a GitHub (si no lo has hecho)"
echo "2. Ve a railway.app y crea un nuevo proyecto"
echo "3. Conecta tu repositorio de GitHub"
echo "4. Configura Dockerfile Path: 'Dockerfile.railway'"
echo "5. Agrega variables de entorno (ver env.railway.example)"
echo "6. ¡Deploy!"
echo ""
echo "📖 Para guía detallada, revisa: RAILWAY_DEPLOYMENT.md"
echo ""

# Mostrar variables de entorno necesarias
echo "🔑 Variables de entorno requeridas en Railway:"
echo "=============================================="
cat env.railway.example
echo ""

# Mostrar comandos útiles
echo "💡 Comandos útiles:"
echo "=================="
echo "# Probar localmente:"
echo "docker build -f Dockerfile.railway -t rag-mcp:test ."
echo "docker run -p 8080:8080 --env-file .env rag-mcp:test"
echo ""
echo "# Verificar que funciona:"
echo "curl http://localhost:8080/health"
echo ""
echo "# Instalar Railway CLI:"
echo "npm install -g @railway/cli"
echo ""

echo "✨ ¡Listo para despegar en Railway! 🚀" 