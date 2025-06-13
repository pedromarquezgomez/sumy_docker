#!/bin/bash
# Script de pruebas automatizadas

echo "🧪 Ejecutando suite de pruebas..."

# Verificar que los servicios estén corriendo
echo "🔍 Verificando servicios..."

# Función para verificar servicio
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:$port/health >/dev/null 2>&1; then
            echo "✅ $service_name está corriendo"
            return 0
        fi
        echo "⏳ Esperando $service_name... (intento $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $service_name no responde después de $max_attempts intentos"
    return 1
}

# Verificar servicios
check_service "RAG Server" 8000
check_service "Memory Server" 8002
check_service "Tester" 8003

# Ejecutar pruebas con el cliente CLI
echo "🔍 Ejecutando pruebas CLI..."
docker-compose exec demo-client python claude_client.py

# Ejecutar pruebas HTTP
echo "🌐 Ejecutando pruebas HTTP..."

# Probar endpoint de salud
echo "Testing RAG server health..."
curl -f http://localhost:8000/health

echo "Testing Memory server health..."
curl -f http://localhost:8002/health

# Probar agregado de documento
echo "Testing document addition..."
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "Test document content", "metadata": {"source": "test"}}'

# Probar consulta RAG
echo "Testing RAG query..."
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test document", "max_results": 3}'

echo "✅ Pruebas completadas!"