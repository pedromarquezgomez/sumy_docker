#!/bin/bash

# Script para probar tu deployment en Railway
# Uso: ./test_railway.sh https://tu-app.railway.app

if [ -z "$1" ]; then
    echo "❌ Error: Debes proporcionar la URL de tu app Railway"
    echo "Uso: $0 https://tu-app.railway.app"
    exit 1
fi

BASE_URL="$1"
echo "🧪 Probando deployment en Railway: $BASE_URL"
echo "================================================"

# Test 1: Health Check
echo ""
echo "1️⃣ Probando Health Check..."
response=$(curl -s -w "HTTP_CODE:%{http_code}" "$BASE_URL/health")
http_code=$(echo "$response" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
body=$(echo "$response" | sed 's/HTTP_CODE:[0-9]*$//')

if [ "$http_code" = "200" ]; then
    echo "✅ Health Check OK"
    echo "📊 Respuesta: $body" | jq . 2>/dev/null || echo "📊 Respuesta: $body"
else
    echo "❌ Health Check Failed (HTTP $http_code)"
    echo "📊 Respuesta: $body"
fi

# Test 2: API Docs
echo ""
echo "2️⃣ Verificando API Docs..."
response=$(curl -s -w "HTTP_CODE:%{http_code}" "$BASE_URL/docs")
http_code=$(echo "$response" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)

if [ "$http_code" = "200" ]; then
    echo "✅ API Docs disponible en: $BASE_URL/docs"
else
    echo "❌ API Docs no disponible (HTTP $http_code)"
fi

# Test 3: Query RAG
echo ""
echo "3️⃣ Probando consulta RAG..."
query_response=$(curl -s -w "HTTP_CODE:%{http_code}" \
    -X POST "$BASE_URL/query" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "¿Qué vinos tienes disponibles?",
        "max_results": 3
    }')

http_code=$(echo "$query_response" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
body=$(echo "$query_response" | sed 's/HTTP_CODE:[0-9]*$//')

if [ "$http_code" = "200" ]; then
    echo "✅ Consulta RAG OK"
    echo "🤖 Respuesta:" 
    echo "$body" | jq '.answer' 2>/dev/null || echo "$body"
else
    echo "❌ Consulta RAG Failed (HTTP $http_code)"
    echo "📊 Respuesta: $body"
fi

# Test 4: Agregar documento
echo ""
echo "4️⃣ Probando agregar documento..."
add_response=$(curl -s -w "HTTP_CODE:%{http_code}" \
    -X POST "$BASE_URL/documents" \
    -H "Content-Type: application/json" \
    -d '{
        "content": "Test document from Railway deployment",
        "metadata": {"type": "test", "source": "railway_test"}
    }')

http_code=$(echo "$add_response" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
body=$(echo "$add_response" | sed 's/HTTP_CODE:[0-9]*$//')

if [ "$http_code" = "200" ]; then
    echo "✅ Agregar documento OK"
    echo "📄 Documento agregado:" 
    echo "$body" | jq . 2>/dev/null || echo "$body"
else
    echo "❌ Agregar documento Failed (HTTP $http_code)"
    echo "📊 Respuesta: $body"
fi

echo ""
echo "🎯 URLs importantes:"
echo "==================="
echo "🏥 Health Check:  $BASE_URL/health"
echo "📚 API Docs:      $BASE_URL/docs"
echo "🤖 Query RAG:     $BASE_URL/query"
echo "📄 Add Document:  $BASE_URL/documents"

echo ""
echo "🎉 Testing completado!" 