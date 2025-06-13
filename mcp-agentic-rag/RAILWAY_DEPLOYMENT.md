# 🚂 Guía de Deployment en Railway

Esta guía te ayuda a desplegar tu sistema **MCP Agéntico RAG** en Railway de forma sencilla.

## 📋 Prerrequisitos

- Cuenta en [Railway](https://railway.app)
- Cuenta de GitHub (para conectar el repositorio)
- API Key de OpenAI (opcional pero recomendado)

## 🚀 Opción 1: Deploy desde GitHub (Recomendado)

### 1. Preparar tu repositorio
```bash
# Asegúrate de que estos archivos estén en tu repo:
- Dockerfile.railway
- requirements_cloud.txt
- rag_mcp_server_cloud.py
- knowledge_base/ (directorio con tus datos)
```

### 2. Crear proyecto en Railway
1. Ve a [railway.app](https://railway.app)
2. Haz clic en **"New Project"**
3. Selecciona **"Deploy from GitHub repo"**
4. Conecta tu cuenta de GitHub si no lo has hecho
5. Selecciona tu repositorio

### 3. Configurar el Dockerfile
1. En Railway, ve a **Settings** → **Build**
2. Cambia **Dockerfile Path** a: `Dockerfile.railway`
3. Guarda los cambios

### 4. Configurar Variables de Entorno
Ve a **Settings** → **Variables** y agrega:

```
OPENAI_API_KEY=tu_api_key_aqui
ENVIRONMENT=railway
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
VECTOR_DB_TYPE=chroma
```

### 5. Deploy!
1. Haz clic en **"Deploy"**
2. Railway construirá y desplegará tu aplicación
3. Una vez terminado, ve a **Settings** → **Domains**
4. Haz clic en **"Generate Domain"**

## 🐳 Opción 2: Deploy desde Imagen Docker

### 1. Construir la imagen
```bash
# En tu terminal local
docker build -f Dockerfile.railway -t tu-usuario/rag-mcp:latest .
docker push tu-usuario/rag-mcp:latest
```

### 2. Deploy en Railway
1. Crea un **"Empty Project"** en Railway
2. Haz clic en **"Add a Service"**
3. Selecciona **"Docker Image"**
4. Ingresa: `tu-usuario/rag-mcp:latest`
5. Configura variables de entorno como en la Opción 1

## 🛠️ Opción 3: Deploy con Railway CLI

### 1. Instalar CLI
```bash
npm install -g @railway/cli
railway login
```

### 2. Deploy desde local
```bash
cd mcp-agentic-rag
railway init
railway up --dockerfile Dockerfile.railway
```

## ✅ Verificar el Deployment

### 1. Revisar Logs
- Ve a tu proyecto en Railway
- Haz clic en **"View Logs"**
- Busca: `🚀 Iniciando servidor RAG Cloud...`

### 2. Probar la API
```bash
# Usa tu dominio de Railway
curl https://tu-app.railway.app/health

# Debería devolver:
{
  "status": "healthy",
  "vector_db": "chroma_in_memory",
  "version": "cloud",
  "documents_loaded": X
}
```

### 3. Probar consulta
```bash
curl -X POST https://tu-app.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué vinos tienes disponibles?",
    "max_results": 3
  }'
```

## 🔧 Configuración Avanzada

### Variables de Entorno Adicionales

```
# OpenAI personalizado
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Para debugging
PYTHONPATH=/app
LOG_LEVEL=INFO
```

### Escalado Automático
Railway escala automáticamente basado en el tráfico. Para configurar:
1. Ve a **Settings** → **Deploy**
2. Ajusta **CPU** y **Memory** según necesites

## 🎯 URLs Importantes

Una vez desplegado, tendrás acceso a:

- **API Health**: `https://tu-app.railway.app/health`
- **API Docs**: `https://tu-app.railway.app/docs`
- **Query Endpoint**: `https://tu-app.railway.app/query`
- **Add Document**: `https://tu-app.railway.app/documents`

## ⚠️ Limitaciones en Railway

- **Base de datos**: Solo ChromaDB in-memory (datos se pierden al reiniciar)
- **Memoria**: Limitada según tu plan de Railway
- **Red**: Solo el servicio principal (no redis/chromadb externos)

## 💡 Consejos

1. **Monitoring**: Usa Railway's built-in monitoring
2. **Logs**: Railway mantiene logs por 7 días
3. **Backup**: Los documentos se cargan desde `knowledge_base/` al iniciar
4. **Performance**: Considera upgrader a Railway Pro para mejor rendimiento

## 🆘 Troubleshooting

### Error: "Module not found"
- Verificar que `requirements_cloud.txt` tenga todas las dependencias
- Revisar que el Dockerfile copie todos los archivos necesarios

### Error: "Port binding failed"
- Railway asigna el puerto automáticamente via `$PORT`
- No hardcodees puertos en tu código

### Error: "Build timeout"
- Usar `requirements_cloud.txt` (más ligero que `requirements.txt`)
- Verificar que no haya dependencias innecesarias

### Performance lento
- Railway free tier tiene limitaciones
- Considera Railway Pro para mejor performance
- Optimiza tu base de conocimientos

¡Tu sistema RAG ya está listo en Railway! 🎉 