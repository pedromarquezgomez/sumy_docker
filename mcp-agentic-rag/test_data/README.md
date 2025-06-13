
# Test Data Directory

Esta carpeta contiene todos los datos y configuraciones necesarias para probar el sistema MCP Agentic RAG.

## 📁 Archivos Incluidos

### `test_queries.json`
- Conjunto de consultas de prueba con diferentes niveles de dificultad
- Documentos de ejemplo para testing básico
- Escenarios de validación predefinidos

### `sample_documents.json`  
- Documentos más detallados sobre MCP, RAG y tecnologías relacionadas
- Metadatos enriquecidos para testing avanzado
- Contenido técnico para validar búsqueda semántica

### `test_config.json`
- Configuración de parámetros de testing
- URLs de servidores y timeouts
- Reglas de validación y métricas de rendimiento

### `load_test_data.py`
- Script automático para cargar todos los datos de prueba
- Verificación de conectividad de servidores
- Ejecución de consultas de validación

## 🚀 Uso Rápido

```bash
# Entrar a la carpeta test_data
cd test_data

# Ejecutar carga automática de datos
python load_test_data.py

# O cargar manualmente via API
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d @sample_documents.json
```

## 🧪 Tipos de Testing Incluidos

1. **Testing Básico**: Documentos simples para validar funcionalidad core
2. **Testing Semántico**: Consultas complejas para validar búsqueda inteligente  
3. **Testing de Memoria**: Escenarios para validar persistencia y contexto
4. **Testing de Rendimiento**: Métricas y benchmarks de velocidad

## 📊 Métricas de Validación

- ✅ Tiempo de respuesta < 3 segundos
- ✅ Relevancia de resultados > 0.3
- ✅ Precisión en búsqueda semántica
- ✅ Persistencia de memoria entre sesiones