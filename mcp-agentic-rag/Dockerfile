FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias optimizadas para cloud
COPY requirements_cloud.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar base de conocimientos
COPY knowledge_base/ knowledge_base/

# Copiar servidor cloud optimizado
COPY rag_mcp_server_cloud.py main.py

# Variables de entorno para cloud
ENV ENVIRONMENT=cloud
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Exponer puerto
EXPOSE 8080

# Comando para ejecutar el servidor HTTP directamente
CMD ["python", "main.py"] 