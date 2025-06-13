#!/usr/bin/env python3
"""
Script de inicio simple para el servidor RAG
"""
import uvicorn
from rag_mcp_server_simple import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 