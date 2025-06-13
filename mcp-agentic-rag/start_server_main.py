#!/usr/bin/env python3
"""
Script de inicio para el servidor RAG MCP principal
"""
import uvicorn
from rag_mcp_server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 