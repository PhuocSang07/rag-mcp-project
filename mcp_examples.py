#!/usr/bin/env python3
"""
MCP Client Example for RAG Server

This example demonstrates how to interact with the RAG MCP Server using the MCP protocol.
In practice, you would use an MCP client like Claude Desktop, but this shows the protocol structure.
"""

import asyncio
import json
from pathlib import Path

def example_mcp_interactions():
    """Show example MCP interactions with the RAG server."""
    
    print("🚀 RAG MCP Server - MCP Protocol Examples")
    print("=" * 50)
    
    print("\n1. Initialize RAG System")
    print("Tool: initialize-rag")
    print("Parameters: {}")
    print("Expected Response: 'RAG system initialized successfully!'")
    
    print("\n2. Add Document from Content")
    print("Tool: add-document")
    add_doc_params = {
        "name": "Machine Learning Basics",
        "content": """
        Machine Learning is a subset of artificial intelligence (AI) that provides 
        systems the ability to automatically learn and improve from experience without 
        being explicitly programmed. Machine learning focuses on the development of 
        computer programs that can access data and use it to learn for themselves.
        """,
        "type": "txt"
    }
    print(f"Parameters: {json.dumps(add_doc_params, indent=2)}")
    print("Expected Response: 'Added document 'Machine Learning Basics' with X chunks to RAG system.'")
    
    print("\n3. Upload Document File")
    print("Tool: upload-document")
    upload_params = {
        "file_path": "C:/Users/Documents/research_paper.pdf",
        "name": "Research Paper"
    }
    print(f"Parameters: {json.dumps(upload_params, indent=2)}")
    print("Expected Response: 'Uploaded document 'Research Paper' with X chunks to RAG system.'")
    
    print("\n4. Query RAG System")
    print("Tool: query-rag")
    query_params = {
        "question": "What is machine learning?",
        "include_sources": True
    }
    print(f"Parameters: {json.dumps(query_params, indent=2)}")
    print("Expected Response: AI-generated answer with source citations")
    
    print("\n5. List All Documents")
    print("Tool: list-documents")
    print("Parameters: {}")
    print("Expected Response: List of all documents with metadata")
    
    print("\n6. Remove Document")
    print("Tool: remove-document")
    remove_params = {
        "document_id": "example-uuid-here"
    }
    print(f"Parameters: {json.dumps(remove_params, indent=2)}")
    print("Expected Response: 'Removed document 'Document Name' from RAG system.'")
    
    print("\n" + "=" * 50)
    print("📚 Available Resources:")
    print("- doc://rag/{id} - Individual document metadata")
    print("- system://rag/status - RAG system status")
    
    print("\n💬 Available Prompts:")
    print("- rag-query - Query the RAG system with questions")
    print("- document-summary - Get summary of stored documents")
    
    print("\n🔧 Claude Desktop Configuration:")
    config = {
        "mcpServers": {
            "rag-mcp-server": {
                "command": "uv",
                "args": ["--directory", str(Path.cwd()), "run", "rag-mcp-server"],
                "description": "RAG MCP Server with LangChain, Qdrant, and Gemini"
            }
        }
    }
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    example_mcp_interactions()
