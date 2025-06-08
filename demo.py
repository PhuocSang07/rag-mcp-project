#!/usr/bin/env python3
"""
Example usage script for RAG MCP Server.
Demonstrates how to use the RAG functionality directly.
"""

import json
import asyncio
import sys
import uuid
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_mcp_server.server import (
    initialize_rag_components, 
    vector_store, 
    retrieval_chain,
    document_registry,
    split_documents
)
from langchain.schema import Document

async def add_sample_document(name: str, content: str) -> str:
    """Add a document directly to the RAG system."""
    # Import the global variables from the server module
    from rag_mcp_server.server import vector_store
    
    if not vector_store:
        raise RuntimeError("RAG system not initialized")
    
    # Create document and split into chunks
    document = Document(page_content=content, metadata={"source": name, "type": "txt"})
    chunks = split_documents([document])
    
    # Add to vector store
    doc_ids = await asyncio.get_event_loop().run_in_executor(
        None, lambda: vector_store.add_documents(chunks)
    )
    
    # Register document
    doc_id = str(uuid.uuid4())
    document_registry[doc_id] = {
        "name": name,
        "type": "txt", 
        "chunks": len(chunks),
        "timestamp": str(asyncio.get_event_loop().time()),
        "vector_ids": doc_ids
    }
    
    return doc_id

async def query_rag_system(question: str) -> str:
    """Query the RAG system directly."""
    # Import the global variables from the server module
    from rag_mcp_server.server import retrieval_chain
    
    if not retrieval_chain:
        raise RuntimeError("RAG system not initialized")
    
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: retrieval_chain({"query": question})
    )
    
    return result["result"]

async def demo_rag_workflow():
    """Demonstrate a complete RAG workflow"""
    print("🚀 RAG MCP Server Demo")
    print("=" * 40)
    
    # Initialize RAG components
    print("1. Initializing RAG components...")
    success = await initialize_rag_components()
    if not success:
        print("❌ Failed to initialize RAG components!")
        return
    print("✅ RAG components initialized!\n")
    
    # Example 1: Add a document
    print("2. Adding a sample document...")
    sample_document = """
    Python is a high-level, interpreted programming language with dynamic semantics. 
    Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
    make it very attractive for Rapid Application Development, as well as for use as a scripting 
    or glue language to connect existing components together.
    
    Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost 
    of program maintenance. Python supports modules and packages, which encourages program 
    modularity and code reuse.
    """
    
    try:
        doc_id = await add_sample_document("Python Programming Guide", sample_document)
        print(f"✅ Document added with ID: {doc_id}\n")
    except Exception as e:
        print(f"❌ Error adding document: {e}\n")
        return
    
    # Example 2: Query the RAG system
    print("3. Querying the RAG system...")
    questions = [
        "What is Python?",
        "What are the benefits of Python for development?",
        "Does Python support modules?"
    ]
    
    for question in questions:
        print(f"❓ Question: {question}")
        try:
            answer = await query_rag_system(question)
            print(f"💡 Answer: {answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    # Example 3: List documents
    print("4. Listing all documents...")
    for doc_id, metadata in document_registry.items():
        print(f"📄 Document: {metadata['name']} (ID: {doc_id})")
        print(f"   Chunks: {metadata['chunks']}, Added: {metadata.get('timestamp', 'N/A')}")
    
    print("\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(demo_rag_workflow())
