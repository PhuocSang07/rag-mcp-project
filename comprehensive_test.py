#!/usr/bin/env python3
"""
Comprehensive RAG MCP Server Test

This script demonstrates the complete RAG workflow including:
1. System initialization
2. Document ingestion (both content and file upload)
3. Querying and retrieval
4. Document management
"""

import asyncio
import sys
import uuid
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_mcp_server.server import (
    initialize_rag_components,
    document_registry,
    split_documents,
    load_document
)
from langchain.schema import Document

async def comprehensive_rag_test():
    """Comprehensive test of RAG functionality."""
    print("🧪 RAG MCP Server - Comprehensive Test")
    print("=" * 50)
    
    # Step 1: Initialize RAG system
    print("1. 🚀 Initializing RAG system...")
    success = await initialize_rag_components()
    if not success:
        print("❌ Failed to initialize RAG system!")
        return False
    print("✅ RAG system initialized successfully!\n")
    
    # Step 2: Add document from content
    print("2. 📝 Adding document from content...")
    success = await add_content_document()
    if not success:
        print("❌ Failed to add content document!")
        return False
    print("✅ Content document added successfully!\n")
    
    # Step 3: Upload document files
    print("3. 📁 Uploading document files...")
    success = await upload_sample_documents()
    if not success:
        print("❌ Failed to upload documents!")
        return False
    print("✅ Sample documents uploaded successfully!\n")
    
    # Step 4: List all documents
    print("4. 📋 Listing all documents...")
    list_documents()
    print()
    
    # Step 5: Query the RAG system
    print("5. ❓ Querying RAG system...")
    await query_documents()
    print()
    
    # Step 6: Test document management
    print("6. 🗂️ Testing document management...")
    test_document_management()
    
    print("🎉 Comprehensive test completed successfully!")
    return True

async def add_content_document():
    """Add a document from text content."""
    try:
        from rag_mcp_server.server import vector_store
        
        content = """
        The Model Context Protocol (MCP) is an open protocol that enables seamless integration 
        between AI assistants and external data sources. MCP allows AI systems to securely 
        access and interact with various tools, databases, and services while maintaining 
        proper security boundaries.
        
        Key features of MCP include:
        - Standardized protocol for AI-tool integration
        - Security-first design with proper access controls
        - Support for various data sources and tools
        - Real-time bidirectional communication
        - Extensible architecture for custom implementations
        """
        
        # Create document and split into chunks
        document = Document(page_content=content, metadata={"source": "MCP Overview", "type": "txt"})
        chunks = split_documents([document])
        
        # Add to vector store
        doc_ids = await asyncio.get_event_loop().run_in_executor(
            None, lambda: vector_store.add_documents(chunks)
        )
        
        # Register document
        doc_id = str(uuid.uuid4())
        document_registry[doc_id] = {
            "name": "MCP Overview",
            "type": "txt",
            "chunks": len(chunks),
            "timestamp": str(asyncio.get_event_loop().time()),
            "vector_ids": doc_ids
        }
        
        print(f"   Added 'MCP Overview' with {len(chunks)} chunks")
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        return False

async def upload_sample_documents():
    """Upload sample documents from files."""
    try:
        from rag_mcp_server.server import vector_store
        
        sample_files = [
            Path("sample_documents/ai_overview.txt"),
            Path("sample_documents/python_best_practices.txt")
        ]
        
        for file_path in sample_files:
            if file_path.exists():
                # Load document
                documents = load_document(str(file_path), "txt")
                chunks = split_documents(documents)
                
                # Add to vector store
                doc_ids = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: vector_store.add_documents(chunks)
                )
                
                # Register document
                doc_id = str(uuid.uuid4())
                document_registry[doc_id] = {
                    "name": file_path.stem,
                    "type": "txt",
                    "chunks": len(chunks),
                    "timestamp": str(asyncio.get_event_loop().time()),
                    "vector_ids": doc_ids,
                    "source_path": str(file_path)
                }
                
                print(f"   Uploaded '{file_path.name}' with {len(chunks)} chunks")
            else:
                print(f"   Skipped '{file_path.name}' (file not found)")
                
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        return False

def list_documents():
    """List all documents in the system."""
    if not document_registry:
        print("   No documents in the system")
        return
    
    print(f"   Found {len(document_registry)} documents:")
    for doc_id, metadata in document_registry.items():
        print(f"   📄 {metadata['name']} ({metadata['type']}) - {metadata['chunks']} chunks")

async def query_documents():
    """Query the RAG system with various questions."""
    from rag_mcp_server.server import retrieval_chain
    
    if not retrieval_chain:
        print("   ❌ Retrieval chain not available")
        return
    
    questions = [
        "What is the Model Context Protocol?",
        "What are Python best practices?", 
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the key features of MCP?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"   ❓ Question {i}: {question}")
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda q=question: retrieval_chain({"query": q})
            )
            answer = result["result"]
            # Truncate long answers for display
            if len(answer) > 200:
                answer = answer[:200] + "..."
            print(f"   💡 Answer: {answer}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

def test_document_management():
    """Test document management capabilities."""
    print(f"   📊 Total documents: {len(document_registry)}")
    
    total_chunks = sum(doc['chunks'] for doc in document_registry.values())
    print(f"   📦 Total chunks: {total_chunks}")
    
    # Show document types
    doc_types = {}
    for doc in document_registry.values():
        doc_type = doc['type']
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print(f"   📊 Document types: {dict(doc_types)}")

if __name__ == "__main__":
    asyncio.run(comprehensive_rag_test())
