#!/usr/bin/env python3
"""
Simple test script to validate RAG MCP Server functionality.
This script tests the RAG initialization and basic functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_mcp_server.server import initialize_rag_components, server

async def test_initialization():
    """Test RAG component initialization"""
    print("🚀 Testing RAG MCP Server initialization...")
    
    try:
        await initialize_rag_components()
        print("✅ RAG components initialized successfully!")
        
        # Test basic functionality
        from rag_mcp_server.server import qdrant_client, embeddings, llm
        
        if qdrant_client is not None:
            print("✅ Qdrant client initialized")
        else:
            print("❌ Qdrant client not initialized")
            
        if embeddings is not None:
            print("✅ Google embeddings initialized")
        else:
            print("❌ Google embeddings not initialized")
            
        if llm is not None:
            print("✅ Google LLM initialized")
        else:
            print("❌ Google LLM not initialized")
            
        print("\n🎉 RAG MCP Server is ready for use!")
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

async def main():
    """Main test function"""
    print("RAG MCP Server Test Suite")
    print("=" * 40)
    
    success = await test_initialization()
    
    if success:
        print("\n✅ All tests passed! Server is ready.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Check your configuration.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
