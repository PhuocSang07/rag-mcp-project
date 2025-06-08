import asyncio
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

# RAG dependencies
from langchain_google_genai import GoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Global RAG components
qdrant_client: Optional[QdrantClient] = None
vector_store: Optional[QdrantVectorStore] = None
embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
llm: Optional[GoogleGenerativeAI] = None
retrieval_chain: Optional[RetrievalQA] = None
documents_collection = "rag_documents"

# Store document metadata
document_registry: Dict[str, Dict[str, Any]] = {}

server = Server("rag-mcp-server")

async def initialize_rag_components():
    """Initialize RAG components with environment variables."""
    global qdrant_client, vector_store, embeddings, llm, retrieval_chain
    
    try:
        # Check for required environment variables (loaded from .env file)
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        if not gemini_api_key:
            logger.warning("GOOGLE_API_KEY not found in .env file. RAG functionality will be limited.")
            return False
        
        # Initialize Qdrant client (in-memory for demo, can be configured for persistent storage)
        qdrant_client = QdrantClient(location=":memory:")
          # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
          # Initialize LLM
        llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.7
        )
        
        # Create collection in Qdrant
        try:
            qdrant_client.create_collection(
                collection_name=documents_collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        except Exception as e:
            logger.info(f"Collection might already exist: {e}")
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=documents_collection,
            embedding=embeddings
        )
        
        # Initialize retrieval chain
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        logger.info("RAG components initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        return False

def load_document(file_path: str, file_type: str) -> List[Document]:
    """Load document based on file type."""
    try:
        if file_type.lower() == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type.lower() in ['docx', 'doc']:
            loader = Docx2txtLoader(file_path)
        elif file_type.lower() == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        raise

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available document resources in the RAG system.
    Each document is exposed as a resource with a custom doc:// URI scheme.
    """
    resources = []
    
    # Add documents from the registry
    for doc_id, metadata in document_registry.items():
        resources.append(
            types.Resource(
                uri=AnyUrl(f"doc://rag/{doc_id}"),
                name=f"Document: {metadata['name']}",
                description=f"Document of type {metadata['type']} with {metadata['chunks']} chunks",
                mimeType="text/plain",
            )
        )
    
    # Add RAG system status resource
    resources.append(
        types.Resource(
            uri=AnyUrl("system://rag/status"),
            name="RAG System Status", 
            description="Current status of the RAG system components",
            mimeType="application/json",
        )
    )
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific document's metadata or system status by its URI.
    """
    if uri.scheme == "doc":
        # Extract document ID from URI
        doc_id = uri.path.lstrip("/").split("/")[-1] if uri.path else None
        if doc_id and doc_id in document_registry:
            metadata = document_registry[doc_id]
            return f"Document: {metadata['name']}\nType: {metadata['type']}\nChunks: {metadata['chunks']}\nAdded: {metadata['added_at']}"
        raise ValueError(f"Document not found: {doc_id}")
    
    elif uri.scheme == "system":
        if "status" in str(uri):
            status = {
                "rag_initialized": vector_store is not None,
                "documents_count": len(document_registry),
                "llm_model": "gemini-2.0-flash" if llm else None,
                "embedding_model": "gemini-embedding-exp-03-07" if embeddings else None,
                "vector_store": "Qdrant (in-memory)" if qdrant_client else None
            }
            import json
            return json.dumps(status, indent=2)
    
    raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available RAG prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return [
        types.Prompt(
            name="rag-query",
            description="Query the RAG system with a question",
            arguments=[
                types.PromptArgument(
                    name="question",
                    description="The question to ask the RAG system",
                    required=True,
                ),
                types.PromptArgument(
                    name="include_sources",
                    description="Whether to include source documents (true/false)",
                    required=False,
                )
            ],
        ),
        types.Prompt(
            name="document-summary",
            description="Get a summary of all documents in the RAG system",
            arguments=[
                types.PromptArgument(
                    name="detail_level",
                    description="Level of detail (brief/detailed)",
                    required=False,
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with RAG system state.
    """
    if name == "rag-query":
        question = (arguments or {}).get("question", "")
        if not question:
            raise ValueError("Question is required for RAG query")
        
        include_sources = (arguments or {}).get("include_sources", "false").lower() == "true"
        
        return types.GetPromptResult(
            description="Query the RAG system",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Question for RAG system: {question}\nInclude sources: {include_sources}",
                    ),
                )
            ],
        )
    
    elif name == "document-summary":
        detail_level = (arguments or {}).get("detail_level", "brief")
        detail_prompt = " with extensive details" if detail_level == "detailed" else ""
        
        doc_info = []
        for doc_id, metadata in document_registry.items():
            doc_info.append(f"- {metadata['name']} ({metadata['type']}, {metadata['chunks']} chunks)")
        
        return types.GetPromptResult(
            description="Summary of documents in RAG system",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Provide a summary{detail_prompt} of the following documents in the RAG system:\n\n" +
                        "\n".join(doc_info) if doc_info else "No documents currently loaded in the RAG system.",
                    ),
                )
            ],
        )
    
    raise ValueError(f"Unknown prompt: {name}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available RAG tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="add-document",
            description="Add a document to the RAG system from content",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the document"},
                    "content": {"type": "string", "description": "Text content of the document"},
                    "type": {"type": "string", "enum": ["txt"], "default": "txt"}
                },
                "required": ["name", "content"],
            },
        ),
        types.Tool(
            name="upload-document",
            description="Upload a document file (PDF, DOCX, TXT) to the RAG system",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the document file"},
                    "name": {"type": "string", "description": "Optional custom name for the document"}
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="query-rag",
            description="Query the RAG system with a question",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question to ask the RAG system"},
                    "include_sources": {"type": "boolean", "default": False, "description": "Include source documents in response"}
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="list-documents",
            description="List all documents in the RAG system",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="remove-document",
            description="Remove a document from the RAG system",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "ID of the document to remove"}
                },
                "required": ["document_id"],
            },
        ),
        types.Tool(
            name="initialize-rag",
            description="Initialize or reinitialize the RAG system components",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle RAG tool execution requests.
    Tools can modify server state and interact with the RAG system.
    """
    if not arguments:
        arguments = {}
    
    try:
        if name == "initialize-rag":
            success = await initialize_rag_components()
            if success:
                await server.request_context.session.send_resource_list_changed()
                return [types.TextContent(type="text", text="RAG system initialized successfully!")]
            else:
                return [types.TextContent(type="text", text="Failed to initialize RAG system. Check GOOGLE_API_KEY in .env file.")]
        
        elif name == "add-document":
            if not vector_store:
                return [types.TextContent(type="text", text="RAG system not initialized. Please run initialize-rag first.")]
            
            doc_name = arguments.get("name")
            content = arguments.get("content")
            doc_type = arguments.get("type", "txt")
            
            if not doc_name or not content:
                raise ValueError("Missing name or content")
            
            # Create document and split into chunks
            document = Document(page_content=content, metadata={"source": doc_name, "type": doc_type})
            chunks = split_documents([document])
            
            # Add to vector store
            doc_ids = await asyncio.get_event_loop().run_in_executor(
                None, lambda: vector_store.add_documents(chunks)
            )
            
            # Register document
            doc_id = str(uuid.uuid4())
            document_registry[doc_id] = {
                "name": doc_name,
                "type": doc_type,
                "chunks": len(chunks),
                "added_at": str(asyncio.get_event_loop().time()),
                "vector_ids": doc_ids
            }
            
            await server.request_context.session.send_resource_list_changed()
            return [types.TextContent(type="text", text=f"Added document '{doc_name}' with {len(chunks)} chunks to RAG system.")]
        
        elif name == "upload-document":
            if not vector_store:
                return [types.TextContent(type="text", text="RAG system not initialized. Please run initialize-rag first.")]
            
            file_path = arguments.get("file_path")
            custom_name = arguments.get("name")
            
            if not file_path:
                raise ValueError("Missing file_path")
            
            if not os.path.exists(file_path):
                return [types.TextContent(type="text", text=f"File not found: {file_path}")]
            
            # Determine file type
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            if file_ext not in ['pdf', 'docx', 'doc', 'txt']:
                return [types.TextContent(type="text", text=f"Unsupported file type: {file_ext}")]
            
            # Load and process document
            try:
                documents = load_document(file_path, file_ext)
                chunks = split_documents(documents)
                
                # Add to vector store
                doc_ids = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: vector_store.add_documents(chunks)
                )
                
                # Register document
                doc_id = str(uuid.uuid4())
                doc_name = custom_name or Path(file_path).name
                document_registry[doc_id] = {
                    "name": doc_name,
                    "type": file_ext,
                    "chunks": len(chunks),
                    "added_at": str(asyncio.get_event_loop().time()),
                    "vector_ids": doc_ids,
                    "source_path": file_path
                }
                
                await server.request_context.session.send_resource_list_changed()
                return [types.TextContent(type="text", text=f"Uploaded document '{doc_name}' with {len(chunks)} chunks to RAG system.")]
                
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error processing document: {str(e)}")]
        
        elif name == "query-rag":
            if not retrieval_chain:
                return [types.TextContent(type="text", text="RAG system not initialized. Please run initialize-rag first.")]
            
            question = arguments.get("question")
            include_sources = arguments.get("include_sources", False)
            
            if not question:
                raise ValueError("Missing question")
            
            try:
                # Query the RAG system
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: retrieval_chain({"query": question})
                )
                
                response_text = result["result"]
                
                if include_sources and "source_documents" in result:
                    sources = []
                    for i, doc in enumerate(result["source_documents"]):
                        source_info = f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}"
                        sources.append(f"{source_info}\n{doc.page_content[:200]}...")
                    
                    response_text += "\n\nSources:\n" + "\n\n".join(sources)
                
                return [types.TextContent(type="text", text=response_text)]
                
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error querying RAG system: {str(e)}")]
        
        elif name == "list-documents":
            if not document_registry:
                return [types.TextContent(type="text", text="No documents in the RAG system.")]
            
            doc_list = []
            for doc_id, metadata in document_registry.items():
                doc_list.append(f"• {metadata['name']} ({metadata['type']}) - {metadata['chunks']} chunks")
            
            return [types.TextContent(type="text", text="Documents in RAG system:\n" + "\n".join(doc_list))]
        
        elif name == "remove-document":
            doc_id = arguments.get("document_id")
            if not doc_id:
                raise ValueError("Missing document_id")
            
            if doc_id not in document_registry:
                return [types.TextContent(type="text", text=f"Document not found: {doc_id}")]
            
            # Remove from registry
            doc_name = document_registry[doc_id]["name"]
            del document_registry[doc_id]
            
            await server.request_context.session.send_resource_list_changed()
            return [types.TextContent(type="text", text=f"Removed document '{doc_name}' from RAG system.")]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    # Initialize RAG components on startup
    logger.info("Starting RAG MCP Server...")
    await initialize_rag_components()
    
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-mcp-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )