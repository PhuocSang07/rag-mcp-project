# RAG MCP Server

A powerful Retrieval-Augmented Generation (RAG) server built with the Model Context Protocol (MCP), featuring document ingestion, vector storage, and AI-powered question answering capabilities.

## 🚀 Features

- **Document Processing**: Support for PDF, DOCX, and TXT files
- **Vector Storage**: Qdrant vector database for efficient similarity search
- **AI Integration**: Google Gemini for embeddings and text generation  
- **MCP Protocol**: Full MCP compliance for seamless integration with AI assistants
- **Async Architecture**: High-performance async/await implementation

## 🏗️ Architecture

### Core Components

- **LangChain**: Document processing and retrieval chains
- **Qdrant**: Vector database for document embeddings
- **Google Gemini**: LLM for embeddings (`embedding-001`) and generation (`gemini-pro`)
- **MCP Server**: Protocol-compliant server for AI assistant integration

### Document Processing Pipeline

1. **Ingestion**: Upload documents via file path or direct content
2. **Chunking**: Split documents using RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
3. **Embedding**: Generate 768-dimensional vectors using Google's embedding model
4. **Storage**: Store in Qdrant with metadata tracking
5. **Retrieval**: Query with similarity search (top-3 results)
6. **Generation**: Answer questions using retrieved context

## 📋 Prerequisites

- Python 3.12+
- Google Gemini API key
- UV package manager (recommended) or pip

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rag-mcp
   ```

2. **Install dependencies:**
   ```bash
   # With UV (recommended)
   uv sync --dev --all-extras
   
   # Or with pip
   pip install -e .
   ```

3. **Set up your API key:**
   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your-gemini-api-key
   ```

## 🚀 Usage

### Running the Server

```bash
# With UV
uv run rag-mcp-server

# Or directly
python -m rag_mcp_server
```

### Configuration for Claude Desktop

Add to your Claude desktop configuration:

```json
{
  "mcpServers": {
    "rag-mcp-server": {
      "command": "uv",
      "args": ["run", "rag-mcp-server"]
    }
  }
}
```

## 🛠️ MCP Tools

### Core Tools

#### `initialize-rag`
Initialize the RAG system components (Qdrant, embeddings, LLM).
```
No parameters required
```

#### `add-document`
Add text content as a document to the RAG system.
```json
{
  "name": "Document name",
  "content": "Document text content", 
  "type": "txt"
}
```

#### `upload-document`
Upload a document file (PDF, DOCX, TXT) to the RAG system.
```json
{
  "file_path": "/path/to/document.pdf",
  "name": "Optional custom name"
}
```

#### `query-rag`
Ask questions to the RAG system.
```json
{
  "question": "Your question here",
  "include_sources": false
}
```

#### `list-documents`
List all documents in the RAG system.
```
No parameters required
```

#### `remove-document`
Remove a document from the RAG system.
```json
{
  "document_id": "document-uuid"
}
```

## 📚 MCP Resources

- `doc://rag/{id}`: Individual document metadata and information
- `system://rag/status`: Current status of RAG system components

## 💬 MCP Prompts

#### `rag-query`
Query the RAG system with a question.
- `question` (required): The question to ask
- `include_sources` (optional): Include source documents in response

#### `document-summary`
Get a summary of all documents in the RAG system.
- `detail_level` (optional): "brief" or "detailed"

## Configuration

[TODO: Add configuration details specific to your implementation]

## Quickstart

### Install

#### Claude Desktop

On MacOS: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

<details>
  <summary>Development/Unpublished Servers Configuration</summary>
  ```
  "mcpServers": {
    "rag-mcp-server": {
      "command": "uv",
      "args": [
        "--directory",
        "D:\Github\rag-mcp",
        "run",
        "rag-mcp-server"
      ]
    }
  }
  ```
</details>

<details>
  <summary>Published Servers Configuration</summary>
  ```
  "mcpServers": {
    "rag-mcp-server": {
      "command": "uvx",
      "args": [
        "rag-mcp-server"
      ]
    }
  }
  ```
</details>

## Development

### Building and Publishing

To prepare the package for distribution:

1. Sync dependencies and update lockfile:
```bash
uv sync
```

2. Build package distributions:
```bash
uv build
```

This will create source and wheel distributions in the `dist/` directory.

3. Publish to PyPI:
```bash
uv publish
```

Note: You'll need to set PyPI credentials via environment variables or command flags:
- Token: `--token` or `UV_PUBLISH_TOKEN`
- Or username/password: `--username`/`UV_PUBLISH_USERNAME` and `--password`/`UV_PUBLISH_PASSWORD`

### Debugging

Since MCP servers run over stdio, debugging can be challenging. For the best debugging
experience, we strongly recommend using the [MCP Inspector](https://github.com/modelcontextprotocol/inspector).


You can launch the MCP Inspector via [`npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) with this command:

```bash
npx @modelcontextprotocol/inspector uv --directory D:\Github\rag-mcp run rag-mcp-server
```


Upon launching, the Inspector will display a URL that you can access in your browser to begin debugging.