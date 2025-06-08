# RAG MCP Server - Quick Start Guide

## 🎯 What You Have

A fully functional RAG (Retrieval-Augmented Generation) MCP Server with:

- ✅ **Document Processing**: PDF, DOCX, TXT support
- ✅ **Vector Storage**: Qdrant in-memory database
- ✅ **AI Integration**: Google Gemini for embeddings and generation
- ✅ **MCP Protocol**: Full compliance for AI assistant integration
- ✅ **Async Architecture**: High-performance implementation

## 🚀 Quick Setup

### 1. Get Your API Key
Get your Google Gemini API key from: https://ai.google.dev/

### 2. Create .env File
Create a `.env` file in the project root with your API key:
```bash
GOOGLE_API_KEY=your-api-key-here
```

### 3. Run the Server
```bash
uv run rag-mcp-server
```

## 🔧 Claude Desktop Integration

Add to your Claude desktop configuration file:

**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-mcp-server": {
      "command": "uv",
      "args": ["--directory", "D:\\Github\\rag-mcp", "run", "rag-mcp-server"]
    }
  }
}
```

## 📋 Available Tools

Once connected to Claude, you can use these tools:

### 1. **initialize-rag** - Start the RAG system
```
Initialize RAG system components
```

### 2. **add-document** - Add text content
```json
{
  "name": "My Document",
  "content": "Your text content here..."
}
```

### 3. **upload-document** - Upload files
```json
{
  "file_path": "C:\\path\\to\\document.pdf",
  "name": "Optional custom name"
}
```

### 4. **query-rag** - Ask questions
```json
{
  "question": "What is the main topic of the documents?",
  "include_sources": true
}
```

### 5. **list-documents** - See all documents
```
List all stored documents
```

### 6. **remove-document** - Delete documents
```json
{
  "document_id": "document-uuid"
}
```

## 🔍 Resources Available

- `doc://rag/{id}` - Individual document metadata
- `system://rag/status` - RAG system status

## 💬 Prompts Available

- **rag-query** - Query with questions
- **document-summary** - Get document summaries

## 🎯 Example Workflow

1. **Initialize**: Use `initialize-rag` tool
2. **Add Documents**: Use `upload-document` or `add-document`
3. **Query**: Use `query-rag` to ask questions
4. **Manage**: Use `list-documents` and `remove-document` as needed

## 🐛 Troubleshooting

### Server Won't Start
- Check if `GEMINI_API_KEY` is set
- Run `uv sync` to ensure dependencies are installed

### No API Key Warning
- The server will start but with limited functionality
- Make sure your `.env` file contains `GOOGLE_API_KEY=your-api-key-here`

### Connection Issues
- Ensure the path in Claude config matches your project location
- Restart Claude Desktop after config changes

## 🎉 You're Ready!

Your RAG MCP Server is now ready to use with Claude or any MCP-compatible client!
