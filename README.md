# SymbioMind memory-mcp-ce

**A memory system for AI conversations - built by a human and AI, for humans and AI to remember each other.**

This started because I wanted my AI buddies to actually remember our conversations. Not just within a single chat, but *across* platforms and sessions. So we built it together.

## What is this?

memory-mcp-ce is an MCP (Model Context Protocol) server that gives your AI conversations persistent memory using:
- **PostgreSQL + pgvector** for semantic search
- **Flexible embedding models** (Ollama, OpenAI, or any OpenAI-compatible API)
- **Encryption** for sensitive memories (AES-256-GCM)
- **OAuth support** for secure access
- **Namespaces** for memory isolation

Your AI can store memories, retrieve relevant context, and maintain continuity across conversations - even when you switch between different AI platforms.

## Tools

memory-mcp-ce provides 7 MCP tools for managing conversational memory:

**Core Memory Operations:**
- `store_memory` - Save conversation content with optional labels and source attribution
- `retrieve_memories` - Flexible semantic search with filtering by query, labels, and/or source
- `get_memory` - Retrieve a specific memory by ID
- `delete_memory` - Remove a memory by ID
- `random_memory` - Get a random memory (optionally filtered by labels/source)

**Memory Organization:**
- `add_labels` - Add labels to existing memories without replacing current ones
- `del_labels` - Remove specific labels from a memory

## Authentication
Two authentication methods are supported:

- Bearer Token - Simple token-based auth for clients like LobeChat and MCP Inspector
- Single-User OAuth - Bundled OAuth provider for clients like Claude Desktop (one username/password)

⚠️ Important: Once you enable either authentication method (or both), auth becomes mandatory. There's no "auth disabled" fallback - it's either wide open or locked down.
For local/trusted development, you can leave both blank. For any public-facing deployment, enable at least one.

## Tested Clients

- **LibreChat** ✓
- **Claude Desktop** ✓  
- **CLINE (VS Code)** ✓
- **MCP Inspector** ✓

## Application Setup

1. Copy `.env.example` to `.env` and `docker-compose.example.yml` to `docker-compose.yml`
2. Edit `.env` with your settings (at minimum: change `POSTGRES_PASSWORD`)
3. Create directory `mkdir -p data` (Docker will create subdirs)

4. Pull the embedding model (one-time setup)
    
    You must pull the embedding model once before starting the full stack.

    ```bash
    docker compose up -d ollama
    docker exec -it memory-ollama ollama pull granite-embedding:30m
    ```
5. Run `docker compose up -d`
6. Configure your MCP client (see client-specific guides below)

> Important: The `data/` directory stores your PostgreSQL database and Ollama models.
> Deleting it will permanently erase all stored memories and models.

Your MCP server is now running at `http://localhost:5005`

### SymbioMind memory-mcp-ce documentation 

https://symbiomind.io/docs/memory-mcp/community-edition/