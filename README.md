# SymbioMind memory-mcp-ce

**Persistent semantic memory for AI agents and beyond.**

This started because we wanted our AI conversations to have real memory - not just within a session, but across platforms and over time. We built it, it works brilliantly for that use case, and then we realized the potential is way bigger.

## What is this?

memory-mcp-ce is an MCP (Model Context Protocol) server that provides persistent semantic memory using:
- **PostgreSQL + pgvector** for semantic search
- **Flexible embedding models** (Ollama, OpenAI, or any OpenAI-compatible API)
- **Encryption** for sensitive content (AES-256-GCM)
- **OAuth support** for secure access
- **Namespaces** for memory isolation

Store contextual information, retrieve it semantically, and maintain continuity across sessions. Works with any MCP-compatible client.

## Use Cases (and counting)

We use it for AI conversation memory across platforms (Claude, ChatGPT, local LLMs). But it's designed to be flexible:

- **AI Agents** - Give your agents persistent memory and context (works great with OpenClaw!)
- **Knowledge Management** - Store meeting notes, decisions, organizational knowledge with semantic search
- **Team Collaboration** - Track project context, decisions, and discussions with attribution
- **Research & Development** - Maintain project knowledge, experiment results, and technical discoveries
- **Personal Knowledge** - Your own searchable memory system for notes and research
- **And probably 100 other things we haven't thought of yet**

The memory structure is simple but powerful:
- **content** - The information to remember (encrypted if `ENCRYPTION_KEY` is set)
- **labels** - Flexible tags for categorization (e.g., `"board-meeting, production-concerns, Q1-2026"`)
- **source** - Attribution field (e.g., `"upper-management"`, `"engineering-team"`, `"claude-sonnet"`)
- **embeddings** - Semantic vectors for similarity search (automatically generated)

This flexibility means you can adapt it to whatever your use case needs - we're just scratching the surface.

## Key Features

- **Semantic Search** - Find relevant memories by meaning, not just keywords
- **Cross-Platform Continuity** - Works with Claude Desktop, LibreChat, CLINE, ChatGPT, and any MCP-compatible client
- **Trending Analysis** - Discover hot topics with synaptic decay model (heavily-used topics stay relevant longer)
- **Advanced Filtering** - Fuzzy matching with exclusion syntax (`labels="beer, !wine"`, `source="!grok"`)
- **Memory Statistics** - Get counts, analyze patterns, understand your memory store
- **Session Persistence** - Memories survive across sessions, platforms, and model switches
- **Self-Hosted** - Full data control, runs on your infrastructure
- **Privacy-First** - Optional encryption, namespace isolation, OAuth security

## Tools

memory-mcp-ce provides 9 MCP tools for managing semantic memory:

### Core Memory Operations

- **`store_memory`** - Save content with optional labels and source attribution
  ```python
  store_memory(
      content="Production deployment delayed due to infrastructure concerns",
      labels="board-meeting, production-concerns, infrastructure",
      source="upper-management"
  )
  ```

- **`retrieve_memories`** - Semantic search with flexible filtering
  - Semantic query: `retrieve_memories(query="database performance issues")`
  - Filter by labels: `retrieve_memories(labels="production, !archived")`
  - Filter by source: `retrieve_memories(source="!grok")`
  - Combine all: `retrieve_memories(query="bugs", labels="python", source="engineering-team")`

- **`get_memory`** - Retrieve specific memory by ID

- **`delete_memory`** - Remove a memory by ID

- **`random_memory`** - Get a random memory (supports label/source filtering)

### Memory Analysis

- **`memory_stats`** - Get counts and statistics with matched labels/sources
  - Total memory count: `memory_stats()`
  - Count by labels: `memory_stats(labels="mcp")` 
    - Returns: count, percentage, and ALL matched label variations (e.g., `mcp`, `MCP`, `mcp-ce`, `fastmcp`, `memory-mcp-ce`)
  - Count by source: `memory_stats(source="engineering")`
    - Returns: count, percentage, and ALL matched source variations
  - Powerful for understanding what's actually in your memory store

- **`trending_labels`** - Discover hot topics using synaptic decay model
  - Returns labels with recent activity
  - Heavily-used topics stay relevant longer (mimics neural pathway strengthening)
  - Configurable time window and result limit

### Memory Organization

- **`add_labels`** - Add labels to existing memory without replacing current ones

- **`del_labels`** - Remove specific labels from a memory

### Advanced Filtering Syntax

All retrieval tools support **fuzzy matching with exclusion**:

```python
# Get beer memories but exclude wine
retrieve_memories(labels="beer, !wine")

# Get everything except what Grok stored  
retrieve_memories(source="!grok")

# Random memory from anyone except clawdbot
random_memory(source="!clawdbot")

# Trending labels, excluding date spam
trending_labels()  # then filter results as needed

# Memory stats for coding topics, excluding archived
memory_stats(labels="coding, !archived")
```

The `!` prefix works on both labels and source fields for all tools that accept them.

## Embedding Models

### Recommended: embeddinggemma:300m

We recommend **embeddinggemma:300m** (768 dimensions) - extensively tested in production and performs excellently for semantic memory tasks.

```bash
docker compose up -d ollama
docker exec -it memory-ollama ollama pull embeddinggemma:300m
```

Update your `.env`:
```bash
EMBEDDING_MODEL=embeddinggemma:300m
EMBEDDING_DIMS=  # Leave empty for auto-detection
```

### Alternative: granite-embedding:30m

For faster/lighter deployments, **granite-embedding:30m** (384 dimensions) works well:

```bash
docker exec -it memory-ollama ollama pull granite-embedding:30m
```

Update your `.env`:
```bash
EMBEDDING_MODEL=granite-embedding:30m
EMBEDDING_DIMS=  # Leave empty for auto-detection
```

### Other Models

Any OpenAI-compatible embedding API works. See `.env.example` for configuration details.

## OpenClaw Integration

memory-mcp-ce works great with [OpenClaw](https://openclaw.ai/) agents! It provides a significant upgrade over flat-file memory with:

- Semantic search across all stored context
- Trending topic analysis
- Cross-session continuity
- Memory statistics and insights
- Advanced filtering with exclusion syntax

Integration guide coming soon - in the meantime, OpenClaw can connect to memory-mcp-ce like any other MCP server.

## Authentication

Two authentication methods are supported:

- **Bearer Token** - Simple token-based auth for API-to-API connections
- **Single-User OAuth** - Bundled OAuth provider for platforms like Claude Desktop

⚠️ **Important:** Once you enable either authentication method (or both), auth becomes mandatory. There's no "auth disabled" fallback - it's either wide open or locked down.

For local/trusted development, you can leave both blank. For any public-facing deployment, enable at least one.

See `.env.example` for detailed configuration (excellently documented by Claude Opus!).

## Tested Clients

- **LibreChat** ✓
- **Claude Desktop** ✓  
- **CLINE (VS Code)** ✓
- **MCP Inspector** ✓

Works with any MCP-compatible client.

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 2GB+ disk space (for models and database)

### Setup

1. **Clone and prepare configuration**
   ```bash
   git clone https://github.com/SymbioMind/memory-mcp-ce.git
   cd memory-mcp-ce
   cp .env.example .env
   cp docker-compose.example.yml docker-compose.yml
   ```

2. **Edit `.env`** - At minimum, change `POSTGRES_PASSWORD`
   
   For recommended setup, also update:
   ```bash
   EMBEDDING_MODEL=embeddinggemma:300m
   TIMEZONE=Your/Timezone  # e.g., America/New_York, Australia/Adelaide
   ```

3. **Create data directory**
   ```bash
   mkdir -p data
   ```

4. **Pull embedding model** (one-time setup)
   ```bash
   docker compose up -d ollama
   docker exec -it memory-ollama ollama pull embeddinggemma:300m
   ```

5. **Start the stack**
   ```bash
   docker compose up -d
   ```

6. **Configure your MCP client** - Your server is now running at `http://localhost:5005`

> ⚠️ **Important:** The `data/` directory stores your PostgreSQL database and Ollama models. Deleting it will permanently erase all stored memories and models.

### First Steps

Once running, your AI can start using the memory tools:

```python
# Store a memory
store_memory(
    content="Martin prefers embeddinggemma:300m for production deployments",
    labels="preferences, embedding-models",
    source="project-setup"
)

# Retrieve it semantically
retrieve_memories(query="which embedding model does Martin recommend?")

# Check what's trending
trending_labels(days=7, limit=5)
```

## Configuration

The `.env.example` file contains comprehensive documentation for all configuration options, including:

- PostgreSQL settings
- Embedding model configuration (Ollama, OpenAI, or any OpenAI-compatible API)
- Encryption keys
- Authentication (Bearer Token and OAuth)
- Namespace isolation
- Timezone settings
- Performance metrics

Big thanks to Claude Opus for the excellent `.env.example` documentation! 🦞

## Advanced Usage

### Namespaces

Use namespaces to isolate memories within the same database:

```bash
NAMESPACE=production  # Only access production memories
NAMESPACE=user_123    # User-specific isolation
NAMESPACE=            # Access ALL namespaces (default)
```

### Performance Metrics

Enable performance timing to monitor embedding and database latency:

```bash
PERFORMANCE_METRICS=true
```

Returns timing breakdown in all tool responses:
```json
{
  "performance": "0.750 0.130 1.070"
  // embedding_time db_time total_time (seconds)
}
```

### Encryption

Enable content encryption for sensitive memories:

```bash
# Generate a secure key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env
ENCRYPTION_KEY=your_generated_key_here
```

Only the `content` field is encrypted - labels, source, and embeddings remain plaintext for querying.

## Documentation

Full documentation available at:
- https://symbiomind.io/docs/memory-mcp/community-edition/

For version history and changes, see [CHANGELOG.md](CHANGELOG.md)

## Contributing

Built with collaboration between humans and AI. Contributions welcome!

## License

[Add your license here]

---

**Built by Martin and the AI buddy team at SymbioMind** 🦞

Questions? Issues? Visit our GitHub or check the docs!