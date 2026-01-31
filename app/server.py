"""
Memory MCP-CE Server

A FastMCP-based MCP server with support for:
- No authentication (if neither MCP_API_KEY nor OAUTH_BUNDLED is set)
- API Key authentication (for LobeChat)
- OAuth 2.0 authentication (for Claude Desktop)

Based on MCP Python SDK v1.24.0 patterns.
"""

import logging
import functools
from typing import Any

from pydantic import AnyHttpUrl, ValidationError
from starlette.requests import Request
from starlette.responses import Response
from starlette.exceptions import HTTPException

from app.templates import init_templates, get_static_content

from mcp.server.fastmcp import FastMCP
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions

from app.config import (
    BEARER_TOKEN,
    API_BEARER_TOKEN,
    SERVER_URL as config_server_url,
    OAUTH_BUNDLED,
    OAUTH_CLIENT_ID,
    ENCRYPTION_KEY,
)
from app.database import init_database
from app.embedding import get_embedding_dimension
from app.encryption import is_encryption_enabled
from app import tools
from app.tools import add_timezone_to_response, add_performance_to_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5005
# Use config_server_url from env if set, otherwise default to localhost
# For public deployments, set SERVER_URL to your public URL (e.g., https://memory-mcp.yourdomain.com)
SERVER_URL = config_server_url if config_server_url else f"http://localhost:{SERVER_PORT}"

# MCP scope for OAuth
MCP_SCOPE = "mcp"


def validation_error_handler(func):
    """
    Decorator to catch Pydantic ValidationError and return clean MCP Tool Execution Error.
    
    Per MCP spec, input validation errors should be Tool Execution Errors (isError: true)
    with actionable feedback, NOT Protocol Errors. This allows AI clients to self-correct
    and retry with adjusted parameters.
    
    Converts raw Pydantic tracebacks into clean JSON responses like:
    {
        "error": "Invalid parameter",
        "details": "days: Input should be a valid integer, received 0.5 (float)",
        "performance": "0.000 0.000 0.000"
    }
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            # Parse Pydantic error into clean, actionable format
            errors = e.errors()
            if errors:
                err = errors[0]
                # Get parameter name from location tuple
                loc = err.get('loc', ())
                param = loc[0] if loc else 'unknown'
                # Get the error message
                msg = err.get('msg', 'validation error')
                # Get the actual input value and its type
                input_val = err.get('input')
                input_type = type(input_val).__name__
                
                details = f"{param}: {msg}, received {input_val} ({input_type})"
            else:
                details = str(e)
            
            # Return clean error response with timezone and performance
            response = {
                "error": "Invalid parameter",
                "details": details
            }
            response = add_timezone_to_response(response)
            return add_performance_to_response(response, 0.0, 0.0, 0.0)
    
    return wrapper


def create_mcp_server() -> FastMCP:
    """
    Create and configure the FastMCP server based on authentication settings.
    
    Authentication modes:
    1. No Auth: Neither BEARER_TOKEN nor OAUTH_BUNDLED set → open access
    2. API Key Only: BEARER_TOKEN set, OAUTH_BUNDLED=false → Bearer token = API key
    3. OAuth + API Key: OAUTH_BUNDLED=true → Both methods work
    """
    
    # Determine authentication mode
    has_api_key = bool(BEARER_TOKEN)
    has_oauth = OAUTH_BUNDLED
    
    logger.info(f"🔧 Authentication configuration:")
    logger.info(f"   - API Key: {'✅ Configured' if has_api_key else '❌ Not set'}")
    logger.info(f"   - OAuth: {'✅ Enabled' if has_oauth else '❌ Disabled'}")
    
    # Build FastMCP configuration
    mcp_kwargs: dict[str, Any] = {
        "name": "memory-mcp-ce",
        "instructions": "Memory storage and retrieval MCP server with semantic search capabilities.",
        "host": SERVER_HOST,
        "port": SERVER_PORT,
        "debug": True,
        "json_response": True,  # Enable JSON responses instead of SSE streams
        "stateless_http": True,  # Enable stateless mode for LobeChat compatibility
    }
    
    # Configure authentication
    if has_oauth:
        # OAuth mode (with API key support built into the OAuth provider's load_access_token)
        from app.oauth import MemoryOAuthProvider
        
        # Create OAuth provider
        # The provider's load_access_token method handles BOTH OAuth tokens AND API keys
        oauth_provider = MemoryOAuthProvider(
            server_url=SERVER_URL,
            login_path="/login"
        )
        
        # Auth settings for bundled OAuth
        auth_settings = AuthSettings(
            issuer_url=AnyHttpUrl(SERVER_URL),
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=[MCP_SCOPE],
                default_scopes=[MCP_SCOPE],
            ),
            required_scopes=[MCP_SCOPE],
            resource_server_url=None,  # Bundled mode - no separate RS
        )
        
        mcp_kwargs["auth_server_provider"] = oauth_provider
        mcp_kwargs["auth"] = auth_settings
        # Note: FastMCP will use ProviderTokenVerifier internally which calls
        # oauth_provider.load_access_token() - this method handles both OAuth and API keys
        
        logger.info(f"🔐 OAuth enabled with bundled authorization server")
        logger.info(f"   - Issuer URL: {SERVER_URL}")
        logger.info(f"   - Client ID: {OAUTH_CLIENT_ID}")
        if has_api_key:
            logger.info(f"   - API Key: Also supported via Bearer token")
        
    elif has_api_key:
        # API Key only mode
        from app.token_verifier import HybridTokenVerifier
        
        token_verifier = HybridTokenVerifier(oauth_provider=None)
        
        auth_settings = AuthSettings(
            issuer_url=AnyHttpUrl(SERVER_URL),
            required_scopes=[MCP_SCOPE],
            resource_server_url=AnyHttpUrl(SERVER_URL),
        )
        
        mcp_kwargs["token_verifier"] = token_verifier
        mcp_kwargs["auth"] = auth_settings
        
        logger.info(f"🔑 API Key authentication enabled")
        
    else:
        # No authentication mode
        logger.info(f"🔓 No authentication configured - open access mode")
    
    # Create FastMCP instance
    mcp = FastMCP(**mcp_kwargs)
    
    # Register custom routes for OAuth login (if enabled)
    if has_oauth:
        from app.oauth import MemoryOAuthProvider
        oauth_provider = mcp_kwargs.get("auth_server_provider")
        
        if oauth_provider and isinstance(oauth_provider, MemoryOAuthProvider):
            @mcp.custom_route("/login", methods=["GET"])
            async def login_page(request: Request) -> Response:
                """Show OAuth login page."""
                state = request.query_params.get("state")
                if not state:
                    raise HTTPException(400, "Missing state parameter")
                return await oauth_provider.get_login_page(state)
            
            @mcp.custom_route("/login/callback", methods=["POST"])
            async def login_callback(request: Request) -> Response:
                """Handle OAuth login form submission."""
                return await oauth_provider.handle_login_callback(request)
            
            @mcp.custom_route("/auth/success", methods=["GET"])
            async def auth_success(request: Request) -> Response:
                """Show success page after OAuth authentication."""
                import urllib.parse
                redirect_url = request.query_params.get("redirect")
                if not redirect_url:
                    raise HTTPException(400, "Missing redirect parameter")
                # Decode the URL
                redirect_url = urllib.parse.unquote(redirect_url)
                return await oauth_provider.get_success_page(redirect_url)
            
            logger.info(f"📝 OAuth login routes registered: /login, /login/callback, /auth/success")
    
    # Register static file route for CSS/JS (always available for OAuth pages)
    @mcp.custom_route("/static/{path:path}", methods=["GET"])
    async def serve_static(request: Request) -> Response:
        """Serve static files (CSS, JS, images) for OAuth pages."""
        path = request.path_params.get("path", "")
        
        result = get_static_content(path)
        if result is None:
            raise HTTPException(404, f"Static file not found: {path}")
        
        content, mime_type = result
        return Response(content=content, media_type=mime_type)
    
    logger.info(f"📁 Static file route registered: /static/*")
    
    # Register API routes (if API_BEARER_TOKEN is set)
    register_api_routes(mcp)
    
    # Register tools
    register_tools(mcp)
    
    return mcp


def register_api_routes(mcp: FastMCP) -> None:
    """
    Register REST API routes (separate from MCP endpoints).
    
    These routes are protected by API_BEARER_TOKEN (different from BEARER_TOKEN for MCP).
    If API_BEARER_TOKEN is not set, routes return 404 as if they don't exist.
    """
    from starlette.responses import JSONResponse
    from app.api.embeddings import generate_embeddings_handler
    
    @mcp.custom_route("/api/embeddings/generate", methods=["POST"])
    async def api_generate_embeddings(request: Request) -> Response:
        """
        POST /api/embeddings/generate
        
        Re-embed memories with a new embedding model in the background.
        Returns 202 Accepted immediately while processing continues in background.
        """
        # If API_BEARER_TOKEN not set, return 404 (API disabled)
        if not API_BEARER_TOKEN:
            raise HTTPException(404, "Not Found")
        
        # Validate Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(401, "Missing Authorization header")
        
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Invalid Authorization header format")
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        if token != API_BEARER_TOKEN:
            raise HTTPException(401, "Invalid API token")
        
        # Parse request body
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON body")
        
        # Process request
        try:
            result = await generate_embeddings_handler(body)
            return JSONResponse(content=result, status_code=202)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.error(f"❌ API error: {str(e)}", exc_info=True)
            raise HTTPException(500, f"Internal server error: {str(e)}")
    
    # Log API status
    if API_BEARER_TOKEN:
        logger.info(f"🔌 REST API enabled: POST /api/embeddings/generate")
    else:
        logger.info(f"🔒 REST API disabled (API_BEARER_TOKEN not set)")


def register_tools(mcp: FastMCP) -> None:
    """Register all MCP tools with validation error handling."""
    
    @mcp.tool(
        annotations={
            "title": "Store new memory",
            "readOnlyHint": False,
            "openWorldHint": False,
            "destructiveHint": False,
            "idempotentHint": False
        }
    )
    @validation_error_handler
    async def store_memory(
        content: str,
        labels: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """
        Store a conversation memory for later retrieval.
        
        Args:
            content: The conversation content or summary to remember
            labels: Optional comma-separated labels for categorizing the memory
            source: Optional source attribution (e.g., 'Wikipedia', 'user')
            
        Returns:
            Result with success status and memory ID
        """
        return tools.store_memory(content, labels, source)
    
    @mcp.tool(
        annotations={
            "title": "Retrieve stored memories",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    )
    @validation_error_handler
    async def retrieve_memories(
        query: str | None = None,
        labels: str | None = None,
        source: str | None = None,
        num_results: int = 5,
    ) -> dict[str, Any]:
        """
        Retrieve memories with flexible filtering. All parameters are optional.
        
        Filtering modes:
        - Query only: Semantic search using embeddings
        - Labels only: Filter by labels, return most recent
        - Source only: Filter by source, return most recent
        - Query + Labels: Semantic search filtered by labels
        - Query + Source: Semantic search filtered by source
        - Labels + Source: Filter by both, return most recent
        - Query + Labels + Source: Semantic search with all filters
        - No parameters: Return most recent N memories
        
        Args:
            query: Optional query text for semantic search (uses embeddings)
            labels: Optional comma-separated labels for filtering (fuzzy match). Use ! prefix to exclude (e.g., 'beer,!wine')
            source: Optional source filter (fuzzy match). Use ! prefix to exclude (e.g., '!clawdbot')
            num_results: Number of results to return (default: 5)
            
        Returns:
            List of matching memories (with similarity scores if query provided)
        """
        return tools.retrieve_memories(query, labels, source, num_results)
    
    @mcp.tool(
        annotations={
            "title": "Add labels to a memory",
            "readOnlyHint": False,
            "openWorldHint": False,
            "destructiveHint": False,
            "idempotentHint": True
        }
    )
    @validation_error_handler
    async def add_labels(
        memory_id: int,
        labels: str,
    ) -> dict[str, Any]:
        """
        Add labels to an existing memory without replacing existing ones.
        
        Args:
            memory_id: The unique ID of the memory
            labels: Labels to add (comma-separated string or JSON array)
            
        Returns:
            Result with success status and updated labels
        """
        return tools.add_labels(memory_id, labels)
    
    @mcp.tool(
        annotations={
            "title": "Remove labels from a memory",
            "readOnlyHint": False,
            "openWorldHint": False,
            "destructiveHint": True,
            "idempotentHint": True
        }
    )
    @validation_error_handler
    async def del_labels(
        memory_id: int,
        labels: str,
    ) -> dict[str, Any]:
        """
        Remove specific labels from an existing memory.
        
        Args:
            memory_id: The unique ID of the memory
            labels: Labels to remove (comma-separated string or JSON array)
            
        Returns:
            Result with success status and updated labels
        """
        return tools.del_labels(memory_id, labels)
    
    @mcp.tool(
        annotations={
            "title": "Delete memory by ID",
            "readOnlyHint": False,
            "openWorldHint": False,
            "destructiveHint": True,
            "idempotentHint": True
        }
    )
    @validation_error_handler
    async def delete_memory(
        memory_id: int,
    ) -> dict[str, Any]:
        """
        Delete a specific memory by its ID.
        
        Args:
            memory_id: The unique ID of the memory to delete
            
        Returns:
            Result with success status
        """
        return tools.delete_memory(memory_id)
    
    @mcp.tool(
        annotations={
            "title": "Get specific memory by ID",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    )
    @validation_error_handler
    async def get_memory(
        memory_id: int,
    ) -> dict[str, Any]:
        """
        Get a specific memory by its ID.
        
        Args:
            memory_id: The unique ID of the memory to retrieve
            
        Returns:
            The full memory object with all metadata
        """
        return tools.get_memory(memory_id)
    
    @mcp.tool(
        annotations={
            "title": "Get a random memory",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    )
    @validation_error_handler
    async def random_memory(
        labels: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve a random memory from the namespace.
        
        Args:
            labels: Optional comma-separated labels for filtering (fuzzy match). Use ! prefix to exclude (e.g., 'beer,!wine')
            source: Optional source filter (fuzzy match). Use ! prefix to exclude (e.g., '!clawdbot')
            
        Returns:
            A random memory matching the filters
        """
        return tools.random_memory(labels, source)
    
    @mcp.tool(
        annotations={
            "title": "Get memory statistics",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    )
    @validation_error_handler
    async def memory_stats(
        labels: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """
        Return memory statistics for the configured namespace(s).
        
        Three modes:
        - No parameters: Return total memory count
        - labels: Count memories with matching labels (fuzzy match)
        - source: Count memories from matching source (fuzzy match)
        
        Args:
            labels: Optional filter for labels (fuzzy match, comma-separated). Use ! prefix to exclude (e.g., 'beer,!wine')
            source: Optional source filter (fuzzy match). Use ! prefix to exclude (e.g., '!clawdbot')
            
        Returns:
            Statistics including total count, matching count, percentage,
            and list of matched labels/sources (labels_matched, sources_matched)
        """
        return tools.memory_stats(labels, source)
    
    @mcp.tool(
        annotations={
            "title": "Get trending labels",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    )
    @validation_error_handler
    async def trending_labels(
        days: int = 30,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Get currently trending labels based on token activity and synaptic decay model.
        
        Uses a two-stage algorithm:
        1. Find hot tokens from recent label activity with decay calculation
        2. Match tokens to actual labels in current memories
        
        The synaptic decay model means heavily-used topics stay relevant longer,
        while rarely-used tokens fade quickly (mimics neural pathway strengthening).
        
        Args:
            days: Time window for considering tokens (hard cutoff, default: 30)
            limit: Maximum number of trending labels to return (default: 10)
            
        Returns:
            List of trending labels with counts and the top matching token
        """
        return tools.trending_labels(days, limit)
    
    logger.info(f"🛠️ Registered 9 tools: store_memory, retrieve_memories, add_labels, del_labels, delete_memory, get_memory, random_memory, memory_stats, trending_labels")


def main():
    """Main entry point for the server."""
    try:
        # Initialize template system (for OAuth pages)
        init_templates()
        
        # Initialize database
        logger.info("🔍 Detecting embedding dimension...")
        embedding_dim = get_embedding_dimension()
        logger.info(f"📐 Embedding dimension: {embedding_dim}")
        
        logger.info("🗄️ Initializing database...")
        init_database(embedding_dim)
        logger.info("✅ Database initialized")
        
        # Log encryption status
        if is_encryption_enabled():
            logger.info("🔐 Encryption: ✅ ENABLED (AES-256-GCM with Argon2id)")
        else:
            logger.info("🔓 Encryption: ❌ DISABLED (set ENCRYPTION_KEY to enable)")
        
        # Create and run server
        mcp = create_mcp_server()
        
        logger.info(f"🚀 Starting Memory MCP-CE server on {SERVER_HOST}:{SERVER_PORT}")
        logger.info(f"📡 MCP endpoint: http://{SERVER_HOST}:{SERVER_PORT}/mcp")
        
        # Run with streamable-http transport
        mcp.run(transport="streamable-http")
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
