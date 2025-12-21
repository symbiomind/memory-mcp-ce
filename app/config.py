
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "memory")

# Embedding Configuration
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

# MCP Configuration
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
NAMESPACE = os.getenv("NAMESPACE")

# Encryption Configuration
# If set and non-empty, all new memories will be encrypted with AES-256-GCM
# If empty or not set, memories are stored as plain UTF-8 (enc=false)
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

# OAuth Configuration
OAUTH_BUNDLED = os.getenv("OAUTH_BUNDLED", "false").lower() == "true"
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "memory-mcp-ce")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "")
OAUTH_USERNAME = os.getenv("OAUTH_USERNAME", "")
OAUTH_PASSWORD = os.getenv("OAUTH_PASSWORD", "")

# Server URL Configuration (for public-facing deployments)
# This should be the public URL that clients will use to connect
# e.g., "https://memory-mcp.yourdomain.com" for Cloudflare tunnel
SERVER_URL = os.getenv("SERVER_URL", "")

# OAuth Token Expiry Configuration (in seconds)
# Access tokens - default 1 hour (3600 seconds)
OAUTH_ACCESS_TOKEN_EXPIRY = int(os.getenv("OAUTH_ACCESS_TOKEN_EXPIRY", 3600))
# Refresh tokens - default 7 days (604800 seconds)
OAUTH_REFRESH_TOKEN_EXPIRY = int(os.getenv("OAUTH_REFRESH_TOKEN_EXPIRY", 604800))
# Authorization codes - default 5 minutes (300 seconds)
OAUTH_AUTH_CODE_EXPIRY = int(os.getenv("OAUTH_AUTH_CODE_EXPIRY", 300))

# OAuth Redirect URIs (comma-separated list of exact URIs per MCP spec)
# Default includes Claude.ai and localhost for backward compatibility
OAUTH_REDIRECT_URIS = os.getenv(
    "OAUTH_REDIRECT_URIS",
    "https://claude.ai/api/mcp/auth_callback,http://localhost/callback,http://127.0.0.1/callback"
)
