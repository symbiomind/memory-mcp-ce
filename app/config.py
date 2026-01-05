
import os
import logging
from dotenv import load_dotenv
from zoneinfo import ZoneInfo, available_timezones

# Load environment variables from .env file
load_dotenv()

# Get logger for config module
config_logger = logging.getLogger(__name__)

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

# Optional: Request specific embedding dimensions (for MRL models like Qwen)
# If not set, auto-detects from model's native output
# If set, validates model returns exactly this many dimensions at startup
_embedding_dims_raw = os.getenv("EMBEDDING_DIMS", "").strip()
EMBEDDING_DIMS = int(_embedding_dims_raw) if _embedding_dims_raw else None

# MCP Configuration
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
NAMESPACE = os.getenv("NAMESPACE")

# API Configuration (separate from MCP authentication)
# If API_BEARER_TOKEN is set, /api/* endpoints are enabled
# If not set, /api/* endpoints return 404
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")

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

# Timezone Configuration
# Options:
#   - Valid timezone string (e.g., Australia/Adelaide, America/New_York, Europe/London)
#   - UTC (default if not set or invalid)
#   - false (disables timezone feature entirely - no current_time or timezone fields in responses)
def _parse_timezone_config():
    """Parse and validate TIMEZONE configuration."""
    tz_value = os.getenv("TIMEZONE", "").strip()
    
    # Check for "false" (case-insensitive) - disable feature entirely
    if tz_value.lower() == "false":
        config_logger.info("⏰ Timezone feature disabled (TIMEZONE=false)")
        return None, True  # None timezone, feature disabled
    
    # Empty string or not set - default to UTC
    if not tz_value:
        config_logger.info("⏰ Timezone defaulting to UTC (TIMEZONE not set)")
        return ZoneInfo("UTC"), False
    
    # Try to parse as valid timezone
    if tz_value in available_timezones():
        config_logger.info(f"⏰ Timezone configured: {tz_value}")
        return ZoneInfo(tz_value), False
    
    # Invalid timezone - warn and fall back to UTC
    config_logger.warning(f"⚠️ Invalid timezone '{tz_value}', falling back to UTC")
    return ZoneInfo("UTC"), False

# Parse timezone on startup
TIMEZONE, TIMEZONE_DISABLED = _parse_timezone_config()

# Performance Metrics Configuration
# When enabled, all tool responses include timing data
# Format: "embedding_time db_time total_time" (seconds with 3 decimal places)
PERFORMANCE_METRICS = os.getenv("PERFORMANCE_METRICS", "false").lower() == "true"
