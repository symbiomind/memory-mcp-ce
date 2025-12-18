"""
Hybrid Token Verifier for Memory MCP-CE.

Supports both:
1. API Key authentication (for LobeChat)
2. OAuth token authentication (for Claude Desktop)
"""

import logging
from typing import TYPE_CHECKING

from mcp.server.auth.provider import AccessToken, TokenVerifier

from app.config import BEARER_TOKEN, OAUTH_BUNDLED

if TYPE_CHECKING:
    from app.oauth import MemoryOAuthProvider

logger = logging.getLogger(__name__)


class HybridTokenVerifier(TokenVerifier):
    """
    Token verifier that supports both API keys and OAuth tokens.
    
    Authentication priority:
    1. If token matches BEARER_TOKEN → authenticated as API key user
    2. If OAUTH_BUNDLED is enabled → check OAuth token store
    3. Otherwise → reject
    """
    
    def __init__(self, oauth_provider: "MemoryOAuthProvider | None" = None):
        self.oauth_provider = oauth_provider
    
    async def verify_token(self, token: str) -> AccessToken | None:
        """
        Verify a bearer token.
        
        Args:
            token: The bearer token to verify
            
        Returns:
            AccessToken if valid, None if invalid
        """
        # 1. Check API key first (simple string comparison)
        if BEARER_TOKEN and token == BEARER_TOKEN:
            logger.info("✅ Authenticated via API key")
            return AccessToken(
                token=token,
                client_id="api_key_client",
                scopes=["mcp"],
                expires_at=None,  # API keys don't expire
            )
        
        # 2. Check OAuth token if enabled
        if OAUTH_BUNDLED and self.oauth_provider:
            access_token = await self.oauth_provider.load_access_token(token)
            if access_token:
                logger.info(f"✅ Authenticated via OAuth token for client: {access_token.client_id}")
                return access_token
        
        # 3. No valid authentication found
        logger.debug("❌ Token verification failed")
        return None


class NoAuthTokenVerifier(TokenVerifier):
    """
    Token verifier that allows all requests (no authentication).
    Used when neither BEARER_TOKEN nor OAUTH_BUNDLED is configured.
    """
    
    async def verify_token(self, token: str) -> AccessToken | None:
        """Always returns a valid access token (no auth mode)."""
        logger.debug("🔓 No auth mode - allowing request")
        return AccessToken(
            token=token or "anonymous",
            client_id="anonymous",
            scopes=["mcp"],
            expires_at=None,
        )
