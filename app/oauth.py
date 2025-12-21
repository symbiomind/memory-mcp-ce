"""
OAuth 2.0 Authorization Server Provider for Memory MCP-CE.

Implements the MCP SDK's OAuthAuthorizationServerProvider protocol
for bundled OAuth support (combined AS + RS in one server).

Based on the MCP Python SDK v1.24.0 simple-auth example.
"""

import logging
import secrets
import time
from typing import Any

from pydantic import AnyUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from app.config import (
    BEARER_TOKEN,
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    OAUTH_USERNAME,
    OAUTH_PASSWORD,
    OAUTH_ACCESS_TOKEN_EXPIRY,
    OAUTH_REFRESH_TOKEN_EXPIRY,
    OAUTH_AUTH_CODE_EXPIRY,
    OAUTH_REDIRECT_URIS,
)

logger = logging.getLogger(__name__)


class MemoryOAuthProvider(OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]):
    """
    OAuth Authorization Server Provider for Memory MCP-CE.
    
    This provider handles the OAuth flow by:
    1. Providing a simple login form for credential authentication
    2. Issuing MCP tokens after successful authentication
    3. Maintaining token state for validation
    
    Supports both bundled mode (AS + RS together) and can work
    alongside API key authentication.
    """
    
    def __init__(self, server_url: str, login_path: str = "/login"):
        """
        Initialize the OAuth provider.
        
        Args:
            server_url: Base URL of the server (e.g., "http://localhost:5005")
            login_path: Path to the login page (default: "/login")
        """
        self.server_url = server_url.rstrip("/")
        self.login_path = login_path
        
        # In-memory storage
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.auth_codes: dict[str, AuthorizationCode] = {}
        self.tokens: dict[str, AccessToken] = {}
        self.refresh_tokens: dict[str, RefreshToken] = {}  # refresh_token -> RefreshToken object
        self.refresh_to_access: dict[str, str] = {}  # refresh_token -> access_token (for cleanup)
        self.state_mapping: dict[str, dict[str, Any]] = {}
        
        # Pre-register default client for Claude Desktop compatibility
        # Claude Desktop may use a pre-configured client_id without dynamic registration
        self._register_default_client()
        
        logger.info(f"MemoryOAuthProvider initialized with server_url: {self.server_url}")
    
    def _register_default_client(self) -> None:
        """Pre-register default OAuth client for Claude Desktop compatibility."""
        # Parse redirect URIs from config (comma-separated list)
        redirect_uris = [
            AnyUrl(uri.strip())
            for uri in OAUTH_REDIRECT_URIS.split(",")
            if uri.strip()
        ]
        
        default_client = OAuthClientInformationFull(
            client_id=OAUTH_CLIENT_ID,
            client_secret=OAUTH_CLIENT_SECRET if OAUTH_CLIENT_SECRET else None,
            redirect_uris=redirect_uris,
            client_name="Memory MCP-CE Default Client",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_post" if OAUTH_CLIENT_SECRET else "none",
            scope="mcp",  # Required scope for MCP access
        )
        
        self.clients[OAUTH_CLIENT_ID] = default_client
        logger.info(f"✅ Pre-registered default OAuth client: {OAUTH_CLIENT_ID} with scope: mcp")
        logger.info(f"   Allowed redirect URIs: {[str(uri) for uri in redirect_uris]}")
    
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        """Get OAuth client information by client ID."""
        client = self.clients.get(client_id)
        if client:
            logger.debug(f"Found client: {client_id}")
        return client
    
    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        """
        Register a new OAuth client (RFC 7591 Dynamic Client Registration).
        
        Args:
            client_info: The client metadata to register
        """
        if not client_info.client_id:
            raise ValueError("No client_id provided")
        
        self.clients[client_info.client_id] = client_info
        logger.info(f"✅ Client registered: {client_info.client_id}")
    
    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        """
        Generate an authorization URL for the login flow.
        
        This is called when a client initiates the OAuth flow. We redirect
        to our login page with the necessary state.
        
        Args:
            client: The client requesting authorization
            params: The authorization parameters (redirect_uri, code_challenge, etc.)
            
        Returns:
            URL to redirect the client to for authentication
        """
        # Generate state for CSRF protection
        state = params.state or secrets.token_hex(16)
        
        # Store state mapping for the callback
        self.state_mapping[state] = {
            "redirect_uri": str(params.redirect_uri),
            "code_challenge": params.code_challenge,
            "redirect_uri_provided_explicitly": str(params.redirect_uri_provided_explicitly),
            "client_id": client.client_id,
            "scopes": params.scopes or ["mcp"],
            "resource": params.resource,  # RFC 8707
        }
        
        # Build login URL
        auth_url = f"{self.server_url}{self.login_path}?state={state}"
        if client.client_id:
            auth_url += f"&client_id={client.client_id}"
        
        logger.info(f"🔐 Authorization requested, redirecting to: {auth_url}")
        return auth_url
    
    async def get_login_page(self, state: str) -> HTMLResponse:
        """
        Generate the login page HTML.
        
        Args:
            state: The OAuth state parameter
            
        Returns:
            HTML response with the login form
        """
        if not state or state not in self.state_mapping:
            return HTMLResponse(
                content="<h1>Error: Invalid or missing state parameter</h1>",
                status_code=400
            )
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Memory MCP-CE - Login</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 400px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h2 {{
                    margin-top: 0;
                    color: #333;
                }}
                .form-group {{
                    margin-bottom: 15px;
                }}
                label {{
                    display: block;
                    margin-bottom: 5px;
                    font-weight: 500;
                    color: #555;
                }}
                input {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 14px;
                    box-sizing: border-box;
                    background: #fff;
                    color: #333;
                }}
                input:focus {{
                    outline: none;
                    border-color: #4CAF50;
                }}
                button {{
                    width: 100%;
                    padding: 12px;
                    background: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 16px;
                    cursor: pointer;
                    margin-top: 10px;
                }}
                button:hover {{
                    background: #45a049;
                }}
                .error {{
                    color: #e74c3c;
                    background: #fadbd8;
                    padding: 10px;
                    border-radius: 4px;
                    margin-bottom: 15px;
                }}
                
                /* Dark Mode */
                @media (prefers-color-scheme: dark) {{
                    body {{
                        background: #121212;
                        color: #e0e0e0;
                    }}
                    .container {{
                        background: #1e1e1e;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
                    }}
                    h2, label {{
                        color: #e0e0e0;
                    }}
                    input {{
                        background: #333;
                        color: #e0e0e0;
                        border-color: #555;
                    }}
                    input:focus {{
                        border-color: #4CAF50;
                    }}
                    .error {{
                        background: #5c2b2b;
                        color: #f5c6cb;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>🔐 Memory MCP-CE Login</h2>
                <form action="{self.server_url}/login/callback" method="post">
                    <input type="hidden" name="state" value="{state}">
                    <div class="form-group">
                        <label for="username">Username</label>
                        <input type="text" id="username" name="username" required autofocus>
                    </div>
                    <div class="form-group">
                        <label for="password">Password</label>
                        <input type="password" id="password" name="password" required>
                    </div>
                    <button type="submit">Sign In</button>
                </form>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    async def handle_login_callback(self, request: Request) -> Response:
        """
        Handle the login form submission.
        
        Args:
            request: The Starlette request object
            
        Returns:
            Redirect response to the client's redirect_uri with auth code
        """
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        state = form.get("state")
        
        if not username or not password or not state:
            return HTMLResponse(
                content="<h1>Error: Missing username, password, or state</h1>",
                status_code=400
            )
        
        # Ensure we have strings
        if not isinstance(username, str) or not isinstance(password, str) or not isinstance(state, str):
            return HTMLResponse(
                content="<h1>Error: Invalid parameter types</h1>",
                status_code=400
            )
        
        # Validate credentials
        if not self._authenticate_user(username, password):
            # Show login page again with error
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Memory MCP-CE - Login</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        max-width: 400px;
                        margin: 50px auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }}
                    .container {{
                        background: white;
                        padding: 30px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    h2 {{ margin-top: 0; color: #333; }}
                    .form-group {{ margin-bottom: 15px; }}
                    label {{ display: block; margin-bottom: 5px; font-weight: 500; color: #555; }}
                    input {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; box-sizing: border-box; }}
                    input:focus {{ outline: none; border-color: #4CAF50; }}
                    button {{ width: 100%; padding: 12px; background: #4CAF50; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; margin-top: 10px; }}
                    button:hover {{ background: #45a049; }}
                    .error {{ color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 4px; margin-bottom: 15px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>🔐 Memory MCP-CE Login</h2>
                    <div class="error">❌ Invalid username or password</div>
                    <form action="{self.server_url}/login/callback" method="post">
                        <input type="hidden" name="state" value="{state}">
                        <div class="form-group">
                            <label for="username">Username</label>
                            <input type="text" id="username" name="username" value="{username}" required>
                        </div>
                        <div class="form-group">
                            <label for="password">Password</label>
                            <input type="password" id="password" name="password" required>
                        </div>
                        <button type="submit">Sign In</button>
                    </form>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content, status_code=401)
        
        # Get state data
        state_data = self.state_mapping.get(state)
        if not state_data:
            return HTMLResponse(
                content="<h1>Error: Invalid or expired state</h1>",
                status_code=400
            )
        
        redirect_uri = state_data["redirect_uri"]
        code_challenge = state_data["code_challenge"]
        redirect_uri_provided_explicitly = state_data["redirect_uri_provided_explicitly"] == "True"
        client_id = state_data["client_id"]
        scopes = state_data.get("scopes", ["mcp"])
        resource = state_data.get("resource")
        
        # Generate authorization code
        auth_code = f"mcp_{secrets.token_hex(16)}"
        self.auth_codes[auth_code] = AuthorizationCode(
            code=auth_code,
            client_id=client_id,
            redirect_uri=AnyUrl(redirect_uri),
            redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
            expires_at=time.time() + OAUTH_AUTH_CODE_EXPIRY,
            scopes=scopes,
            code_challenge=code_challenge,
            resource=resource,
        )
        
        # Clean up state
        del self.state_mapping[state]
        
        # Redirect back to client with authorization code
        redirect_url = construct_redirect_uri(redirect_uri, code=auth_code, state=state)
        logger.info(f"✅ User {username} authenticated, redirecting to success page")
        
        # Redirect to success page, which will then redirect to the client
        import urllib.parse
        success_url = f"{self.server_url}/auth/success?redirect={urllib.parse.quote(redirect_url, safe='')}"
        return RedirectResponse(url=success_url, status_code=302)
    
    async def get_success_page(self, redirect_url: str) -> HTMLResponse:
        """
        Generate the success page HTML after successful authentication.
        
        Args:
            redirect_url: The URL to redirect to (client's callback with auth code)
            
        Returns:
            HTML response with success message and auto-close script
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Memory MCP-CE - Authentication Successful</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 400px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .success-icon {{
                    font-size: 48px;
                    margin-bottom: 15px;
                }}
                h2 {{
                    margin-top: 0;
                    color: #333;
                }}
                p {{
                    color: #666;
                    margin-bottom: 20px;
                }}
                
                /* Dark Mode */
                @media (prefers-color-scheme: dark) {{
                    body {{
                        background: #121212;
                        color: #e0e0e0;
                    }}
                    .container {{
                        background: #1e1e1e;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
                    }}
                    h2, p {{
                        color: #e0e_0e0;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">✅</div>
                <h2>Authentication Successful!</h2>
                <p>You may now close this window.</p>
            </div>
            <script>
                // Redirect to client callback immediately
                window.location.href = "{redirect_url}";
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    def _authenticate_user(self, username: str, password: str) -> bool:
        """
        Authenticate user against configured credentials.
        
        Args:
            username: The username to authenticate
            password: The password to authenticate
            
        Returns:
            True if credentials are valid, False otherwise
        """
        if not OAUTH_USERNAME or not OAUTH_PASSWORD:
            logger.error("OAuth credentials not configured in environment")
            return False
        
        is_valid = username == OAUTH_USERNAME and password == OAUTH_PASSWORD
        
        if is_valid:
            logger.info(f"✅ User authenticated: {username}")
        else:
            logger.warning(f"❌ Authentication failed for user: {username}")
        
        return is_valid
    
    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        """
        Load an authorization code by its code string.
        
        Args:
            client: The client that requested the code
            authorization_code: The authorization code string
            
        Returns:
            The AuthorizationCode object if found and valid, None otherwise
        """
        auth_code = self.auth_codes.get(authorization_code)
        if not auth_code:
            logger.warning(f"Authorization code not found: {authorization_code[:10]}...")
            return None
        
        # Check expiry
        if auth_code.expires_at < time.time():
            logger.warning("Authorization code expired")
            del self.auth_codes[authorization_code]
            return None
        
        # Verify client_id matches
        if auth_code.client_id != client.client_id:
            logger.warning(f"Client ID mismatch: {auth_code.client_id} != {client.client_id}")
            return None
        
        return auth_code
    
    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        """
        Exchange an authorization code for access and refresh tokens.
        
        Args:
            client: The client exchanging the code
            authorization_code: The authorization code to exchange
            
        Returns:
            OAuthToken containing access token and refresh token
        """
        if authorization_code.code not in self.auth_codes:
            raise ValueError("Invalid authorization code")
        
        if not client.client_id:
            raise ValueError("No client_id provided")
        
        # Generate access token
        access_token = f"mcp_{secrets.token_hex(32)}"
        access_expires_at = int(time.time()) + OAUTH_ACCESS_TOKEN_EXPIRY
        
        # Generate refresh token
        refresh_token_str = f"mcp_refresh_{secrets.token_hex(32)}"
        refresh_expires_at = int(time.time()) + OAUTH_REFRESH_TOKEN_EXPIRY
        
        # Store access token
        self.tokens[access_token] = AccessToken(
            token=access_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=access_expires_at,
            resource=authorization_code.resource,
        )
        
        # Store refresh token
        self.refresh_tokens[refresh_token_str] = RefreshToken(
            token=refresh_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=refresh_expires_at,
        )
        
        # Map refresh token to access token (for cleanup on rotation)
        self.refresh_to_access[refresh_token_str] = access_token
        
        # Remove used authorization code
        del self.auth_codes[authorization_code.code]
        
        logger.info(f"✅ Access token and refresh token issued for client: {client.client_id}")
        logger.info(f"   Access token expires in: {OAUTH_ACCESS_TOKEN_EXPIRY}s, Refresh token expires in: {OAUTH_REFRESH_TOKEN_EXPIRY}s")
        
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=OAUTH_ACCESS_TOKEN_EXPIRY,
            scope=" ".join(authorization_code.scopes),
            refresh_token=refresh_token_str,
        )
    
    async def load_access_token(self, token: str) -> AccessToken | None:
        """
        Load and validate an access token.
        
        This method supports BOTH:
        1. API Key authentication (for LobeChat) - checks against BEARER_TOKEN
        2. OAuth token authentication (for Claude Desktop) - checks token store
        
        Args:
            token: The access token string (could be API key or OAuth token)
            
        Returns:
            AccessToken if valid, None if invalid or expired
        """
        # 1. Check if token is the API key (for LobeChat compatibility)
        if BEARER_TOKEN and token == BEARER_TOKEN:
            logger.info("✅ Authenticated via API key")
            return AccessToken(
                token=token,
                client_id="api_key_client",
                scopes=["mcp"],
                expires_at=None,  # API keys don't expire
            )
        
        # 2. Check OAuth token store
        access_token = self.tokens.get(token)
        if not access_token:
            return None
        
        # Check expiry
        if access_token.expires_at and access_token.expires_at < time.time():
            logger.info(f"Token expired, removing: {token[:10]}...")
            del self.tokens[token]
            return None
        
        logger.info(f"✅ Authenticated via OAuth token for client: {access_token.client_id}")
        return access_token
    
    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        """
        Load a refresh token by its token string.
        
        Args:
            client: The client that owns the refresh token
            refresh_token: The refresh token string
            
        Returns:
            RefreshToken if found and valid, None otherwise
        """
        stored_token = self.refresh_tokens.get(refresh_token)
        if not stored_token:
            logger.warning(f"Refresh token not found: {refresh_token[:20]}...")
            return None
        
        # Verify client_id matches
        if stored_token.client_id != client.client_id:
            logger.warning(f"Refresh token client mismatch: {stored_token.client_id} != {client.client_id}")
            return None
        
        # Check expiry
        if stored_token.expires_at and stored_token.expires_at < time.time():
            logger.warning(f"Refresh token expired: {refresh_token[:20]}...")
            # Clean up expired token
            del self.refresh_tokens[refresh_token]
            if refresh_token in self.refresh_to_access:
                del self.refresh_to_access[refresh_token]
            return None
        
        logger.info(f"✅ Refresh token loaded for client: {client.client_id}")
        return stored_token
    
    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """
        Exchange a refresh token for new access and refresh tokens.
        
        Implements OAuth 2.1 refresh token rotation:
        - Issues new access token
        - Issues new refresh token (rotation)
        - Invalidates old refresh token
        - Invalidates old access token
        
        Args:
            client: The client exchanging the token
            refresh_token: The refresh token to exchange
            scopes: Requested scopes (must be subset of original)
            
        Returns:
            OAuthToken with new access and refresh tokens
        """
        if not client.client_id:
            raise ValueError("No client_id provided")
        
        # Verify refresh token exists
        if refresh_token.token not in self.refresh_tokens:
            raise ValueError("Invalid refresh token")
        
        # Use original scopes if none requested, otherwise validate
        token_scopes = scopes if scopes else refresh_token.scopes
        
        # Validate requested scopes are subset of original
        for scope in token_scopes:
            if scope not in refresh_token.scopes:
                raise ValueError(f"Cannot request scope '{scope}' not in original grant")
        
        # Get old access token for cleanup
        old_access_token = self.refresh_to_access.get(refresh_token.token)
        
        # Generate new access token
        new_access_token = f"mcp_{secrets.token_hex(32)}"
        access_expires_at = int(time.time()) + OAUTH_ACCESS_TOKEN_EXPIRY
        
        # Generate new refresh token (rotation per OAuth 2.1)
        new_refresh_token_str = f"mcp_refresh_{secrets.token_hex(32)}"
        refresh_expires_at = int(time.time()) + OAUTH_REFRESH_TOKEN_EXPIRY
        
        # Store new access token
        self.tokens[new_access_token] = AccessToken(
            token=new_access_token,
            client_id=client.client_id,
            scopes=token_scopes,
            expires_at=access_expires_at,
        )
        
        # Store new refresh token
        self.refresh_tokens[new_refresh_token_str] = RefreshToken(
            token=new_refresh_token_str,
            client_id=client.client_id,
            scopes=token_scopes,
            expires_at=refresh_expires_at,
        )
        
        # Map new refresh token to new access token
        self.refresh_to_access[new_refresh_token_str] = new_access_token
        
        # Invalidate old refresh token (rotation)
        del self.refresh_tokens[refresh_token.token]
        if refresh_token.token in self.refresh_to_access:
            del self.refresh_to_access[refresh_token.token]
        
        # Invalidate old access token
        if old_access_token and old_access_token in self.tokens:
            del self.tokens[old_access_token]
        
        logger.info(f"🔄 Tokens rotated for client: {client.client_id}")
        logger.info(f"   New access token expires in: {OAUTH_ACCESS_TOKEN_EXPIRY}s")
        logger.info(f"   New refresh token expires in: {OAUTH_REFRESH_TOKEN_EXPIRY}s")
        
        return OAuthToken(
            access_token=new_access_token,
            token_type="Bearer",
            expires_in=OAUTH_ACCESS_TOKEN_EXPIRY,
            scope=" ".join(token_scopes),
            refresh_token=new_refresh_token_str,
        )
    
    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        """
        Revoke an access or refresh token.
        
        When revoking:
        - Access token: Also revokes associated refresh tokens
        - Refresh token: Also revokes associated access token
        
        Args:
            token: The token to revoke
        """
        if isinstance(token, RefreshToken):
            # Revoke refresh token
            if token.token in self.refresh_tokens:
                del self.refresh_tokens[token.token]
                logger.info(f"Refresh token revoked: {token.token[:20]}...")
            
            # Also revoke associated access token
            if token.token in self.refresh_to_access:
                access_token = self.refresh_to_access[token.token]
                if access_token in self.tokens:
                    del self.tokens[access_token]
                    logger.info(f"Associated access token revoked: {access_token[:10]}...")
                del self.refresh_to_access[token.token]
                
        elif isinstance(token, AccessToken):
            # Revoke access token
            if token.token in self.tokens:
                del self.tokens[token.token]
                logger.info(f"Access token revoked: {token.token[:10]}...")
            
            # Also revoke any refresh tokens that point to this access token
            refresh_tokens_to_remove = [
                rt for rt, at in self.refresh_to_access.items() 
                if at == token.token
            ]
            for rt in refresh_tokens_to_remove:
                if rt in self.refresh_tokens:
                    del self.refresh_tokens[rt]
                    logger.info(f"Associated refresh token revoked: {rt[:20]}...")
                del self.refresh_to_access[rt]
