"""
Memory MCP-CE API Module

REST API endpoints for administrative and SaaS operations.
Separate from MCP endpoints - uses API_BEARER_TOKEN for authentication.
"""

from app.api.embeddings import generate_embeddings_handler

__all__ = ['generate_embeddings_handler']
