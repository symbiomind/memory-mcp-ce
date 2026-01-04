"""
Memory MCP-CE Migrations Module

Provides database schema migrations from V1 → V2 → V3 → V4 → V5.
"""

from app.migrations.runner import CURRENT_DB_VERSION, run_migrations

__all__ = ['CURRENT_DB_VERSION', 'run_migrations']
