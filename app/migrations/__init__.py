"""
Memory MCP-CE Migrations Module

Provides database schema migrations from V1 through V8.
"""

from app.migrations.runner import CURRENT_DB_VERSION, run_migrations

__all__ = ['CURRENT_DB_VERSION', 'run_migrations']
