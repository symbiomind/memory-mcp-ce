"""
Memory MCP-CE Database Module - V5 Key-Value System State

This module handles:
- Database connection management
- Schema initialization (memories, system_state, memory_{dims} tables)
- Version tracking via system_state key-value store

Migrations are handled by the app.migrations module.

V5 Schema:
- HNSW indexes for unlimited embedding dimensions
- state.embedding_tables as object structure for A/B testing
- Flexible key-value system_state table
"""

import psycopg2
import psycopg2.extras
import logging
from typing import Optional
from app.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB

# Configure logging
logger = logging.getLogger(__name__)


def get_db_connection():
    """Establishes connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB
    )
    return conn


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table_name,))
        exists = cur.fetchone()[0]
        return exists
    finally:
        cur.close()
        conn.close()


def get_existing_embedding_tables() -> list[str]:
    """Find all existing memory_{dims} tables in the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'memory_%'
            AND table_name != 'memories';
        """)
        tables = [row[0] for row in cur.fetchall()]
        return tables
    finally:
        cur.close()
        conn.close()


def get_system_state() -> Optional[dict]:
    """
    Get the system state as a dictionary from the key-value table.
    
    V5 Schema: key-value pairs are returned as a dict.
    For backwards compatibility during migration, also handles V4 fixed-column schema.
    
    Returns:
        Dict with system state values, or None if table doesn't exist
    """
    if not table_exists('system_state'):
        return None
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Check if this is V5 schema (has 'key' column) or V4 schema (has 'db_version' column)
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'system_state' AND column_name = 'key';
        """)
        is_v5_schema = cur.fetchone() is not None
        
        if is_v5_schema:
            # V5 key-value schema
            cur.execute("SELECT key, value FROM system_state;")
            rows = cur.fetchall()
            if not rows:
                return None
            result = {}
            for key, value in rows:
                result[key] = value
            return result
        else:
            # V4 fixed-column schema (for backwards compatibility during migration)
            cur.execute("SELECT * FROM system_state WHERE id = 1;")
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if row:
                return dict(zip(columns, row))
            return None
    finally:
        cur.close()
        conn.close()


def set_system_state(db_version: int = None, **kwargs) -> None:
    """
    Update system state key-value pairs.
    
    V5 Schema: each parameter becomes a key-value pair.
    
    Args:
        db_version: Database version (special case for compatibility)
        **kwargs: Any additional key-value pairs to upsert
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Collect all key-value pairs to upsert
        updates = {}
        if db_version is not None:
            updates['db_version'] = db_version
        updates.update(kwargs)
        
        for key, value in updates.items():
            cur.execute("""
                INSERT INTO system_state (key, value) 
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET 
                    value = EXCLUDED.value,
                    updated_at = CURRENT_TIMESTAMP;
            """, (key, psycopg2.extras.Json(value)))
        
        conn.commit()
    finally:
        cur.close()
        conn.close()


def create_system_state_table() -> None:
    """
    Create the V5 key-value system_state table.
    
    V5 Schema:
    - id: SERIAL PRIMARY KEY
    - key: TEXT UNIQUE NOT NULL (the setting name)
    - value: JSONB NOT NULL (the setting value)
    - created_at/updated_at: timestamps
    """
    from app.migrations.runner import CURRENT_DB_VERSION
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                id SERIAL PRIMARY KEY,
                key TEXT UNIQUE NOT NULL,
                value JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Initialize with db_version
        cur.execute("""
            INSERT INTO system_state (key, value) 
            VALUES ('db_version', %s)
            ON CONFLICT (key) DO NOTHING;
        """, (psycopg2.extras.Json(CURRENT_DB_VERSION),))
        
        conn.commit()
        logger.info("✅ Created V5 system_state table")
    finally:
        cur.close()
        conn.close()


def create_memories_table() -> None:
    """Create the main memories table (source of truth)."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                content BYTEA NOT NULL,
                namespace VARCHAR(100) DEFAULT 'default',
                labels JSONB DEFAULT '[]'::JSONB,
                source VARCHAR(255),
                timestamp TIMESTAMP DEFAULT NOW(),
                enc BOOLEAN DEFAULT FALSE,
                state JSONB DEFAULT '{}'::JSONB
            );
        """)
        
        # Create indexes for non-semantic queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_namespace 
            ON memories(namespace);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_labels_gin 
            ON memories USING GIN(labels);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_source 
            ON memories(source);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
            ON memories(timestamp DESC);
        """)
        
        conn.commit()
        logger.info("✅ Created memories table with indexes")
    finally:
        cur.close()
        conn.close()


def create_embedding_table(embedding_dim: int) -> None:
    """Create an embedding table for a specific dimension (V2 schema with foreign key)."""
    table_name = f"memory_{embedding_dim}"
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Create the embedding table with foreign key to memories
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                embedding VECTOR({embedding_dim}) NOT NULL,
                namespace VARCHAR(100) NOT NULL,
                embedding_model VARCHAR(255) NOT NULL,
                UNIQUE(memory_id, embedding_model)
            );
        """)
        
        # Create indexes for semantic queries (HNSW for unlimited dimensions)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_embedding_{embedding_dim} 
            ON {table_name} USING hnsw (embedding vector_cosine_ops);
        """)
        
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_namespace_{embedding_dim} 
            ON {table_name}(namespace);
        """)
        
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_model_{embedding_dim} 
            ON {table_name}(embedding_model);
        """)
        
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_memory_id_{embedding_dim} 
            ON {table_name}(memory_id);
        """)
        
        conn.commit()
        logger.info(f"✅ Created/verified embedding table {table_name} with indexes")
    finally:
        cur.close()
        conn.close()


def init_database(embedding_dim: int) -> None:
    """
    Initialize the database schema.
    
    This function:
    1. Ensures the vector extension is installed
    2. Runs migrations via the migrations module
    3. Creates/verifies embedding table for current dimension
    """
    # Import here to avoid circular imports
    from app.migrations import run_migrations
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Ensure vector extension is installed
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    finally:
        cur.close()
        conn.close()
    
    # Run migrations (handles fresh install, V1→V2→V3→V4, and version checks)
    run_migrations(embedding_dim)
    
    # Ensure embedding table for current dimension exists
    create_embedding_table(embedding_dim)
    
    logger.info(f"✅ Database initialization complete (embedding dim: {embedding_dim})")


def update_memory_state(memory_id: int, state_updates: dict) -> None:
    """
    Update the state JSONB field for a memory.
    
    Args:
        memory_id: The memory ID to update
        state_updates: Dictionary of state fields to update/merge
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Merge state_updates into existing state
        cur.execute("""
            UPDATE memories
            SET state = COALESCE(state, '{}'::jsonb) || %s::jsonb
            WHERE id = %s;
        """, (psycopg2.extras.Json(state_updates), memory_id))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def add_embedding_to_state(memory_id: int, table_name: str, model_name: str) -> None:
    """
    Add an embedding model to a memory's state.embedding_tables[table_name] array.
    
    V3 Structure:
    {
        "embedding_tables": {
            "memory_384": ["granite:30m", "other_model:32m"],
            "memory_768": ["embeddinggemma:300m"]
        }
    }
    
    This tracks which embedding models have generated embeddings in each table
    for this memory, enabling A/B testing and proper cleanup on delete.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Use PostgreSQL JSONB operations to add model to table's array
        # 1. Ensure embedding_tables exists as an object
        # 2. Ensure the table_name key exists with an array
        # 3. Append model_name if not already present
        cur.execute("""
            UPDATE memories
            SET state = jsonb_set(
                jsonb_set(
                    COALESCE(state, '{}'::jsonb),
                    '{embedding_tables}',
                    COALESCE(state->'embedding_tables', '{}'::jsonb),
                    true
                ),
                ARRAY['embedding_tables', %s],
                (
                    SELECT CASE 
                        WHEN COALESCE(state->'embedding_tables'->%s, '[]'::jsonb) @> %s::jsonb
                        THEN COALESCE(state->'embedding_tables'->%s, '[]'::jsonb)
                        ELSE COALESCE(state->'embedding_tables'->%s, '[]'::jsonb) || %s::jsonb
                    END
                ),
                true
            )
            WHERE id = %s;
        """, (table_name, table_name, f'["{model_name}"]', table_name, table_name, f'["{model_name}"]', memory_id))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def get_memory_embedding_tables(memory_id: int) -> dict[str, list[str]]:
    """
    Get the embedding tables and their models for a memory.
    
    V3 Structure:
    {
        "memory_384": ["granite:30m", "other_model:32m"],
        "memory_768": ["embeddinggemma:300m"]
    }
    
    Returns:
        Dict mapping table names to list of model names
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT state->'embedding_tables' 
            FROM memories 
            WHERE id = %s;
        """, (memory_id,))
        result = cur.fetchone()
        if result and result[0]:
            # Handle both V2 (array) and V3 (object) formats for backwards compatibility
            embedding_tables = result[0]
            if isinstance(embedding_tables, list):
                # V2 format - convert to V3 format with empty model arrays
                # This shouldn't happen after migration, but handle gracefully
                return {table: [] for table in embedding_tables}
            elif isinstance(embedding_tables, dict):
                # V3 format - return as-is
                return embedding_tables
        return {}
    finally:
        cur.close()
        conn.close()
