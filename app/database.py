"""
Memory MCP-CE Database Module - V5 Key-Value System State

This module handles:
- Database connection management
- Schema initialization (memories, system_state, memory_{dims} tables)
- Version tracking via system_state key-value store
- OAuth session persistence

Migrations are handled by the app.migrations module.

V5 Schema:
- HNSW indexes for unlimited embedding dimensions
- state.embedding_tables as object structure for A/B testing
- Flexible key-value system_state table
"""

import psycopg2
import psycopg2.extras
import logging
import hashlib
import time
from typing import Optional, Any
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
    """Create the main memories table (source of truth) with V6 schema."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                content_id BIGINT NOT NULL,
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
        
        # V6: Index for efficient MAX(content_id) queries per namespace
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_namespace_content_id 
            ON memories(namespace, content_id DESC);
        """)
        
        conn.commit()
        logger.info("✅ Created memories table with indexes (V6 schema)")
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


# =============================================================================
# OAuth Session Persistence Functions
# =============================================================================

def _oauth_key_hash(token: str) -> str:
    """
    Generate a short hash for OAuth token keys.
    
    Uses SHA256 truncated to 16 characters for clean, predictable keys
    while the full token is stored in JSONB value.
    
    Args:
        token: The OAuth token string
        
    Returns:
        16-character hex hash
    """
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def save_oauth_client(client_id: str, client_data: dict) -> None:
    """
    Save an OAuth client registration to the database.
    
    Args:
        client_id: The client ID
        client_data: Client data (serialized OAuthClientInformationFull)
    """
    key = f"oauth:client:{client_id}"
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO system_state (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = CURRENT_TIMESTAMP;
        """, (key, psycopg2.extras.Json(client_data)))
        conn.commit()
        logger.debug(f"💾 Saved OAuth client: {client_id}")
    finally:
        cur.close()
        conn.close()


def save_oauth_access_token(token: str, token_data: dict) -> None:
    """
    Save an OAuth access token to the database.
    
    Args:
        token: The access token string
        token_data: Token data (token, client_id, scopes, expires_at, resource)
    """
    key = f"oauth:access_token:{_oauth_key_hash(token)}"
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO system_state (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = CURRENT_TIMESTAMP;
        """, (key, psycopg2.extras.Json(token_data)))
        conn.commit()
        logger.debug(f"💾 Saved OAuth access token: {token[:10]}...")
    finally:
        cur.close()
        conn.close()


def save_oauth_refresh_token(token: str, token_data: dict, access_token: str) -> None:
    """
    Save an OAuth refresh token and its mapping to access token.
    
    Args:
        token: The refresh token string
        token_data: Token data (token, client_id, scopes, expires_at)
        access_token: The associated access token
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Save refresh token
        refresh_key = f"oauth:refresh_token:{_oauth_key_hash(token)}"
        cur.execute("""
            INSERT INTO system_state (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = CURRENT_TIMESTAMP;
        """, (refresh_key, psycopg2.extras.Json(token_data)))
        
        # Save refresh_to_access mapping
        mapping_key = f"oauth:refresh_to_access:{_oauth_key_hash(token)}"
        cur.execute("""
            INSERT INTO system_state (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = CURRENT_TIMESTAMP;
        """, (mapping_key, psycopg2.extras.Json({"access_token": access_token})))
        
        conn.commit()
        logger.debug(f"💾 Saved OAuth refresh token: {token[:20]}...")
    finally:
        cur.close()
        conn.close()


def delete_oauth_token(token: str, token_type: str = "access") -> None:
    """
    Delete an OAuth token from the database.
    
    Args:
        token: The token string
        token_type: "access" or "refresh"
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if token_type == "access":
            key = f"oauth:access_token:{_oauth_key_hash(token)}"
        else:
            # For refresh tokens, delete both the token and the mapping
            key = f"oauth:refresh_token:{_oauth_key_hash(token)}"
            mapping_key = f"oauth:refresh_to_access:{_oauth_key_hash(token)}"
            cur.execute("DELETE FROM system_state WHERE key = %s;", (mapping_key,))
        
        cur.execute("DELETE FROM system_state WHERE key = %s;", (key,))
        conn.commit()
        logger.debug(f"🗑️ Deleted OAuth {token_type} token: {token[:10]}...")
    finally:
        cur.close()
        conn.close()


def delete_oauth_client(client_id: str) -> None:
    """
    Delete an OAuth client from the database.
    
    Args:
        client_id: The client ID to delete
    """
    key = f"oauth:client:{client_id}"
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM system_state WHERE key = %s;", (key,))
        conn.commit()
        logger.debug(f"🗑️ Deleted OAuth client: {client_id}")
    finally:
        cur.close()
        conn.close()


def load_oauth_sessions() -> dict[str, Any]:
    """
    Load all OAuth session data from the database.
    
    Returns a dict with:
    - clients: dict of client_id -> client_data
    - access_tokens: dict of token -> token_data
    - refresh_tokens: dict of token -> token_data
    - refresh_to_access: dict of refresh_token -> access_token
    
    Also performs cleanup of expired tokens during load.
    
    Returns:
        Dict containing all OAuth session data
    """
    if not table_exists('system_state'):
        return {
            "clients": {},
            "access_tokens": {},
            "refresh_tokens": {},
            "refresh_to_access": {},
        }
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Load all OAuth keys
        cur.execute("""
            SELECT key, value FROM system_state
            WHERE key LIKE 'oauth:%';
        """)
        rows = cur.fetchall()
        
        clients = {}
        access_tokens = {}
        refresh_tokens = {}
        refresh_to_access = {}
        expired_keys = []
        
        current_time = time.time()
        
        for key, value in rows:
            if key.startswith("oauth:client:"):
                client_id = key[len("oauth:client:"):]
                clients[client_id] = value
                
            elif key.startswith("oauth:access_token:"):
                # Check expiry
                expires_at = value.get("expires_at")
                if expires_at and expires_at < current_time:
                    expired_keys.append(key)
                    logger.debug(f"🧹 Found expired access token: {value.get('token', '')[:10]}...")
                else:
                    token = value.get("token")
                    if token:
                        access_tokens[token] = value
                        
            elif key.startswith("oauth:refresh_token:"):
                # Check expiry
                expires_at = value.get("expires_at")
                if expires_at and expires_at < current_time:
                    expired_keys.append(key)
                    # Also mark the mapping for cleanup
                    hash_part = key[len("oauth:refresh_token:"):]
                    expired_keys.append(f"oauth:refresh_to_access:{hash_part}")
                    logger.debug(f"🧹 Found expired refresh token: {value.get('token', '')[:20]}...")
                else:
                    token = value.get("token")
                    if token:
                        refresh_tokens[token] = value
                        
            elif key.startswith("oauth:refresh_to_access:"):
                # Will be validated after loading refresh tokens
                hash_part = key[len("oauth:refresh_to_access:"):]
                access_token = value.get("access_token")
                if access_token:
                    # Find the refresh token that maps to this hash
                    for rt_key, rt_val in rows:
                        if rt_key == f"oauth:refresh_token:{hash_part}":
                            rt = rt_val.get("token")
                            if rt:
                                refresh_to_access[rt] = access_token
                            break
        
        # Cleanup expired tokens from database
        if expired_keys:
            for exp_key in expired_keys:
                cur.execute("DELETE FROM system_state WHERE key = %s;", (exp_key,))
            conn.commit()
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired OAuth tokens from database")
        
        return {
            "clients": clients,
            "access_tokens": access_tokens,
            "refresh_tokens": refresh_tokens,
            "refresh_to_access": refresh_to_access,
        }
    finally:
        cur.close()
        conn.close()


def cleanup_expired_oauth_sessions() -> int:
    """
    Clean up expired OAuth sessions from the database.
    
    Returns:
        Number of expired sessions cleaned up
    """
    if not table_exists('system_state'):
        return 0
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT key, value FROM system_state
            WHERE key LIKE 'oauth:access_token:%' OR key LIKE 'oauth:refresh_token:%';
        """)
        rows = cur.fetchall()
        
        expired_keys = []
        current_time = time.time()
        
        for key, value in rows:
            expires_at = value.get("expires_at")
            if expires_at and expires_at < current_time:
                expired_keys.append(key)
                # If it's a refresh token, also clean up the mapping
                if key.startswith("oauth:refresh_token:"):
                    hash_part = key[len("oauth:refresh_token:"):]
                    expired_keys.append(f"oauth:refresh_to_access:{hash_part}")
        
        if expired_keys:
            for exp_key in expired_keys:
                cur.execute("DELETE FROM system_state WHERE key = %s;", (exp_key,))
            conn.commit()
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired OAuth tokens")
        
        return len(expired_keys)
    finally:
        cur.close()
        conn.close()
