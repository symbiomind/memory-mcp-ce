"""
Memory MCP-CE Database Module - V2 Split-Table Architecture

This module handles:
- Database connection management
- Schema initialization (memories, system_state, memory_{dims} tables)
- V1 → V2 migration for existing installations
- Version tracking via system_state singleton
"""

import psycopg2
import psycopg2.extras
import hashlib
import logging
from typing import Optional, Any
from app.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB

# Configure logging
logger = logging.getLogger(__name__)

# Current database schema version
CURRENT_DB_VERSION = 2


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
    """Get the system state from the singleton table."""
    if not table_exists('system_state'):
        return None
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("SELECT * FROM system_state WHERE id = 1;")
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        cur.close()
        conn.close()


def set_system_state(db_version: int = None, jwt_state: dict = None, service_state: dict = None) -> None:
    """Update the system state singleton."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Build dynamic update
        updates = ["updated_at = NOW()"]
        params = []
        
        if db_version is not None:
            updates.append("db_version = %s")
            params.append(db_version)
            updates.append("last_migration = NOW()")
        
        if jwt_state is not None:
            updates.append("jwt_state = %s")
            params.append(psycopg2.extras.Json(jwt_state))
        
        if service_state is not None:
            updates.append("service_state = %s")
            params.append(psycopg2.extras.Json(service_state))
        
        cur.execute(f"""
            UPDATE system_state 
            SET {', '.join(updates)}
            WHERE id = 1;
        """, params)
        
        conn.commit()
    finally:
        cur.close()
        conn.close()


def create_system_state_table() -> None:
    """Create the system_state singleton table."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY DEFAULT 1,
                db_version INTEGER NOT NULL,
                jwt_state JSONB DEFAULT '{}'::JSONB,
                service_state JSONB DEFAULT '{}'::JSONB,
                last_migration TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                CHECK (id = 1)
            );
        """)
        conn.commit()
        logger.info("✅ Created system_state table")
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
        
        # Create indexes for semantic queries
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_embedding_{embedding_dim} 
            ON {table_name} USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
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


def is_v1_schema(table_name: str) -> bool:
    """Check if an embedding table has V1 schema (content column exists)."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = %s
                AND column_name = 'content'
            );
        """, (table_name,))
        has_content = cur.fetchone()[0]
        return has_content
    finally:
        cur.close()
        conn.close()


def migrate_v1_to_v2(embedding_dim: int) -> None:
    """
    Migrate from V1 (single table) to V2 (split table) architecture.
    
    This migration:
    1. Creates the new memories and system_state tables
    2. Extracts content from existing memory_{dims} tables into memories
    3. Rebuilds embedding tables with foreign keys
    4. Updates state.embedding_tables for each memory
    """
    logger.info("🔄 Starting V1 → V2 migration...")
    
    # Step 1: Create new tables
    create_system_state_table()
    create_memories_table()
    
    # Initialize system_state with version 1 (pre-migration)
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO system_state (id, db_version) 
            VALUES (1, 1)
            ON CONFLICT (id) DO NOTHING;
        """)
        conn.commit()
    finally:
        cur.close()
        conn.close()
    
    # Step 2: Find all existing V1 embedding tables
    existing_tables = get_existing_embedding_tables()
    v1_tables = [t for t in existing_tables if is_v1_schema(t)]
    
    if not v1_tables:
        logger.info("📭 No V1 tables found to migrate")
        # Just set version and return
        set_system_state(db_version=CURRENT_DB_VERSION)
        return
    
    logger.info(f"📋 Found {len(v1_tables)} V1 tables to migrate: {v1_tables}")
    
    # Step 3: Extract unique memories and migrate
    # We'll use content hash for deduplication across tables
    content_to_memory_id: dict[str, int] = {}
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        for table_name in v1_tables:
            logger.info(f"🔄 Migrating table: {table_name}")
            
            # Extract dimension from table name
            dims = int(table_name.replace('memory_', ''))
            
            # Get all rows from V1 table
            cur.execute(f"""
                SELECT id, content, embedding, namespace, labels, source, 
                       timestamp, enc, embedding_model, state
                FROM {table_name}
                ORDER BY id;
            """)
            rows = cur.fetchall()
            
            logger.info(f"   Found {len(rows)} memories in {table_name}")
            
            for row in rows:
                # Create content hash for deduplication
                content_bytes = bytes(row['content'])
                content_hash = hashlib.sha256(content_bytes).hexdigest()
                
                if content_hash in content_to_memory_id:
                    # Content already migrated, just add embedding reference
                    memory_id = content_to_memory_id[content_hash]
                    logger.debug(f"   Dedup: content already exists as memory #{memory_id}")
                else:
                    # Insert new memory into memories table
                    cur.execute("""
                        INSERT INTO memories (content, namespace, labels, source, timestamp, enc, state)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (
                        content_bytes,
                        row['namespace'] or 'default',
                        psycopg2.extras.Json(row['labels'] if row['labels'] else []),
                        row['source'],
                        row['timestamp'],
                        row['enc'] if row['enc'] is not None else False,
                        psycopg2.extras.Json({'embedding_tables': [table_name]})
                    ))
                    memory_id = cur.fetchone()['id']
                    content_to_memory_id[content_hash] = memory_id
                
                # We'll rebuild the embedding table after dropping the old one
                # For now, just track the mapping
            
            conn.commit()
        
        logger.info(f"✅ Migrated {len(content_to_memory_id)} unique memories to memories table")
        
        # Step 4: Rebuild embedding tables with V2 schema
        for table_name in v1_tables:
            dims = int(table_name.replace('memory_', ''))
            
            logger.info(f"🔄 Rebuilding embedding table: {table_name}")
            
            # Get embeddings from old table before dropping
            cur.execute(f"""
                SELECT content, embedding, namespace, embedding_model
                FROM {table_name};
            """)
            old_embeddings = cur.fetchall()
            
            # Drop old table
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            conn.commit()
            
            # Create new V2 table
            create_embedding_table(dims)
            
            # Re-insert embeddings with foreign key references
            for emb_row in old_embeddings:
                content_bytes = bytes(emb_row['content'])
                content_hash = hashlib.sha256(content_bytes).hexdigest()
                memory_id = content_to_memory_id.get(content_hash)
                
                if memory_id:
                    cur.execute(f"""
                        INSERT INTO {table_name} (memory_id, embedding, namespace, embedding_model)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (memory_id, embedding_model) DO NOTHING;
                    """, (
                        memory_id,
                        emb_row['embedding'],
                        emb_row['namespace'] or 'default',
                        emb_row['embedding_model']
                    ))
            
            conn.commit()
            logger.info(f"✅ Rebuilt {table_name} with V2 schema")
        
        # Step 5: Update system state to V2
        set_system_state(db_version=CURRENT_DB_VERSION)
        
        logger.info("🎉 V1 → V2 migration complete!")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def init_database(embedding_dim: int) -> None:
    """
    Initialize the database schema.
    
    This function:
    1. Ensures the vector extension is installed
    2. Checks database version via system_state
    3. Runs migration if needed (V1 → V2)
    4. Creates/verifies all required tables
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Ensure vector extension is installed
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    finally:
        cur.close()
        conn.close()
    
    # Check current database state
    system_state = get_system_state()
    
    if system_state is None:
        # No system_state table - could be fresh install or V1
        existing_tables = get_existing_embedding_tables()
        
        if existing_tables:
            # V1 installation detected - need migration
            v1_tables = [t for t in existing_tables if is_v1_schema(t)]
            if v1_tables:
                logger.info("🔍 Detected V1 schema - migration required")
                migrate_v1_to_v2(embedding_dim)
            else:
                # Tables exist but already V2 schema (partial migration?)
                logger.info("🔍 Tables exist with V2 schema - initializing system_state")
                create_system_state_table()
                conn = get_db_connection()
                cur = conn.cursor()
                try:
                    cur.execute("""
                        INSERT INTO system_state (id, db_version) 
                        VALUES (1, %s)
                        ON CONFLICT (id) DO UPDATE SET db_version = %s;
                    """, (CURRENT_DB_VERSION, CURRENT_DB_VERSION))
                    conn.commit()
                finally:
                    cur.close()
                    conn.close()
        else:
            # Fresh installation - create everything from scratch
            logger.info("🆕 Fresh installation detected - creating V2 schema")
            create_system_state_table()
            create_memories_table()
            
            # Initialize system_state
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO system_state (id, db_version) 
                    VALUES (1, %s)
                    ON CONFLICT (id) DO NOTHING;
                """, (CURRENT_DB_VERSION,))
                conn.commit()
            finally:
                cur.close()
                conn.close()
    
    else:
        # system_state exists - check version
        current_version = system_state.get('db_version', 1)
        
        if current_version < CURRENT_DB_VERSION:
            logger.info(f"🔍 Database version {current_version} < {CURRENT_DB_VERSION} - migration required")
            migrate_v1_to_v2(embedding_dim)
        else:
            logger.info(f"✅ Database schema is up to date (version {current_version})")
    
    # Ensure memories table exists (idempotent)
    if not table_exists('memories'):
        create_memories_table()
    
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


def add_embedding_table_to_state(memory_id: int, table_name: str) -> None:
    """
    Add an embedding table to a memory's state.embedding_tables array.
    
    This tracks which embedding tables have embeddings for this memory,
    enabling proper cleanup on delete.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Add table_name to embedding_tables array if not already present
        cur.execute("""
            UPDATE memories
            SET state = jsonb_set(
                COALESCE(state, '{}'::jsonb),
                '{embedding_tables}',
                COALESCE(state->'embedding_tables', '[]'::jsonb) || 
                CASE WHEN state->'embedding_tables' @> %s::jsonb 
                     THEN '[]'::jsonb 
                     ELSE %s::jsonb 
                END,
                true
            )
            WHERE id = %s;
        """, (f'["{table_name}"]', f'["{table_name}"]', memory_id))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def get_memory_embedding_tables(memory_id: int) -> list[str]:
    """
    Get the list of embedding tables that have embeddings for a memory.
    
    Returns:
        List of table names (e.g., ['memory_384', 'memory_768'])
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
            return result[0]
        return []
    finally:
        cur.close()
        conn.close()
