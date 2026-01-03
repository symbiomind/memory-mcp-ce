"""
Memory MCP-CE Database Module - V4 HNSW Index Architecture

This module handles:
- Database connection management
- Schema initialization (memories, system_state, memory_{dims} tables)
- V1 → V2 → V3 → V4 migration for existing installations
- Version tracking via system_state singleton

V4 Changes:
- Migrated from ivfflat to HNSW indexes for embedding tables
- HNSW indexes support unlimited dimensions (ivfflat had 2000-dim limit)
- Enables use of large embedding models (e.g., Qwen 4096D)

V3 Changes:
- state.embedding_tables changed from array to object structure
- Now tracks which models generated embeddings in each table
- Enables A/B testing of embedding models

V3 Structure:
{
    "embedding_tables": {
        "memory_384": ["granite:30m", "other_model:32m"],
        "memory_768": ["embeddinggemma:300m"]
    }
}
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
CURRENT_DB_VERSION = 4


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


def migrate_v2_to_v3() -> None:
    """
    Migrate from V2 (array) to V3 (object) embedding_tables structure.
    
    V2 Structure: {"embedding_tables": ["memory_384", "memory_768"]}
    V3 Structure: {"embedding_tables": {"memory_384": ["model1"], "memory_768": ["model2"]}}
    
    This migration:
    1. Finds all memories with V2 array format in state.embedding_tables
    2. For each memory, queries embedding tables to find actual models used
    3. Converts to V3 object format with model arrays
    """
    logger.info("🔄 Starting V2 → V3 migration (embedding_tables array → object)...")
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        # Find all memories that need migration
        # V2 format has embedding_tables as an array (jsonb_typeof = 'array')
        cur.execute("""
            SELECT id, state->'embedding_tables' as embedding_tables
            FROM memories
            WHERE state->'embedding_tables' IS NOT NULL
            AND jsonb_typeof(state->'embedding_tables') = 'array';
        """)
        memories_to_migrate = cur.fetchall()
        
        if not memories_to_migrate:
            logger.info("📭 No V2 memories found to migrate")
            set_system_state(db_version=CURRENT_DB_VERSION)
            return
        
        logger.info(f"📋 Found {len(memories_to_migrate)} memories to migrate to V3 format")
        
        migrated_count = 0
        for memory in memories_to_migrate:
            memory_id = memory['id']
            old_tables = memory['embedding_tables']  # This is an array like ["memory_384"]
            
            # Build new V3 structure by querying each embedding table
            new_structure = {}
            
            for table_name in old_tables:
                # Check if this table exists
                if not table_exists(table_name):
                    logger.debug(f"   Table {table_name} no longer exists, skipping")
                    continue
                
                # Query the embedding table to find models used for this memory
                try:
                    cur.execute(f"""
                        SELECT DISTINCT embedding_model
                        FROM {table_name}
                        WHERE memory_id = %s;
                    """, (memory_id,))
                    models = [row['embedding_model'] for row in cur.fetchall()]
                    
                    if models:
                        new_structure[table_name] = models
                    else:
                        # Table entry exists in state but no embeddings found
                        # This could happen if embeddings were deleted but state wasn't updated
                        logger.debug(f"   No embeddings found in {table_name} for memory #{memory_id}")
                except Exception as e:
                    logger.warning(f"   Error querying {table_name}: {e}")
                    continue
            
            # Update memory with new V3 structure
            cur.execute("""
                UPDATE memories
                SET state = jsonb_set(
                    COALESCE(state, '{}'::jsonb),
                    '{embedding_tables}',
                    %s::jsonb,
                    true
                )
                WHERE id = %s;
            """, (psycopg2.extras.Json(new_structure), memory_id))
            
            migrated_count += 1
            
            if migrated_count % 100 == 0:
                conn.commit()
                logger.info(f"   Migrated {migrated_count}/{len(memories_to_migrate)} memories...")
        
        conn.commit()
        
        # Update system state to V3
        set_system_state(db_version=CURRENT_DB_VERSION)
        
        logger.info(f"🎉 V2 → V3 migration complete! Migrated {migrated_count} memories")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ V2 → V3 migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def migrate_v3_to_v4() -> None:
    """
    Migrate from V3 (ivfflat indexes) to V4 (HNSW indexes).
    
    HNSW indexes support unlimited dimensions and offer better query performance
    compared to ivfflat which has a 2000-dimension hard limit.
    
    This migration:
    1. Finds all existing memory_{dims} tables
    2. Drops old ivfflat indexes
    3. Creates new HNSW indexes
    """
    logger.info("🔄 Starting V3 → V4 migration (ivfflat → HNSW indexes)...")
    
    # Check if already at V4
    system_state = get_system_state()
    if system_state and system_state.get('db_version', 0) >= 4:
        logger.info("✅ Already at V4, skipping migration")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Find all existing memory_{dims} tables
        existing_tables = get_existing_embedding_tables()
        
        if not existing_tables:
            logger.info("📭 No embedding tables found to migrate")
            set_system_state(db_version=CURRENT_DB_VERSION)
            return
        
        logger.info(f"📋 Found {len(existing_tables)} embedding tables to migrate: {existing_tables}")
        
        for table_name in existing_tables:
            # Extract dimension number from table name (e.g., memory_768 → 768)
            dims = table_name.replace('memory_', '')
            
            logger.info(f"📊 Migrating {table_name} index to HNSW...")
            
            # Drop old ivfflat index
            cur.execute(f"DROP INDEX IF EXISTS idx_embedding_{dims};")
            
            # Create new HNSW index
            cur.execute(f"""
                CREATE INDEX idx_embedding_{dims} 
                ON {table_name} USING hnsw (embedding vector_cosine_ops);
            """)
            
            logger.info(f"✅ Migrated {table_name} to HNSW index")
        
        conn.commit()
        
        # Update schema version to V4
        set_system_state(db_version=CURRENT_DB_VERSION)
        
        logger.info("🎉 V3 → V4 migration complete!")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ V3 → V4 migration failed: {e}")
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
            
            # Run migrations in sequence
            if current_version == 1:
                # V1 → V2 migration (split tables)
                migrate_v1_to_v2(embedding_dim)
                # After V1→V2, run V2→V3 as well
                migrate_v2_to_v3()
                # After V2→V3, run V3→V4 as well
                migrate_v3_to_v4()
            elif current_version == 2:
                # V2 → V3 migration (embedding_tables array → object)
                migrate_v2_to_v3()
                # After V2→V3, run V3→V4 as well
                migrate_v3_to_v4()
            elif current_version == 3:
                # V3 → V4 migration (ivfflat → HNSW indexes)
                migrate_v3_to_v4()
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
