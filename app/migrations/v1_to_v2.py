"""
V1 → V2 Migration: Split Table Architecture

V1 Schema: Single memory_{dims} table with content + embedding
V2 Schema: Separate memories table (content) + memory_{dims} tables (embeddings with FK)

This migration:
1. Creates the new memories and system_state tables
2. Extracts content from existing memory_{dims} tables into memories
3. Rebuilds embedding tables with foreign keys
4. Updates state.embedding_tables for each memory
"""

import psycopg2
import psycopg2.extras
import hashlib
import logging

from app.database import (
    get_db_connection,
    get_existing_embedding_tables,
    create_system_state_table,
    create_memories_table,
    create_embedding_table,
    set_system_state,
)
from app.migrations.runner import CURRENT_DB_VERSION

logger = logging.getLogger(__name__)


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
