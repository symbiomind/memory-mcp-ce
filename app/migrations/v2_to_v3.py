"""
V2 → V3 Migration: Embedding Tables Object Structure

V2 Structure: {"embedding_tables": ["memory_384", "memory_768"]}
V3 Structure: {"embedding_tables": {"memory_384": ["model1"], "memory_768": ["model2"]}}

This migration:
1. Finds all memories with V2 array format in state.embedding_tables
2. For each memory, queries embedding tables to find actual models used
3. Converts to V3 object format with model arrays
"""

import psycopg2
import psycopg2.extras
import logging

from app.database import (
    get_db_connection,
    table_exists,
    set_system_state,
)
from app.migrations.runner import CURRENT_DB_VERSION

logger = logging.getLogger(__name__)


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
