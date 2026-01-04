"""
V3 → V4 Migration: HNSW Index Architecture

V3 Schema: ivfflat indexes (2000-dimension limit)
V4 Schema: HNSW indexes (unlimited dimensions)

HNSW indexes support unlimited dimensions and offer better query performance
compared to ivfflat which has a 2000-dimension hard limit.

This migration:
1. Finds all existing memory_{dims} tables
2. Drops old ivfflat indexes
3. Creates new HNSW indexes
"""

import logging

from app.database import (
    get_db_connection,
    get_existing_embedding_tables,
    get_system_state,
    set_system_state,
)
from app.migrations.runner import CURRENT_DB_VERSION

logger = logging.getLogger(__name__)


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
