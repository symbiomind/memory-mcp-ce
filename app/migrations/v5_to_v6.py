"""
V5 → V6 Migration: Add content_id for namespace-scoped memory numbering

V5 Schema: memories table with id as primary key
V6 Schema: memories table with content_id for per-namespace sequential numbering

The content_id provides clean sequential numbering per namespace:
- Each namespace gets its own 1, 2, 3... sequence
- Enables multi-tenant CE deployments and future SaaS
- Users see content_id as their "memory number"
- Internal id remains for FK relationships

This migration:
1. Adds content_id BIGINT column to memories table
2. Populates existing memories with content_id = id (preserves references)
3. Makes content_id NOT NULL
4. Creates index for efficient MAX queries per namespace
"""

import logging

from app.database import get_db_connection, table_exists, set_system_state

logger = logging.getLogger(__name__)


def migrate_v5_to_v6() -> None:
    """
    Migrate from V5 to V6: Add content_id column for namespace-scoped numbering.
    
    The content_id column provides sequential numbering per namespace,
    independent of the global auto-increment id.
    """
    logger.info("🔄 Starting V5 → V6 migration (content_id for namespace-scoped numbering)...")
    
    # Check if memories table exists
    if not table_exists('memories'):
        logger.info("📭 No memories table found - will be created with V6 schema on init")
        set_system_state(db_version=6)
        logger.info("🎉 V5 → V6 migration complete (no memories table)!")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Check if content_id column already exists
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'memories' AND column_name = 'content_id';
        """)
        has_content_id = cur.fetchone() is not None
        
        if has_content_id:
            logger.info("✅ content_id column already exists, skipping migration")
            set_system_state(db_version=6)
            return
        
        # Step 1: Add content_id column (nullable initially)
        logger.info("📋 Adding content_id column to memories table...")
        cur.execute("""
            ALTER TABLE memories ADD COLUMN content_id BIGINT;
        """)
        
        # Step 2: Populate existing memories with content_id = id
        # This preserves existing memory references - no renumbering
        logger.info("📋 Populating content_id for existing memories (content_id = id)...")
        cur.execute("""
            UPDATE memories SET content_id = id;
        """)
        rows_updated = cur.rowcount
        logger.info(f"📋 Updated {rows_updated} memories with content_id = id")
        
        # Step 3: Make content_id NOT NULL
        logger.info("📋 Setting content_id to NOT NULL...")
        cur.execute("""
            ALTER TABLE memories ALTER COLUMN content_id SET NOT NULL;
        """)
        
        # Step 4: Create index for efficient MAX queries per namespace
        logger.info("📋 Creating index for namespace + content_id...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_namespace_content_id 
            ON memories(namespace, content_id DESC);
        """)
        
        conn.commit()
        logger.info("✅ Schema changes committed")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ V5 → V6 migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()
    
    # Update db_version to 6
    set_system_state(db_version=6)
    
    logger.info("🎉 V5 → V6 migration complete!")
