"""
V6 → V7 Migration: Add label_tokens table for trending labels feature

V6 Schema: memories table with content_id
V7 Schema: Add label_tokens table for tracking label token usage

The label_tokens table tracks which label tokens are being used recently
to help AI buddies understand what topics are hot (trending_labels feature).

This migration:
1. Creates label_tokens table with composite primary key (namespace, token)
2. Creates indexes for namespace filtering and time-based queries
"""

import logging

from app.database import get_db_connection, table_exists, set_system_state

logger = logging.getLogger(__name__)


def migrate_v6_to_v7() -> None:
    """
    Migrate from V6 to V7: Add label_tokens table for trending labels.
    
    The label_tokens table provides:
    - namespace: VARCHAR(100) DEFAULT 'default' - matches memories table namespace
    - token: VARCHAR(255) NOT NULL - the tokenized label fragment
    - count: INTEGER DEFAULT 0 - usage count
    - last_seen: TIMESTAMP DEFAULT NOW() - when token was last used
    - last_decay: TIMESTAMP DEFAULT NOW() - when decay was last run for this namespace
    - PRIMARY KEY (namespace, token) - composite key for namespace isolation
    """
    logger.info("🔄 Starting V6 → V7 migration (label_tokens table for trending labels)...")
    
    # Check if label_tokens table already exists
    if table_exists('label_tokens'):
        logger.info("✅ label_tokens table already exists, skipping migration")
        set_system_state(db_version=7)
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Create label_tokens table
        logger.info("📋 Creating label_tokens table...")
        cur.execute("""
            CREATE TABLE label_tokens (
                namespace VARCHAR(100) DEFAULT 'default',
                token VARCHAR(255) NOT NULL,
                count INTEGER DEFAULT 0,
                last_seen TIMESTAMP DEFAULT NOW(),
                last_decay TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (namespace, token)
            );
        """)
        
        # Create index on namespace for fast namespace filtering
        logger.info("📋 Creating index on namespace...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_label_tokens_namespace 
            ON label_tokens(namespace);
        """)
        
        # Create index on last_seen for time-based queries
        logger.info("📋 Creating index on last_seen...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_label_tokens_last_seen 
            ON label_tokens(last_seen);
        """)
        
        conn.commit()
        logger.info("✅ label_tokens table created with indexes")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ V6 → V7 migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()
    
    # Update db_version to 7
    set_system_state(db_version=7)
    
    logger.info("🎉 V6 → V7 migration complete!")
