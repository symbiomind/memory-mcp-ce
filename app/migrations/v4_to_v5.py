"""
V4 → V5 Migration: Key-Value System State

V4 Schema: Fixed-column system_state table (id, db_version, jwt_state, etc.)
V5 Schema: Flexible key-value system_state table

The new key-value design allows adding system-wide settings without schema changes.

This migration:
1. Reads current db_version from old table
2. Drops the old system_state table
3. Creates new key-value system_state table
4. Inserts db_version as a key-value pair
"""

import logging
import psycopg2.extras

from app.database import get_db_connection, table_exists

logger = logging.getLogger(__name__)


def migrate_v4_to_v5() -> None:
    """
    Migrate from V4 (fixed-column system_state) to V5 (key-value system_state).
    
    The new schema:
    CREATE TABLE system_state (
        id SERIAL PRIMARY KEY,
        key TEXT UNIQUE NOT NULL,
        value JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    logger.info("🔄 Starting V4 → V5 migration (key-value system_state)...")
    
    # Check if system_state exists
    if not table_exists('system_state'):
        logger.info("📭 No system_state table found - creating fresh V5 schema")
        _create_v5_system_state_table()
        _insert_db_version(5)
        logger.info("🎉 V4 → V5 migration complete (fresh install)!")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Check if already V5 schema (has 'key' column)
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'system_state' AND column_name = 'key';
        """)
        is_v5_schema = cur.fetchone() is not None
        
        if is_v5_schema:
            # Already V5 schema - just ensure version is set correctly
            cur.execute("SELECT value FROM system_state WHERE key = 'db_version';")
            row = cur.fetchone()
            if row:
                current_version = row[0] if isinstance(row[0], int) else int(str(row[0]).strip('"'))
                if current_version >= 5:
                    logger.info("✅ Already at V5, skipping migration")
                else:
                    # Update version to 5
                    _insert_db_version(5)
                    logger.info("🎉 V5 schema exists, updated version to 5")
            else:
                # No db_version key - insert it
                _insert_db_version(5)
                logger.info("🎉 V5 schema exists, inserted db_version = 5")
            return
        
        # V4 schema - read current db_version
        cur.execute("SELECT db_version FROM system_state WHERE id = 1;")
        row = cur.fetchone()
        old_db_version = row[0] if row else 4
        
        logger.info(f"📋 Current db_version: {old_db_version}")
        
        # Drop old table
        cur.execute("DROP TABLE system_state;")
        logger.info("🗑️ Dropped old system_state table")
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ V4 → V5 migration failed during read/drop: {e}")
        raise
    finally:
        cur.close()
        conn.close()
    
    # Create new V5 schema
    _create_v5_system_state_table()
    
    # Insert db_version as key-value (set to 5, the new version)
    _insert_db_version(5)
    
    logger.info("🎉 V4 → V5 migration complete!")


def _create_v5_system_state_table() -> None:
    """Create the V5 key-value system_state table."""
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
        conn.commit()
        logger.info("✅ Created V5 system_state table")
    finally:
        cur.close()
        conn.close()


def _insert_db_version(version: int) -> None:
    """Insert db_version as a key-value pair."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        json_value = psycopg2.extras.Json(version)
        cur.execute("""
            INSERT INTO system_state (key, value) 
            VALUES ('db_version', %s)
            ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = CURRENT_TIMESTAMP;
        """, (json_value, json_value))
        conn.commit()
        logger.info(f"✅ Set db_version = {version}")
    finally:
        cur.close()
        conn.close()
