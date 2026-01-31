"""
Migration Runner

Handles version checking and sequential execution of database migrations.
GRAVY Pattern: Fresh installs get latest schema, upgrades get incremental migrations.

Uses PostgreSQL advisory locks to prevent concurrent migrations across multiple
workers or containers.
"""

import logging

logger = logging.getLogger(__name__)

# Current database schema version
CURRENT_DB_VERSION = 7

# Advisory lock ID for migrations (unique arbitrary number)
MIGRATION_LOCK_ID = 123456789


def run_migrations(embedding_dim: int) -> None:
    """
    Check database version and run any required migrations in sequence.
    
    This function:
    1. Acquires PostgreSQL advisory lock to prevent concurrent migrations
    2. Gets current database version from system_state
    3. Runs migrations sequentially from current version to CURRENT_DB_VERSION
    4. Handles fresh installs and partial migrations
    5. Releases advisory lock
    
    GRAVY Pattern: Fresh installs skip migrations and create latest schema.
    
    Advisory Lock: Only one process runs migrations at a time. Others skip gracefully.
    
    Args:
        embedding_dim: The embedding dimension (needed for V1→V2 migration)
    """
    # Import here to avoid circular imports
    from app.database import (
        get_system_state,
        get_existing_embedding_tables,
        create_system_state_table,
        create_memories_table,
        create_label_tokens_table,
        get_db_connection,
        table_exists,
    )
    from app.migrations.v1_to_v2 import migrate_v1_to_v2, is_v1_schema
    from app.migrations.v2_to_v3 import migrate_v2_to_v3
    from app.migrations.v3_to_v4 import migrate_v3_to_v4
    from app.migrations.v4_to_v5 import migrate_v4_to_v5
    from app.migrations.v5_to_v6 import migrate_v5_to_v6
    from app.migrations.v6_to_v7 import migrate_v6_to_v7
    
    # Get connection for advisory lock
    conn = get_db_connection()
    cur = conn.cursor()
    lock_acquired = False
    
    try:
        # Try to acquire advisory lock (non-blocking)
        cur.execute("SELECT pg_try_advisory_lock(%s);", (MIGRATION_LOCK_ID,))
        lock_acquired = cur.fetchone()[0]
        
        if not lock_acquired:
            logger.info("⏳ Another process is running migrations, skipping...")
            return
        
        logger.info("🔒 Acquired migration lock")
        
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
                    # After V1→V2, continue with remaining migrations
                    migrate_v2_to_v3()
                    migrate_v3_to_v4()
                    migrate_v4_to_v5()
                    migrate_v5_to_v6()
                    migrate_v6_to_v7()
                else:
                    # Tables exist but already V2+ schema (partial migration?)
                    # Run v4→v5 to create fresh V5 system_state, then continue
                    logger.info("🔍 Tables exist with V2+ schema - creating V5 system_state")
                    migrate_v4_to_v5()
                    migrate_v5_to_v6()
                    migrate_v6_to_v7()
            else:
                # Fresh installation - create V7 schema from scratch
                logger.info("🆕 Fresh installation detected - creating V7 schema")
                create_system_state_table()
                create_memories_table()
                create_label_tokens_table()
        
        else:
            # system_state exists - check version
            current_version = system_state.get('db_version', 1)
            
            if current_version < CURRENT_DB_VERSION:
                logger.info(f"🔍 Database version {current_version} < {CURRENT_DB_VERSION} - migration required")
                
                # Run migrations in sequence
                if current_version == 1:
                    # V1 → V2 migration (split tables)
                    migrate_v1_to_v2(embedding_dim)
                    current_version = 2
                
                if current_version == 2:
                    # V2 → V3 migration (embedding_tables array → object)
                    migrate_v2_to_v3()
                    current_version = 3
                
                if current_version == 3:
                    # V3 → V4 migration (ivfflat → HNSW indexes)
                    migrate_v3_to_v4()
                    current_version = 4
                
                if current_version == 4:
                    # V4 → V5 migration (key-value system_state)
                    migrate_v4_to_v5()
                    current_version = 5
                
                if current_version == 5:
                    # V5 → V6 migration (content_id for namespace-scoped numbering)
                    migrate_v5_to_v6()
                    current_version = 6
                
                if current_version == 6:
                    # V6 → V7 migration (label_tokens table for trending labels)
                    migrate_v6_to_v7()
                    current_version = 7
            else:
                logger.info(f"✅ Database schema is up to date (version {current_version})")
        
        # Ensure memories table exists (idempotent)
        if not table_exists('memories'):
            create_memories_table()
        
        # Ensure label_tokens table exists (idempotent)
        if not table_exists('label_tokens'):
            create_label_tokens_table()
    
    finally:
        # Release advisory lock if we acquired it
        if lock_acquired:
            cur.execute("SELECT pg_advisory_unlock(%s);", (MIGRATION_LOCK_ID,))
            logger.info("🔓 Released migration lock")
        cur.close()
        conn.close()
