"""
Migration Runner

Handles version checking and sequential execution of database migrations.
"""

import logging

logger = logging.getLogger(__name__)

# Current database schema version
CURRENT_DB_VERSION = 4


def run_migrations(embedding_dim: int) -> None:
    """
    Check database version and run any required migrations in sequence.
    
    This function:
    1. Gets current database version from system_state
    2. Runs migrations sequentially from current version to CURRENT_DB_VERSION
    3. Handles fresh installs and partial migrations
    
    Args:
        embedding_dim: The embedding dimension (needed for V1→V2 migration)
    """
    # Import here to avoid circular imports
    from app.database import (
        get_system_state,
        get_existing_embedding_tables,
        create_system_state_table,
        create_memories_table,
        get_db_connection,
        table_exists,
    )
    from app.migrations.v1_to_v2 import migrate_v1_to_v2, is_v1_schema
    from app.migrations.v2_to_v3 import migrate_v2_to_v3
    from app.migrations.v3_to_v4 import migrate_v3_to_v4
    
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
            logger.info("🆕 Fresh installation detected - creating V4 schema")
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
