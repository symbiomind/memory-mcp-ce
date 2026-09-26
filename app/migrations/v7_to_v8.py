"""
V7 → V8 Migration: Add classifiers column for caller-owned categorical scores

V7 Schema: memories table with label_tokens for trending labels
V8 Schema: memories table gains a classifiers JSONB column

The classifiers column holds a flat map of scalar values produced by an
external classifier (a calibrated decision model, a script, or an AI buddy).
memory-mcp-ce never interprets the contents - it stores scalars and compares
them when filtering. Populating the column is the caller's job.

Unlike labels, which match fuzzily (substring ILIKE) on purpose, classifier
keys are exact. That is why they need their own column instead of living in
labels, where "!class" would fuzz-match "ClassGreeting".

NULL is a meaningful value here:
- NULL  = never classified
- '{}'  = classifier ran, produced nothing
- {...} = classified

So there is no backfill. Existing memories stay NULL, which is the honest
state, and callers can find unclassified rows by filtering on a missing key.

This migration:
1. Adds classifiers JSONB column to memories table (nullable, no default)
2. Creates a GIN index for key existence and containment queries
"""

import logging

from app.database import get_db_connection, table_exists, set_system_state

logger = logging.getLogger(__name__)


def migrate_v7_to_v8() -> None:
    """
    Migrate from V7 to V8: Add classifiers column to memories.

    Adding the column without a DEFAULT is instant on PostgreSQL 11+ - no
    table rewrite - and leaves existing rows NULL rather than pretending
    they were classified.
    """
    logger.info("🔄 Starting V7 → V8 migration (classifiers column)...")

    # Check if memories table exists
    if not table_exists('memories'):
        logger.info("📭 No memories table found - will be created with V8 schema on init")
        set_system_state(db_version=8)
        logger.info("🎉 V7 → V8 migration complete (no memories table)!")
        return

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Check if classifiers column already exists
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'memories' AND column_name = 'classifiers';
        """)
        has_classifiers = cur.fetchone() is not None

        if has_classifiers:
            logger.info("✅ classifiers column already exists, skipping migration")
            set_system_state(db_version=8)
            return

        # Step 1: Add classifiers column (nullable, no default - no rewrite, no backfill)
        logger.info("📋 Adding classifiers column to memories table...")
        cur.execute("""
            ALTER TABLE memories ADD COLUMN classifiers JSONB;
        """)

        # Step 2: Create GIN index for key existence and containment queries
        # Range comparisons on individual keys are not served by this index -
        # they filter the candidate set, which is what the semantic path hands them.
        logger.info("📋 Creating GIN index on classifiers...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_classifiers 
            ON memories USING GIN(classifiers);
        """)

        conn.commit()
        logger.info("✅ Schema changes committed")

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ V7 → V8 migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    # Update db_version to 8
    set_system_state(db_version=8)

    logger.info("🎉 V7 → V8 migration complete!")
