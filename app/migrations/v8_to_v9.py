"""
V8 → V9 Migration: Make content_id unique per namespace

V8 Schema: memories table with a plain index on (namespace, content_id)
V9 Schema: the same index, made UNIQUE

content_id is the user-facing memory number ("#812"). store_memory mints it
as MAX(content_id) + 1 for the namespace, and until now nothing stopped two
concurrent stores from reading the same MAX and inserting the same number.
resolve_memory_id then picks an arbitrary one of the pair, so get_memory,
delete_memory and set_classifiers on "#812" could act on the wrong row.

The race was latent: the blocking event loop ran every tool call one at a
time inside a process, so it needed two mmce processes sharing one database
and one namespace. store_memory now serialises minting with an advisory
lock, and this index is what makes a duplicate impossible rather than
unlikely.

Existing duplicates would make CREATE UNIQUE INDEX fail, and a failed
migration stops the server from starting. So they are renumbered first: in
each duplicated (namespace, content_id) group the oldest row (lowest id)
keeps its number, and the others get fresh numbers after the namespace's
current MAX. Every change is logged with the real id, because a memory's
number changes and whoever referred to it by the old number needs to know.
Backlinks in state.related store real ids, so they are unaffected.

This migration:
1. Renumbers any duplicate (namespace, content_id) rows
2. Replaces idx_memories_namespace_content_id with a UNIQUE index of the
   same name and columns - the MAX lookup keeps its index, and no second
   index is added
"""

import logging

from app.database import get_db_connection, table_exists, set_system_state

logger = logging.getLogger(__name__)


def migrate_v8_to_v9() -> None:
    """
    Migrate from V8 to V9: Unique content_id per namespace.

    The renumbering and the index swap run in one transaction, with the
    memories table locked against writes, so no store can slip a new
    duplicate in between the check and the index build.
    """
    logger.info("🔄 Starting V8 → V9 migration (unique content_id per namespace)...")

    # Check if memories table exists
    if not table_exists('memories'):
        logger.info("📭 No memories table found - will be created with V9 schema on init")
        set_system_state(db_version=9)
        logger.info("🎉 V8 → V9 migration complete (no memories table)!")
        return

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Block writers (another mmce process on the same database) until commit.
        # Reads still work.
        cur.execute("LOCK TABLE memories IN SHARE ROW EXCLUSIVE MODE;")

        # Check if the index is already unique
        cur.execute("""
            SELECT i.indisunique
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            WHERE c.relname = 'idx_memories_namespace_content_id';
        """)
        row = cur.fetchone()
        if row is not None and row[0]:
            conn.rollback()
            logger.info("✅ content_id index is already unique, skipping migration")
            set_system_state(db_version=9)
            return

        # Step 1: Renumber duplicates, keeping the oldest row's number
        cur.execute("""
            SELECT id, namespace, content_id
            FROM (
                SELECT id, namespace, content_id,
                       row_number() OVER (PARTITION BY namespace, content_id ORDER BY id) AS rn
                FROM memories
            ) ranked
            WHERE rn > 1
            ORDER BY namespace, id;
        """)
        duplicates = cur.fetchall()

        if duplicates:
            logger.warning(
                f"⚠️ Found {len(duplicates)} memories sharing a content_id with an older memory - renumbering them"
            )
            next_by_namespace = {}
            for memory_id, namespace, old_content_id in duplicates:
                if namespace not in next_by_namespace:
                    cur.execute(
                        "SELECT COALESCE(MAX(content_id), 0) + 1 FROM memories WHERE namespace = %s;",
                        (namespace,)
                    )
                    next_by_namespace[namespace] = cur.fetchone()[0]
                new_content_id = next_by_namespace[namespace]
                next_by_namespace[namespace] += 1

                cur.execute(
                    "UPDATE memories SET content_id = %s WHERE id = %s;",
                    (new_content_id, memory_id)
                )
                logger.warning(
                    f"⚠️ Renumbered memory id={memory_id} in namespace '{namespace}': "
                    f"#{old_content_id} → #{new_content_id}"
                )
        else:
            logger.info("📋 No duplicate content_ids found")

        # Step 2: Swap the plain index for a unique one with the same name and columns
        logger.info("📋 Replacing content_id index with a UNIQUE index...")
        cur.execute("DROP INDEX IF EXISTS idx_memories_namespace_content_id;")
        cur.execute("""
            CREATE UNIQUE INDEX idx_memories_namespace_content_id
            ON memories(namespace, content_id DESC);
        """)

        conn.commit()
        logger.info("✅ Schema changes committed")

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ V8 → V9 migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    # Update db_version to 9
    set_system_state(db_version=9)

    logger.info("🎉 V8 → V9 migration complete!")
