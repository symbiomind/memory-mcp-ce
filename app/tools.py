
from app.database import get_db_connection, add_embedding_to_state, get_memory_embedding_tables
from app.embedding import get_embedding, get_embedding_dimension
from app.utils import tokenize_labels
import psycopg2.extras
from app.config import EMBEDDING_MODEL, NAMESPACE, TIMEZONE, TIMEZONE_DISABLED, PERFORMANCE_METRICS
import time
from app.encryption import (
    encrypt_content,
    is_encryption_enabled,
    decode_or_decrypt_content,
    should_include_memory,
)
import json
import logging
from typing import List, Any
from datetime import datetime, timezone

# Get logger
logger = logging.getLogger(__name__)


def is_wildcard_namespace() -> bool:
    """
    Check if the current namespace is wildcard (empty/unset).
    
    Wildcard namespace means NAMESPACE env var is empty string or not set.
    In this case, users work with real database IDs directly.
    
    Returns:
        True if wildcard (no namespace restriction), False if namespaced
    """
    return NAMESPACE is None or NAMESPACE == ""


def resolve_memory_id(input_id: int, namespace: str) -> tuple[int | None, str | None]:
    """
    Resolve user-facing ID to real database ID based on namespace mode.
    
    - Wildcard namespace: input_id IS the real database ID
    - Specific namespace: input_id is content_id, need to lookup real ID
    
    Args:
        input_id: The ID provided by the user
        namespace: The namespace to search in
        
    Returns:
        Tuple of (real_db_id, error_message)
        - Success: (id, None)
        - Not found: (None, error_message)
    """
    if is_wildcard_namespace():
        # Wildcard mode - input_id is the real database ID
        return input_id, None
    
    # Namespaced mode - input_id is content_id, need to find real ID
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id FROM memories WHERE content_id = %s AND namespace = %s;",
            (input_id, namespace)
        )
        result = cur.fetchone()
        if result:
            return result[0], None
        else:
            return None, f"❌ Memory #{input_id} not found in namespace"
    finally:
        cur.close()
        conn.close()


def get_display_id(real_id: int, content_id: int) -> int:
    """
    Get the ID to display to the user based on namespace mode.
    
    - Wildcard namespace: show real database ID
    - Specific namespace: show content_id (namespace-scoped sequential ID)
    
    Args:
        real_id: The actual database row ID
        content_id: The namespace-scoped sequential ID
        
    Returns:
        The appropriate ID to display to the user
    """
    if is_wildcard_namespace():
        return real_id
    return content_id


def get_ordinal_suffix(day: int) -> str:
    """
    Get the ordinal suffix for a day number (st, nd, rd, th).
    
    Args:
        day: Day of the month (1-31)
    
    Returns:
        Ordinal suffix string
    """
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def format_current_time() -> tuple[str, str] | tuple[None, None]:
    """
    Format current time in human-readable format with timezone info.
    
    Returns:
        Tuple of (formatted_time, timezone_string) or (None, None) if disabled
    """
    if TIMEZONE_DISABLED or TIMEZONE is None:
        return None, None
    
    # Get current time in configured timezone
    now = datetime.now(TIMEZONE)
    
    # Get timezone abbreviation (handles DST automatically)
    tz_abbrev = now.strftime("%Z")
    
    # Get the timezone string from the environment (for the response)
    import os
    tz_string = os.getenv("TIMEZONE", "").strip()
    if not tz_string or tz_string.lower() == "false":
        tz_string = "UTC"
    
    # Format: "Friday, 2nd January 2026 - 04:07 PM ACDT"
    day = now.day
    ordinal = get_ordinal_suffix(day)
    formatted = now.strftime(f"%A, {day}{ordinal} %B %Y - %I:%M %p {tz_abbrev}")
    
    return formatted, tz_string


def add_timezone_to_response(response: dict) -> dict:
    """
    Add current_time and timezone fields to a response dict.
    
    Args:
        response: The response dictionary to augment
    
    Returns:
        Response dict with timezone info prepended (if enabled)
    """
    current_time, tz_string = format_current_time()
    
    if current_time is None:
        # Timezone feature disabled - return response unchanged
        return response
    
    # Create new dict with timezone fields first, then original response
    return {
        "current_time": current_time,
        "timezone": tz_string,
        **response
    }


def format_performance(embedding_time: float, db_time: float, total_time: float) -> str:
    """
    Format performance timing as a space-separated string.
    
    Args:
        embedding_time: Time spent on embedding API call (seconds)
        db_time: Time spent on database operations (seconds)
        total_time: Total function execution time (seconds)
    
    Returns:
        String in format "0.750 0.130 1.070" (3 decimal places each)
    """
    return f"{embedding_time:.3f} {db_time:.3f} {total_time:.3f}"


def add_performance_to_response(
    response: dict,
    embedding_time: float,
    db_time: float,
    total_time: float
) -> dict:
    """
    Add performance timing to a response dict (if PERFORMANCE_METRICS enabled).
    
    Args:
        response: The response dictionary to augment
        embedding_time: Time spent on embedding API call (seconds)
        db_time: Time spent on database operations (seconds)
        total_time: Total function execution time (seconds)
    
    Returns:
        Response dict with performance field added (if enabled)
    """
    if not PERFORMANCE_METRICS:
        return response
    
    # Add performance field to response
    response["performance"] = format_performance(embedding_time, db_time, total_time)
    return response


def format_time_ago(timestamp_str: str) -> str:
    """
    Convert an ISO timestamp to a human-readable 'time ago' format.
    
    Args:
        timestamp_str: ISO format timestamp string
    
    Returns:
        Human-readable string like "2 hours ago", "1 day ago", etc.
    """
    try:
        # Parse the timestamp (handle both with and without timezone)
        if timestamp_str.endswith('Z'):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        elif '+' in timestamp_str or timestamp_str.count('-') > 2:
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            # No timezone info, assume UTC
            timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
        
        # Get current time in UTC
        now = datetime.now(timezone.utc)
        
        # Calculate difference
        diff = now - timestamp
        
        # Convert to total seconds
        total_seconds = diff.total_seconds()
        
        # Handle future timestamps (shouldn't happen, but just in case)
        if total_seconds < 0:
            return "just now"
        
        # Less than 1 minute
        if total_seconds < 60:
            return "less than 1 minute ago"
        
        # Minutes
        minutes = int(total_seconds / 60)
        if minutes < 60:
            if minutes == 1:
                return "1 minute ago"
            return f"{minutes} minutes ago"
        
        # Hours
        hours = int(total_seconds / 3600)
        if hours < 24:
            if hours == 1:
                return "1 hour ago"
            return f"{hours} hours ago"
        
        # Days
        days = int(total_seconds / 86400)
        if days < 7:
            if days == 1:
                return "1 day ago"
            return f"{days} days ago"
        
        # Weeks
        weeks = int(days / 7)
        if weeks < 4:
            if weeks == 1:
                return "1 week ago"
            return f"{weeks} weeks ago"
        
        # Months (approximate: 30.44 days average)
        months = round(days / 30.44)
        if months < 12:
            if months == 1:
                return "1 month ago"
            return f"{months} months ago"
        
        # Years
        years = int(days / 365.25)
        if years == 1:
            return "1 year ago"
        return f"{years} years ago"
        
    except Exception as e:
        logger.warning(f"Error formatting time ago: {e}")
        return "unknown"

def normalize_labels(labels_value: Any) -> List[str]:
    """
    Normalize labels from various formats to a list of strings.
    Accepts: list, comma-separated string, or None
    Returns: list of cleaned label strings
    """
    if labels_value is None:
        return []
    
    if isinstance(labels_value, list):
        # Already a list - clean up each item
        return [str(label).strip() for label in labels_value if label]
    
    if isinstance(labels_value, str):
        # Comma-separated string - split and clean
        return [label.strip() for label in labels_value.split(',') if label.strip()]
    
    # Unknown type - return empty list
    logger.warning(f"⚠️ Unexpected label type: {type(labels_value)}, returning empty list")
    return []


def parse_labels_with_exclusions(labels: str | None) -> tuple[List[str], List[str]]:
    """
    Parse comma-separated labels into include and exclude lists.
    Labels starting with '!' are exclusions (fuzzy NOT match).
    
    Args:
        labels: Comma-separated labels string (e.g., "beer,!wine,ale,!cider")
    
    Returns:
        Tuple of (include_labels, exclude_labels)
        Example: "beer,!wine,ale" -> (["beer", "ale"], ["wine"])
    """
    if labels is None or not isinstance(labels, str):
        return [], []
    
    include_labels = []
    exclude_labels = []
    
    for label in labels.split(','):
        label = label.strip()
        if not label:
            continue
        if label.startswith('!'):
            # Exclusion - strip the ! prefix
            exclude_label = label[1:].strip()
            if exclude_label:
                exclude_labels.append(exclude_label)
        else:
            include_labels.append(label)
    
    return include_labels, exclude_labels


def parse_source_with_exclusion(source: str | None) -> tuple[str | None, bool]:
    """
    Parse source parameter for exclusion prefix.
    Source starting with '!' means exclude (fuzzy NOT match).
    
    Args:
        source: Source string, optionally prefixed with '!'
    
    Returns:
        Tuple of (source_value, is_exclusion)
        Example: "!clawdbot" -> ("clawdbot", True)
        Example: "clawdbot" -> ("clawdbot", False)
    """
    if source is None or not isinstance(source, str):
        return None, False
    
    source = source.strip()
    if not source:
        return None, False
    
    if source.startswith('!'):
        # Exclusion - strip the ! prefix
        exclude_source = source[1:].strip()
        return exclude_source if exclude_source else None, True
    
    return source, False

def extract_json_params(param_value: str, required_key: str) -> tuple[str, str | None, str | None]:
    """
    Extract content, labels, and source from JSON-embedded parameter (Grok workaround).
    
    Args:
        param_value: The parameter value that might contain JSON
        required_key: The required key to extract (e.g., 'content' or 'query')
    
    Returns:
        Tuple of (extracted_value, labels_or_none, source_or_none)
        - If valid JSON with required key: returns (extracted_value, optional_labels, optional_source)
        - If not valid JSON or missing required key: returns (original_param_value, None, None)
    """
    # Trim whitespace
    trimmed = param_value.strip()
    
    # Check if it looks like JSON (starts with { and ends with })
    if not (trimmed.startswith('{') and trimmed.endswith('}')):
        return param_value, None, None
    
    # Try to parse as JSON
    try:
        parsed = json.loads(trimmed)
        
        # Must be a dict
        if not isinstance(parsed, dict):
            return param_value, None, None
        
        # Check for required key
        if required_key not in parsed:
            return param_value, None, None
        
        # Extract the required value, optional labels, AND optional source
        extracted_value = parsed[required_key]
        extracted_labels = parsed.get('labels')
        extracted_source = parsed.get('source')
        
        return extracted_value, extracted_labels, extracted_source
        
    except (json.JSONDecodeError, ValueError):
        # Not valid JSON, treat as normal string
        return param_value, None, None

def store_memory(content: str, labels: str = None, source: str = None, mcp_settings: dict = None) -> dict:
    """
    Stores a memory in the database with duplicate detection.
    
    V2 Split-Table Architecture:
    1. Insert content into memories table (source of truth)
    2. Generate embedding and insert into memory_{dims} table
    3. Update state.embedding_tables to track which tables have embeddings
    """
    # Performance timing
    total_start = time.time()
    embedding_time = 0.0
    db_time = 0.0
    
    # Extract JSON-embedded parameters (Grok workaround)
    extracted_content, extracted_labels, extracted_source = extract_json_params(content, 'content')
    
    # Use extracted values if found in JSON, otherwise use the original parameters
    if extracted_labels is not None:
        labels = extracted_labels
    if extracted_source is not None:
        source = extracted_source
    
    # Use extracted content
    content = extracted_content
    
    # Get MCP settings (passed as parameter or default to empty dict)
    settings = mcp_settings if mcp_settings is not None else {}
    logger.info(f"🔍 Debug - Settings received: {settings}")
    
    # Normalize labels from MCP call
    label_list = normalize_labels(labels)
    logger.info(f"🔍 Debug - Labels from MCP call (normalized): {label_list}")
    
    # Get labels to append from MCP-Settings header
    store_labels_append = settings.get('store_labels_append', [])
    logger.info(f"🔍 Debug - store_labels_append from settings: {store_labels_append} (type: {type(store_labels_append)})")
    
    append_labels = normalize_labels(store_labels_append)
    logger.info(f"🔍 Debug - Normalized append_labels: {append_labels}")
    
    # Merge labels: MCP call labels + header labels
    if append_labels:
        label_list.extend(append_labels)
        # Remove duplicates while preserving order
        seen = set()
        label_list = [x for x in label_list if not (x in seen or seen.add(x))]
        logger.info(f"🏷️ Labels merged: {label_list} (appended from header: {append_labels})")
    else:
        logger.info(f"🔍 Debug - No labels to append (append_labels is empty)")
    
    # Auto-populate from config
    embedding_model = EMBEDDING_MODEL
    namespace = NAMESPACE if NAMESPACE is not None else "default"

    # Generate embedding (timed)
    emb_start = time.time()
    embedding = get_embedding(content)
    embedding_time = time.time() - emb_start
    
    embedding_dim = len(embedding)
    table_name = f"memory_{embedding_dim}"

    # Database operations (timed)
    db_start = time.time()
    conn = get_db_connection()
    cur = conn.cursor()

    # Check for similar memories (duplicate detection) - query embedding table with JOIN to memories
    warnings = []
    try:
        check_sql = f"""
            SELECT m.id, m.content, m.enc, 1 - (e.embedding <=> %s::vector) as similarity, m.content_id
            FROM memories m
            JOIN {table_name} e ON m.id = e.memory_id
            WHERE e.namespace = %s AND e.embedding_model = %s
            ORDER BY similarity DESC
            LIMIT 2;
        """
        cur.execute(check_sql, (embedding, namespace, embedding_model))
        similar_memories = cur.fetchall()
        
        for row in similar_memories:
            mem_id, mem_content_bytes, mem_enc, similarity, mem_content_id = row
            # Safely decode or decrypt content for comparison
            mem_enc = mem_enc if mem_enc is not None else False
            mem_content = decode_or_decrypt_content(bytes(mem_content_bytes), mem_enc)
            if mem_content is None:
                # Skip encrypted memories we can't decrypt for duplicate detection
                continue
            if similarity >= 0.70:  # 70% threshold
                percentage = int(similarity * 100)
                # Get the correct display ID for the warning message
                display_id = get_display_id(mem_id, mem_content_id)
                # Tiered warning messages based on similarity level
                if similarity >= 1.0:
                    warnings.append(f"❌ Exact match with memory #{display_id}")
                elif similarity >= 0.91:
                    warnings.append(f"⚠️ Worth reviewing for context to memory #{display_id} ({percentage}% match)")
                elif similarity >= 0.81:
                    warnings.append(f"📌 Explores similar territory to memory #{display_id} ({percentage}% match)")
                else:  # 0.70 - 0.80
                    warnings.append(f"ℹ️ Semantically related to memory #{display_id} ({percentage}% match)")
    except Exception as e:
        # Table might not exist yet or be empty - that's OK for first memory
        logger.debug(f"Duplicate check skipped: {e}")

    # Determine if encryption is enabled and prepare content
    encrypted_blob = encrypt_content(content)
    if encrypted_blob is not None:
        # Encryption enabled - store encrypted content with enc=true
        content_bytes = encrypted_blob
        is_encrypted = True
        logger.info(f"🔐 Encrypting memory content ({len(content)} chars → {len(encrypted_blob)} bytes)")
    else:
        # No encryption - store plain UTF-8 with enc=false
        content_bytes = content.encode('utf-8')
        is_encrypted = False
    
    # V6: Get next content_id for this namespace (namespace-scoped sequential numbering)
    cur.execute(
        """SELECT COALESCE(MAX(content_id), 0) + 1 FROM memories WHERE namespace = %s;""",
        (namespace,)
    )
    next_content_id = cur.fetchone()[0]
    
    # V6: Step 1 - Insert into memories table (source of truth)
    # V3 structure: embedding_tables is an object mapping table names to model arrays
    cur.execute(
        """INSERT INTO memories (content_id, content, namespace, labels, source, enc, state)
        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;""",
        (
            next_content_id,
            content_bytes,
            namespace,
            psycopg2.extras.Json(label_list),
            source,
            is_encrypted,
            psycopg2.extras.Json({'embedding_tables': {table_name: [embedding_model]}})
        )
    )
    memory_id = cur.fetchone()[0]
    
    # V2: Step 2 - Insert embedding into memory_{dims} table
    cur.execute(
        f"""INSERT INTO {table_name} (memory_id, embedding, namespace, embedding_model)
        VALUES (%s, %s::vector, %s, %s);""",
        (memory_id, embedding, namespace, embedding_model)
    )

    conn.commit()
    cur.close()
    conn.close()
    db_time = time.time() - db_start

    # Track label token popularity (only if labels exist)
    # This runs AFTER memory is committed - token tracking failure won't affect memory storage
    if label_list:
        token_counts = tokenize_labels(label_list)
        
        if token_counts:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                
                # Batch upsert using unnest() - single query for all tokens
                tokens = list(token_counts.keys())
                counts = list(token_counts.values())
                namespaces = [namespace] * len(tokens)
                
                cur.execute("""
                    INSERT INTO label_tokens (namespace, token, count, last_seen)
                    SELECT unnest(%s::varchar[]), unnest(%s::varchar[]), unnest(%s::int[]), NOW()
                    ON CONFLICT (namespace, token) 
                    DO UPDATE SET 
                        count = label_tokens.count + EXCLUDED.count,
                        last_seen = NOW()
                """, (namespaces, tokens, counts))
                conn.commit()
                logger.debug(f"📊 Tracked {len(tokens)} label tokens for memory")
            except Exception as e:
                # Don't fail the whole operation - memory was already stored successfully
                logger.warning(f"⚠️ Failed to track label tokens: {e}")
            finally:
                cur.close()
                conn.close()

    # Display appropriate ID based on namespace mode
    display_id = get_display_id(memory_id, next_content_id)
    
    result = {
        "current_embedding": embedding_model,
        "id": display_id,
        "source": source,
        "message": f"✅ Memory stored with ID {display_id}" + (" 🔐" if is_encrypted else "")
    }
    
    if warnings:
        result["warnings"] = warnings
    
    # Add performance metrics and timezone
    total_time = time.time() - total_start
    result = add_timezone_to_response(result)
    return add_performance_to_response(result, embedding_time, db_time, total_time)

def retrieve_memories(query: str = None, labels: str = None, source: str = None, num_results: int = 5) -> dict:
    """
    Retrieve memories with flexible filtering combinations.
    
    V2 Split-Table Architecture:
    - With query: JOIN memories + embedding table for semantic search
    - Without query: Query memories table directly (source of truth)
    
    Supports 8 filtering modes:
    1. Query only: Semantic search using embeddings
    2. Labels only: Filter by labels, return most recent matches
    3. Source only: Filter by source, return most recent matches
    4. Query + Labels: Semantic search filtered by labels
    5. Query + Source: Semantic search filtered by source
    6. Labels + Source: Filter by both, return most recent
    7. Query + Labels + Source: Semantic search filtered by both labels AND source
    8. None (no parameters): Return most recent N memories ordered by timestamp DESC
    """
    # Performance timing
    total_start = time.time()
    embedding_time = 0.0
    db_time = 0.0
    
    # Extract JSON-embedded parameters (Grok workaround) - only if query is provided
    if query is not None and isinstance(query, str) and query.strip():
        extracted_query, extracted_labels, extracted_source = extract_json_params(query, 'query')
        
        # Use extracted values if found in JSON, otherwise use the original parameters
        if extracted_labels is not None:
            labels = extracted_labels
        if extracted_source is not None:
            source = extracted_source
        
        # Use extracted query
        query = extracted_query
    
    # Parse labels with exclusion support (! prefix)
    include_labels, exclude_labels = parse_labels_with_exclusions(labels)
    
    # Parse source with exclusion support (! prefix)
    source_value, source_is_exclusion = parse_source_with_exclusion(source)
    
    # No validation required - all parameters are optional
    # When no parameters provided, returns most recent memories
    
    # Auto-populate from config
    embedding_model = EMBEDDING_MODEL
    namespace = NAMESPACE if NAMESPACE is not None else "default"
    
    # Check if encryption key is available (affects which memories we can access)
    encryption_available = is_encryption_enabled()
    
    # Get embedding dimension for table name
    embedding_dim = get_embedding_dimension()
    table_name = f"memory_{embedding_dim}"
    
    # Determine which search path to use
    if query:
        # Path A: Semantic search - JOIN memories with embedding table
        # Time the embedding call
        emb_start = time.time()
        embedding = get_embedding(query)
        embedding_time = time.time() - emb_start
    
    # Database operations (timed)
    db_start = time.time()
    conn = get_db_connection()
    cur = conn.cursor()
    
    if query:
        
        # Build SQL query with JOIN to memories table
        sql = f"""
            SELECT m.id, m.content, e.embedding_model, m.namespace, m.labels, m.source, m.timestamp, 
                   1 - (e.embedding <=> %s::vector) as similarity, m.enc, m.state, m.content_id
            FROM memories m
            JOIN {table_name} e ON m.id = e.memory_id
        """
        
        params = [embedding]
        where_clauses = []
        
        # Filter by embedding model on embedding table
        where_clauses.append("e.embedding_model = %s")
        params.append(embedding_model)
        
        # Filter by namespace on embedding table (denormalized for performance)
        if namespace:
            where_clauses.append("e.namespace = %s")
            params.append(namespace)
        
        # If no encryption key, only return unencrypted memories
        if not encryption_available:
            where_clauses.append("m.enc = false")
        
        # Label filtering on memories table with fuzzy matching (include/exclude)
        # Include labels: fuzzy OR match
        if include_labels:
            include_conditions = []
            for label in include_labels:
                include_conditions.append(f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(m.labels) AS label WHERE label ILIKE %s)")
                params.append(f"%{label}%")
            where_clauses.append(f"({' OR '.join(include_conditions)})")
        
        # Exclude labels: fuzzy AND NOT match (each exclusion is separate)
        for label in exclude_labels:
            where_clauses.append(f"NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(m.labels) AS label WHERE label ILIKE %s)")
            params.append(f"%{label}%")
        
        # Source filtering on memories table (include or exclude)
        if source_value:
            if source_is_exclusion:
                where_clauses.append("NOT m.source ILIKE %s")
            else:
                where_clauses.append("m.source ILIKE %s")
            params.append(f"%{source_value}%")
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        # Sort by similarity (DESC) then timestamp (DESC)
        fetch_limit = num_results * 2 if encryption_available else num_results
        sql += f" ORDER BY similarity DESC, m.timestamp DESC LIMIT %s;"
        params.append(fetch_limit)
        
        cur.execute(sql, params)
        results = cur.fetchall()
        
        memories = []
        for row in results:
            if len(memories) >= num_results:
                break
                
            content_bytes = bytes(row[1])
            is_encrypted = row[8] if row[8] is not None else False
            
            # Decode or decrypt content
            content = decode_or_decrypt_content(content_bytes, is_encrypted)
            if content is None:
                logger.debug(f"Skipping memory #{row[0]}: could not decrypt/decode content")
                continue
            
            timestamp_iso = row[6].isoformat()
            # Display appropriate ID based on namespace mode
            display_id = get_display_id(row[0], row[10])
            memory = {
                "id": display_id,
                "source": row[5],
                "content": content,
                "time": format_time_ago(timestamp_iso),
                "similarity": f"{int(row[7] * 100)}%"
            }
            
            if row[4]:
                memory["labels"] = row[4] if isinstance(row[4], list) else json.loads(row[4])
            
            # Get embedding_tables from state
            state = row[9] if row[9] else {}
            embedding_tables = state.get('embedding_tables', {})
            
            memory["meta"] = {
                "timestamp": timestamp_iso,
                "embedding_model": row[2],
                "embedding_dims": embedding_dim,
                "encrypted": is_encrypted,
                "embedding_tables": embedding_tables
            }
            
            memories.append(memory)
    
    else:
        # Path B: Non-semantic query - query memories table directly (source of truth)
        # This works regardless of embedding model changes!
        
        sql = """
            SELECT id, content, namespace, labels, source, timestamp, enc, state, content_id
            FROM memories
        """
        
        params = []
        where_clauses = []
        
        # Filter by namespace on memories table
        if namespace:
            where_clauses.append("namespace = %s")
            params.append(namespace)
        
        # If no encryption key, only return unencrypted memories
        if not encryption_available:
            where_clauses.append("enc = false")
        
        # Label filtering with fuzzy matching (include/exclude)
        # Include labels: fuzzy OR match
        if include_labels:
            include_conditions = []
            for label in include_labels:
                include_conditions.append(f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(labels) AS label WHERE label ILIKE %s)")
                params.append(f"%{label}%")
            where_clauses.append(f"({' OR '.join(include_conditions)})")
        
        # Exclude labels: fuzzy AND NOT match (each exclusion is separate)
        for label in exclude_labels:
            where_clauses.append(f"NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(labels) AS label WHERE label ILIKE %s)")
            params.append(f"%{label}%")
        
        # Source filtering (include or exclude)
        if source_value:
            if source_is_exclusion:
                where_clauses.append("NOT source ILIKE %s")
            else:
                where_clauses.append("source ILIKE %s")
            params.append(f"%{source_value}%")
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        # Sort by timestamp (DESC) - most recent first
        fetch_limit = num_results * 2 if encryption_available else num_results
        sql += f" ORDER BY timestamp DESC LIMIT %s;"
        params.append(fetch_limit)
        
        cur.execute(sql, params)
        results = cur.fetchall()
        
        memories = []
        for row in results:
            if len(memories) >= num_results:
                break
                
            content_bytes = bytes(row[1])
            is_encrypted = row[6] if row[6] is not None else False
            
            # Decode or decrypt content
            content = decode_or_decrypt_content(content_bytes, is_encrypted)
            if content is None:
                logger.debug(f"Skipping memory #{row[0]}: could not decrypt/decode content")
                continue
            
            timestamp_iso = row[5].isoformat()
            # Display appropriate ID based on namespace mode
            display_id = get_display_id(row[0], row[8])
            memory = {
                "id": display_id,
                "source": row[4],
                "content": content,
                "time": format_time_ago(timestamp_iso)
            }
            
            if row[3]:
                memory["labels"] = row[3] if isinstance(row[3], list) else json.loads(row[3])
            
            # Get embedding_tables from state
            state = row[7] if row[7] else {}
            embedding_tables = state.get('embedding_tables', {})
            
            # For non-semantic queries, we don't have embedding info from the query
            memory["meta"] = {
                "timestamp": timestamp_iso,
                "namespace": row[2],
                "encrypted": is_encrypted,
                "embedding_tables": embedding_tables
            }
            
            memories.append(memory)
    
    cur.close()
    conn.close()
    db_time = time.time() - db_start
    
    # Build response - add current_embedding for semantic queries only
    response = {
        "memories": memories,
        "count": len(memories)
    }
    if query:
        response = {"current_embedding": embedding_model, **response}
    
    # Add performance metrics and timezone
    total_time = time.time() - total_start
    response = add_timezone_to_response(response)
    return add_performance_to_response(response, embedding_time, db_time, total_time)

def delete_memory(memory_id: int) -> dict:
    """
    Delete a specific memory by its ID, respecting namespace.
    
    V2 Split-Table Architecture:
    1. Get memory's state.embedding_tables to find all embedding tables
    2. Delete from all tracked embedding tables
    3. Delete from memories table (CASCADE handles current dimension table)
    
    V6 Namespace ID Handling:
    - Wildcard namespace: memory_id is the real database ID
    - Specific namespace: memory_id is content_id, resolved to real ID
    """
    # Performance timing
    total_start = time.time()
    embedding_time = 0.0  # No embedding for delete
    db_time = 0.0
    
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else "default"
    
    # Store the user-facing ID for messages
    user_facing_id = memory_id
    
    # Resolve user-facing ID to real database ID
    real_id, error = resolve_memory_id(memory_id, namespace)
    if error:
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "success": False,
            "error": error
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    # Use the resolved real ID for database operations
    memory_id = real_id
    
    # Database operations (timed)
    db_start = time.time()
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # First, check if memory exists and get its state for embedding table cleanup
        # Note: For wildcard namespace, we still need to verify the memory exists
        # For namespaced mode, resolve_memory_id already verified it exists
        select_sql = """
            SELECT id, state FROM memories
            WHERE id = %s;
        """
        cur.execute(select_sql, (memory_id,))
        
        result = cur.fetchone()
        
        if not result:
            db_time = time.time() - db_start
            total_time = time.time() - total_start
            response = add_timezone_to_response({
                "success": False,
                "error": f"❌ Memory #{user_facing_id} not found or access denied"
            })
            return add_performance_to_response(response, embedding_time, db_time, total_time)
        
        # Get embedding tables from state
        # V3 format: {"embedding_tables": {"memory_384": ["model1"], "memory_768": ["model2"]}}
        # V2 format (backwards compat): {"embedding_tables": ["memory_384", "memory_768"]}
        state = result[1] if result[1] else {}
        embedding_tables_data = state.get('embedding_tables', {})
        
        # Handle both V3 (dict) and V2 (list) formats
        if isinstance(embedding_tables_data, dict):
            # V3 format - keys are table names
            table_names = list(embedding_tables_data.keys())
        elif isinstance(embedding_tables_data, list):
            # V2 format - list of table names (backwards compatibility)
            table_names = embedding_tables_data
        else:
            table_names = []
        
        # Delete from all tracked embedding tables (handles cross-dimensional cleanup)
        for table_name in table_names:
            try:
                cur.execute(f"DELETE FROM {table_name} WHERE memory_id = %s;", (memory_id,))
                logger.debug(f"Deleted embeddings from {table_name} for memory #{memory_id}")
            except Exception as e:
                # Table might not exist anymore - that's OK
                logger.debug(f"Could not delete from {table_name}: {e}")
        
        # Delete from memories table
        if namespace:
            delete_sql = """
                DELETE FROM memories
                WHERE id = %s AND namespace = %s
                RETURNING id;
            """
            cur.execute(delete_sql, (memory_id, namespace))
        else:
            delete_sql = """
                DELETE FROM memories
                WHERE id = %s
                RETURNING id;
            """
            cur.execute(delete_sql, (memory_id,))
        
        deleted_id = cur.fetchone()
        conn.commit()
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        
        if deleted_id:
            response = add_timezone_to_response({
                "success": True,
                "message": f"✅ Memory #{user_facing_id} deleted successfully"
            })
            return add_performance_to_response(response, embedding_time, db_time, total_time)
        else:
            response = add_timezone_to_response({
                "success": False,
                "error": f"❌ Memory #{user_facing_id} not found or access denied"
            })
            return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    except Exception as e:
        conn.rollback()
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "success": False,
            "error": f"❌ Error deleting memory: {str(e)}"
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    finally:
        cur.close()
        conn.close()

def get_memory(memory_id: int) -> dict:
    """
    Get a specific memory by its ID, respecting namespace.
    
    V2 Split-Table Architecture:
    Query memories table directly (source of truth).
    
    V6 Namespace ID Handling:
    - Wildcard namespace: memory_id is the real database ID
    - Specific namespace: memory_id is content_id, resolved to real ID
    """
    # Performance timing
    total_start = time.time()
    embedding_time = 0.0  # No embedding for get
    db_time = 0.0
    
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else "default"
    
    # Store the user-facing ID for messages
    user_facing_id = memory_id
    
    # Resolve user-facing ID to real database ID
    real_id, error = resolve_memory_id(memory_id, namespace)
    if error:
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "error": error
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    # Use the resolved real ID for database operations
    memory_id = real_id
    
    # Database operations (timed)
    db_start = time.time()
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Query memories table directly (source of truth)
        # Note: resolve_memory_id already verified namespace access for namespaced mode
        select_sql = """
            SELECT id, content, namespace, labels, source, timestamp, enc, state, content_id
            FROM memories
            WHERE id = %s;
        """
        cur.execute(select_sql, (memory_id,))
        
        result = cur.fetchone()
        
        if result:
            content_bytes = bytes(result[1])
            is_encrypted = result[6] if result[6] is not None else False
            
            # Decode or decrypt content
            content = decode_or_decrypt_content(content_bytes, is_encrypted)
            if content is None:
                db_time = time.time() - db_start
                total_time = time.time() - total_start
                if is_encrypted:
                    response = add_timezone_to_response({
                        "error": f"❌ Memory #{user_facing_id} is encrypted and cannot be decrypted (missing or wrong key)"
                    })
                else:
                    response = add_timezone_to_response({
                        "error": f"❌ Memory #{user_facing_id} content could not be decoded"
                    })
                return add_performance_to_response(response, embedding_time, db_time, total_time)
            
            timestamp_iso = result[5].isoformat()
            # Display appropriate ID based on namespace mode
            display_id = get_display_id(result[0], result[8])
            memory = {
                "id": display_id,
                "source": result[4],
                "content": content,
                "time": format_time_ago(timestamp_iso)
            }
            
            # Include labels if present
            if result[3]:
                memory["labels"] = result[3] if isinstance(result[3], list) else json.loads(result[3])
            
            # Get embedding info from state
            state = result[7] if result[7] else {}
            embedding_tables = state.get('embedding_tables', {})
            
            # Add meta
            memory["meta"] = {
                "timestamp": timestamp_iso,
                "namespace": result[2],
                "encrypted": is_encrypted,
                "embedding_tables": embedding_tables
            }
            
            db_time = time.time() - db_start
            total_time = time.time() - total_start
            response = add_timezone_to_response(memory)
            return add_performance_to_response(response, embedding_time, db_time, total_time)
        else:
            db_time = time.time() - db_start
            total_time = time.time() - total_start
            response = add_timezone_to_response({
                "error": f"❌ Memory #{user_facing_id} not found or access denied"
            })
            return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    except Exception as e:
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "error": f"❌ Error retrieving memory: {str(e)}"
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    finally:
        cur.close()
        conn.close()

def random_memory(labels: str = None, source: str = None) -> dict:
    """
    Retrieve a random memory, optionally filtered by labels and/or source.
    
    V2 Split-Table Architecture:
    Query memories table directly (source of truth).
    """
    # Performance timing
    total_start = time.time()
    embedding_time = 0.0  # No embedding for random
    db_time = 0.0
    
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else "default"
    
    # Parse labels with exclusion support (! prefix)
    include_labels, exclude_labels = parse_labels_with_exclusions(labels)
    
    # Parse source with exclusion support (! prefix)
    source_value, source_is_exclusion = parse_source_with_exclusion(source)
    
    # Check if encryption key is available
    encryption_available = is_encryption_enabled()
    
    # Database operations (timed)
    db_start = time.time()
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Query memories table directly (source of truth)
        sql = """
            SELECT id, content, namespace, labels, source, timestamp, enc, state, content_id
            FROM memories
        """
        
        params = []
        where_clauses = []
        
        # Filter by namespace
        if namespace:
            where_clauses.append("namespace = %s")
            params.append(namespace)
        
        # If no encryption key, only return unencrypted memories
        if not encryption_available:
            where_clauses.append("enc = false")
        
        # Label filtering with fuzzy matching (include/exclude)
        # Include labels: fuzzy OR match
        if include_labels:
            include_conditions = []
            for label in include_labels:
                include_conditions.append(f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(labels) AS label WHERE label ILIKE %s)")
                params.append(f"%{label}%")
            where_clauses.append(f"({' OR '.join(include_conditions)})")
        
        # Exclude labels: fuzzy AND NOT match (each exclusion is separate)
        for label in exclude_labels:
            where_clauses.append(f"NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(labels) AS label WHERE label ILIKE %s)")
            params.append(f"%{label}%")
        
        # Source filtering (include or exclude)
        if source_value:
            if source_is_exclusion:
                where_clauses.append("NOT source ILIKE %s")
            else:
                where_clauses.append("source ILIKE %s")
            params.append(f"%{source_value}%")
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        # Random ordering - fetch a few in case of decryption failures
        fetch_limit = 5 if encryption_available else 1
        sql += f" ORDER BY RANDOM() LIMIT {fetch_limit};"
        
        cur.execute(sql, params)
        results = cur.fetchall()
        
        # Try to find a memory we can decrypt
        for result in results:
            content_bytes = bytes(result[1])
            is_encrypted = result[6] if result[6] is not None else False
            
            # Decode or decrypt content
            content = decode_or_decrypt_content(content_bytes, is_encrypted)
            if content is None:
                logger.debug(f"Skipping memory #{result[0]}: could not decrypt/decode content")
                continue
            
            timestamp_iso = result[5].isoformat()
            # Display appropriate ID based on namespace mode
            display_id = get_display_id(result[0], result[8])
            memory = {
                "id": display_id,
                "source": result[4],
                "content": content,
                "time": format_time_ago(timestamp_iso)
            }
            
            # Include labels if present
            if result[3]:
                memory["labels"] = result[3] if isinstance(result[3], list) else json.loads(result[3])
            
            # Get embedding_tables from state
            state = result[7] if result[7] else {}
            embedding_tables = state.get('embedding_tables', {})
            
            # Add meta
            memory["meta"] = {
                "timestamp": timestamp_iso,
                "namespace": result[2],
                "encrypted": is_encrypted,
                "embedding_tables": embedding_tables
            }
            
            db_time = time.time() - db_start
            total_time = time.time() - total_start
            response = add_timezone_to_response(memory)
            return add_performance_to_response(response, embedding_time, db_time, total_time)
        
        # No valid memory found
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "error": "❌ No memories found matching the criteria"
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    except Exception as e:
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "error": f"❌ Error retrieving random memory: {str(e)}"
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    finally:
        cur.close()
        conn.close()

def add_labels(memory_id: int, labels: str) -> dict:
    """
    Add labels to an existing memory without replacing existing ones.
    
    V2 Split-Table Architecture:
    Update memories table directly (source of truth).
    
    V6 Namespace ID Handling:
    - Wildcard namespace: memory_id is the real database ID
    - Specific namespace: memory_id is content_id, resolved to real ID
    """
    # Performance timing
    total_start = time.time()
    embedding_time = 0.0  # No embedding for add_labels
    db_time = 0.0
    
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else "default"
    
    # Store the user-facing ID for messages
    user_facing_id = memory_id
    
    # Normalize input labels (supports both comma-separated and JSON array)
    try:
        parsed_labels = json.loads(labels)
        if isinstance(parsed_labels, list):
            new_labels = normalize_labels(parsed_labels)
        else:
            new_labels = normalize_labels(labels)
    except (json.JSONDecodeError, ValueError):
        new_labels = normalize_labels(labels)
    
    if not new_labels:
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "success": False,
            "error": "❌ No valid labels provided"
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    # Resolve user-facing ID to real database ID
    real_id, error = resolve_memory_id(memory_id, namespace)
    if error:
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "success": False,
            "error": error
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    # Use the resolved real ID for database operations
    memory_id = real_id
    
    # Database operations (timed)
    db_start = time.time()
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Fetch existing memory from memories table (source of truth)
        # Note: resolve_memory_id already verified namespace access for namespaced mode
        select_sql = """
            SELECT id, labels
            FROM memories
            WHERE id = %s;
        """
        cur.execute(select_sql, (memory_id,))
        
        result = cur.fetchone()
        
        if not result:
            db_time = time.time() - db_start
            total_time = time.time() - total_start
            response = add_timezone_to_response({
                "success": False,
                "error": f"❌ Memory #{user_facing_id} not found or access denied"
            })
            return add_performance_to_response(response, embedding_time, db_time, total_time)
        
        # Parse existing labels
        existing_labels = []
        if result[1]:
            existing_labels = result[1] if isinstance(result[1], list) else json.loads(result[1])
        
        # Merge labels: extend existing with new, remove duplicates (exact match)
        merged_labels = existing_labels.copy()
        for label in new_labels:
            if label not in merged_labels:
                merged_labels.append(label)
        
        # Update memories table (source of truth)
        update_sql = """
            UPDATE memories
            SET labels = %s
            WHERE id = %s;
        """
        cur.execute(update_sql, (psycopg2.extras.Json(merged_labels), memory_id))
        conn.commit()
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        
        response = add_timezone_to_response({
            "success": True,
            "message": f"✅ Labels added to memory #{user_facing_id}",
            "labels": merged_labels
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    except Exception as e:
        conn.rollback()
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "success": False,
            "error": f"❌ Error adding labels: {str(e)}"
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    finally:
        cur.close()
        conn.close()

def memory_stats(labels: str = None, source: str = None) -> dict:
    """
    Return memory statistics for the configured namespace(s).
    
    Three modes:
    1. No parameters: Return total memory count
    2. labels parameter: Count memories with matching labels (fuzzy)
    3. source parameter: Count memories from matching source (fuzzy)
    4. labels + source: Count memories matching both filters
    
    Args:
        labels: Optional filter for labels (fuzzy match, comma-separated). Use ! prefix to exclude.
        source: Optional source filter (fuzzy match). Use ! prefix to exclude.
        
    Returns:
        Statistics including total count, matching count, percentage,
        and list of matched labels/sources
    """
    # Performance timing
    total_start = time.time()
    embedding_time = 0.0  # No embedding for stats
    db_time = 0.0
    
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else "default"
    
    # Parse labels with exclusion support (! prefix)
    include_labels, exclude_labels = parse_labels_with_exclusions(labels)
    
    # Parse source with exclusion support (! prefix)
    source_value, source_is_exclusion = parse_source_with_exclusion(source)
    
    # Database operations (timed)
    db_start = time.time()
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Build base WHERE clause for namespace
        base_where = []
        base_params = []
        
        if namespace:
            base_where.append("namespace = %s")
            base_params.append(namespace)
        
        base_where_sql = " WHERE " + " AND ".join(base_where) if base_where else ""
        
        # Get total count (always needed)
        total_sql = f"SELECT COUNT(*) FROM memories{base_where_sql};"
        cur.execute(total_sql, base_params)
        total_count = cur.fetchone()[0]
        
        # If no filters, return simple total
        if not include_labels and not exclude_labels and not source_value:
            db_time = time.time() - db_start
            total_time = time.time() - total_start
            response = add_timezone_to_response({
                "total_memories": total_count
            })
            return add_performance_to_response(response, embedding_time, db_time, total_time)
        
        # Build filter conditions
        filter_where = base_where.copy()
        filter_params = base_params.copy()
        
        # Label filtering with fuzzy matching (include/exclude)
        # Include labels: fuzzy OR match
        if include_labels:
            include_conditions = []
            for label in include_labels:
                include_conditions.append(
                    f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(labels) AS label WHERE label ILIKE %s)"
                )
                filter_params.append(f"%{label}%")
            filter_where.append(f"({' OR '.join(include_conditions)})")
        
        # Exclude labels: fuzzy AND NOT match (each exclusion is separate)
        for label in exclude_labels:
            filter_where.append(
                f"NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(labels) AS label WHERE label ILIKE %s)"
            )
            filter_params.append(f"%{label}%")
        
        # Source filtering (include or exclude)
        if source_value:
            if source_is_exclusion:
                filter_where.append("NOT source ILIKE %s")
            else:
                filter_where.append("source ILIKE %s")
            filter_params.append(f"%{source_value}%")
        
        filter_where_sql = " WHERE " + " AND ".join(filter_where) if filter_where else ""
        
        # Get matching count
        match_sql = f"SELECT COUNT(*) FROM memories{filter_where_sql};"
        cur.execute(match_sql, filter_params)
        matching_count = cur.fetchone()[0]
        
        # Calculate percentage
        if total_count > 0:
            percentage = int((matching_count / total_count) * 100)
        else:
            percentage = 0
        
        # Build response
        response = {
            "matching": matching_count,
            "total": total_count,
            "ratio": f"{matching_count}/{total_count}",
            "percentage": f"{percentage}%"
        }
        
        # Get matched labels (distinct labels that matched the fuzzy query)
        # Only show for include labels (exclusions don't have "matched" labels)
        if include_labels:
            # Build query to find all unique labels that match any of the fuzzy patterns
            label_match_conditions = []
            label_match_params = base_params.copy()
            
            for label in include_labels:
                label_match_conditions.append("lbl ILIKE %s")
                label_match_params.append(f"%{label}%")
            
            labels_sql = f"""
                SELECT DISTINCT lbl
                FROM memories, jsonb_array_elements_text(labels) AS lbl
                {base_where_sql}
                {"AND" if base_where else "WHERE"} ({' OR '.join(label_match_conditions)})
                ORDER BY lbl;
            """
            cur.execute(labels_sql, label_match_params)
            matched_labels = [row[0] for row in cur.fetchall()]
            response["labels_matched"] = matched_labels
        
        # Get matched sources (distinct sources that matched the fuzzy query)
        # Only show for include source (exclusions don't have "matched" sources)
        if source_value and not source_is_exclusion:
            source_match_params = base_params.copy()
            source_match_params.append(f"%{source_value}%")
            
            sources_sql = f"""
                SELECT DISTINCT source
                FROM memories
                {base_where_sql}
                {"AND" if base_where else "WHERE"} source ILIKE %s
                ORDER BY source;
            """
            cur.execute(sources_sql, source_match_params)
            matched_sources = [row[0] for row in cur.fetchall() if row[0] is not None]
            response["sources_matched"] = matched_sources
        
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        
        response = add_timezone_to_response(response)
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    except Exception as e:
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "error": f"❌ Error getting memory stats: {str(e)}"
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    finally:
        cur.close()
        conn.close()


def del_labels(memory_id: int, labels: str) -> dict:
    """
    Remove specific labels from an existing memory (exact match, case-sensitive).
    
    V2 Split-Table Architecture:
    Update memories table directly (source of truth).
    
    V6 Namespace ID Handling:
    - Wildcard namespace: memory_id is the real database ID
    - Specific namespace: memory_id is content_id, resolved to real ID
    """
    # Performance timing
    total_start = time.time()
    embedding_time = 0.0  # No embedding for del_labels
    db_time = 0.0
    
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else "default"
    
    # Store the user-facing ID for messages
    user_facing_id = memory_id
    
    # Normalize input labels (supports both comma-separated and JSON array)
    try:
        parsed_labels = json.loads(labels)
        if isinstance(parsed_labels, list):
            labels_to_remove = normalize_labels(parsed_labels)
        else:
            labels_to_remove = normalize_labels(labels)
    except (json.JSONDecodeError, ValueError):
        labels_to_remove = normalize_labels(labels)
    
    if not labels_to_remove:
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "success": False,
            "error": "❌ No valid labels provided"
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    # Resolve user-facing ID to real database ID
    real_id, error = resolve_memory_id(memory_id, namespace)
    if error:
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "success": False,
            "error": error
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    # Use the resolved real ID for database operations
    memory_id = real_id
    
    # Database operations (timed)
    db_start = time.time()
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Fetch existing memory from memories table (source of truth)
        # Note: resolve_memory_id already verified namespace access for namespaced mode
        select_sql = """
            SELECT id, labels
            FROM memories
            WHERE id = %s;
        """
        cur.execute(select_sql, (memory_id,))
        
        result = cur.fetchone()
        
        if not result:
            db_time = time.time() - db_start
            total_time = time.time() - total_start
            response = add_timezone_to_response({
                "success": False,
                "error": f"❌ Memory #{user_facing_id} not found or access denied"
            })
            return add_performance_to_response(response, embedding_time, db_time, total_time)
        
        # Parse existing labels
        existing_labels = []
        if result[1]:
            existing_labels = result[1] if isinstance(result[1], list) else json.loads(result[1])
        
        # Remove specified labels (exact string match, case-sensitive)
        # Silently ignore non-existent labels
        remaining_labels = [label for label in existing_labels if label not in labels_to_remove]
        
        # Update memories table (source of truth)
        update_sql = """
            UPDATE memories
            SET labels = %s
            WHERE id = %s;
        """
        cur.execute(update_sql, (psycopg2.extras.Json(remaining_labels), memory_id))
        conn.commit()
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        
        response = add_timezone_to_response({
            "success": True,
            "message": f"✅ Labels removed from memory #{user_facing_id}",
            "labels": remaining_labels
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    
    except Exception as e:
        conn.rollback()
        db_time = time.time() - db_start
        total_time = time.time() - total_start
        response = add_timezone_to_response({
            "success": False,
            "error": f"❌ Error removing labels: {str(e)}"
        })
        return add_performance_to_response(response, embedding_time, db_time, total_time)
    finally:
        cur.close()
        conn.close()
