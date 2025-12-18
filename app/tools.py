
from app.database import get_db_connection
from app.embedding import get_embedding, get_embedding_dimension
from app.config import EMBEDDING_MODEL, NAMESPACE
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
        months = int(days / 30.44)
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
    """Stores a memory in the database with duplicate detection."""
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
    namespace = NAMESPACE if NAMESPACE is not None else ""

    embedding = get_embedding(content)
    embedding_dim = len(embedding)
    table_name = f"memory_{embedding_dim}"

    conn = get_db_connection()
    cur = conn.cursor()

    # Check for similar memories (duplicate detection)
    warnings = []
    check_sql = f"""
        SELECT id, content, enc, 1 - (embedding <=> %s::vector) as similarity
        FROM {table_name}
        WHERE namespace = %s AND embedding_model = %s
        ORDER BY similarity DESC
        LIMIT 2;
    """
    cur.execute(check_sql, (embedding, namespace, embedding_model))
    similar_memories = cur.fetchall()
    
    for row in similar_memories:
        mem_id, mem_content_bytes, mem_enc, similarity = row
        # Safely decode or decrypt content for comparison
        mem_enc = mem_enc if mem_enc is not None else False
        mem_content = decode_or_decrypt_content(bytes(mem_content_bytes), mem_enc)
        if mem_content is None:
            # Skip encrypted memories we can't decrypt for duplicate detection
            continue
        if similarity >= 0.70:  # 70% threshold
            percentage = int(similarity * 100)
            warnings.append(f"⚠️ Very similar to memory #{mem_id} ({percentage}%) - might be redundant")

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
    
    # Store the memory with labels as JSON array
    cur.execute(
        f"""INSERT INTO {table_name} (content, embedding, namespace, labels, source, embedding_model, enc)
        VALUES (%s, %s::vector, %s, %s, %s, %s, %s) RETURNING id;""",
        (content_bytes, embedding, namespace, json.dumps(label_list), source, embedding_model, is_encrypted)
    )

    memory_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    result = {
        "id": memory_id,
        "source": source,
        "message": f"✅ Memory stored with ID {memory_id}" + (" 🔐" if is_encrypted else "")
    }
    
    if warnings:
        result["warnings"] = warnings
    
    return result

def retrieve_memories(query: str = None, labels: str = None, source: str = None, num_results: int = 5) -> dict:
    """
    Retrieve memories with flexible filtering combinations.
    
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
    
    # Parse comma-separated labels into array (safely handle None)
    label_list = []
    if labels is not None and isinstance(labels, str):
        label_list = [t.strip() for t in labels.split(',') if t.strip()]
    
    # No validation required - all parameters are optional
    # When no parameters provided, returns most recent memories
    
    # Auto-populate from config
    embedding_model = EMBEDDING_MODEL
    namespace = NAMESPACE if NAMESPACE is not None else ""
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Check if encryption key is available (affects which memories we can access)
    encryption_available = is_encryption_enabled()
    
    # Determine which search path to use
    if query:
        # Path A: Semantic search (with optional label filtering)
        embedding = get_embedding(query)
        embedding_dim = len(embedding)
        table_name = f"memory_{embedding_dim}"
        
        # Build SQL query with filters - include enc flag
        sql = f"""SELECT id, content, embedding_model, namespace, labels, source, timestamp, 
                         1 - (embedding <=> %s::vector) as similarity, enc
                  FROM {table_name}"""
        
        params = [embedding]
        where_clauses = []
        
        # ALWAYS filter by embedding model (prevents dimension mismatches)
        where_clauses.append("embedding_model = %s")
        params.append(embedding_model)
        
        # Only filter by namespace if NAMESPACE is not empty
        if namespace:
            where_clauses.append("namespace = %s")
            params.append(namespace)
        
        # If no encryption key, only return unencrypted memories
        if not encryption_available:
            where_clauses.append("enc = false")
        
        # Label filtering with fuzzy OR matching on JSONB array elements
        if label_list:
            label_conditions = []
            for label in label_list:
                # Check if ANY element in the JSONB array contains this label (fuzzy)
                label_conditions.append(f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(labels) AS label WHERE label ILIKE %s)")
                params.append(f"%{label}%")
            
            if label_conditions:
                where_clauses.append(f"({' OR '.join(label_conditions)})")
        
        # Source filtering with fuzzy matching
        if source:
            where_clauses.append("source ILIKE %s")
            params.append(f"%{source}%")
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        # Sort by similarity (DESC) then timestamp (DESC)
        # Fetch extra to account for potential decryption failures
        fetch_limit = num_results * 2 if encryption_available else num_results
        sql += f" ORDER BY similarity DESC, timestamp DESC LIMIT %s;"
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
                # Skip this memory (decryption failed or key not available)
                logger.debug(f"Skipping memory #{row[0]}: could not decrypt/decode content")
                continue
            
            timestamp_iso = row[6].isoformat()
            memory = {
                "id": row[0],
                "source": row[5],
                "content": content,
                "time": format_time_ago(timestamp_iso),
                "similarity": f"{int(row[7] * 100)}%"
            }
            
            # Include labels if present (before meta)
            if row[4]:
                memory["labels"] = row[4] if isinstance(row[4], list) else json.loads(row[4])
            
            # Add meta last
            memory["meta"] = {
                "timestamp": timestamp_iso,
                "embedding_model": row[2],
                "embedding_dims": embedding_dim,
                "encrypted": is_encrypted
            }
            
            memories.append(memory)
    
    else:
        # Path B: Pure label-based query (no semantic search)
        embedding_dim = get_embedding_dimension()
        table_name = f"memory_{embedding_dim}"
        
        # Build SQL query without embedding similarity - include enc flag
        sql = f"""SELECT id, content, embedding_model, namespace, labels, source, timestamp, enc
                  FROM {table_name}"""
        
        params = []
        where_clauses = []
        
        # ALWAYS filter by embedding model (prevents dimension mismatches)
        where_clauses.append("embedding_model = %s")
        params.append(embedding_model)
        
        # Only filter by namespace if NAMESPACE is not empty
        if namespace:
            where_clauses.append("namespace = %s")
            params.append(namespace)
        
        # If no encryption key, only return unencrypted memories
        if not encryption_available:
            where_clauses.append("enc = false")
        
        # Label filtering with fuzzy OR matching on JSONB array elements
        label_conditions = []
        for label in label_list:
            # Check if ANY element in the JSONB array contains this label (fuzzy)
            label_conditions.append(f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(labels) AS label WHERE label ILIKE %s)")
            params.append(f"%{label}%")
        
        if label_conditions:
            where_clauses.append(f"({' OR '.join(label_conditions)})")
        
        # Source filtering with fuzzy matching
        if source:
            where_clauses.append("source ILIKE %s")
            params.append(f"%{source}%")
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        # Sort by timestamp (DESC) - most recent first
        # Fetch extra to account for potential decryption failures
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
            is_encrypted = row[7] if row[7] is not None else False
            
            # Decode or decrypt content
            content = decode_or_decrypt_content(content_bytes, is_encrypted)
            if content is None:
                # Skip this memory (decryption failed or key not available)
                logger.debug(f"Skipping memory #{row[0]}: could not decrypt/decode content")
                continue
            
            timestamp_iso = row[6].isoformat()
            memory = {
                "id": row[0],
                "source": row[5],
                "content": content,
                "time": format_time_ago(timestamp_iso)
            }
            
            # Include labels if present (before meta)
            if row[4]:
                memory["labels"] = row[4] if isinstance(row[4], list) else json.loads(row[4])
            
            # Add meta last
            memory["meta"] = {
                "timestamp": timestamp_iso,
                "embedding_model": row[2],
                "embedding_dims": embedding_dim,
                "encrypted": is_encrypted
            }
            
            memories.append(memory)
    
    cur.close()
    conn.close()
    
    return {
        "memories": memories,
        "count": len(memories)
    }

def delete_memory(memory_id: int) -> dict:
    """Delete a specific memory by its ID, respecting namespace."""
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else ""
    
    # Get embedding dimension to determine table
    embedding_dim = get_embedding_dimension()
    table_name = f"memory_{embedding_dim}"
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Build WHERE clause based on namespace
        if namespace:
            # If namespace is set, only delete if it matches
            delete_sql = f"""
                DELETE FROM {table_name}
                WHERE id = %s AND namespace = %s
                RETURNING id;
            """
            cur.execute(delete_sql, (memory_id, namespace))
        else:
            # If namespace is empty, delete from any namespace (user owns all)
            delete_sql = f"""
                DELETE FROM {table_name}
                WHERE id = %s
                RETURNING id;
            """
            cur.execute(delete_sql, (memory_id,))
        
        deleted_id = cur.fetchone()
        conn.commit()
        
        if deleted_id:
            return {
                "success": True,
                "message": f"✅ Memory #{memory_id} deleted successfully"
            }
        else:
            return {
                "success": False,
                "error": f"❌ Memory #{memory_id} not found or access denied"
            }
    
    except Exception as e:
        conn.rollback()
        return {
            "success": False,
            "error": f"❌ Error deleting memory: {str(e)}"
        }
    finally:
        cur.close()
        conn.close()

def get_memory(memory_id: int) -> dict:
    """Get a specific memory by its ID, respecting namespace."""
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else ""
    
    # Get embedding dimension to determine table
    embedding_dim = get_embedding_dimension()
    table_name = f"memory_{embedding_dim}"
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Build WHERE clause based on namespace - include enc flag
        if namespace:
            # If namespace is set, only retrieve if it matches
            select_sql = f"""
                SELECT id, content, embedding_model, namespace, labels, source, timestamp, enc
                FROM {table_name}
                WHERE id = %s AND namespace = %s;
            """
            cur.execute(select_sql, (memory_id, namespace))
        else:
            # If namespace is empty, retrieve from any namespace (user owns all)
            select_sql = f"""
                SELECT id, content, embedding_model, namespace, labels, source, timestamp, enc
                FROM {table_name}
                WHERE id = %s;
            """
            cur.execute(select_sql, (memory_id,))
        
        result = cur.fetchone()
        
        if result:
            content_bytes = bytes(result[1])
            is_encrypted = result[7] if result[7] is not None else False
            
            # Decode or decrypt content
            content = decode_or_decrypt_content(content_bytes, is_encrypted)
            if content is None:
                # Cannot decrypt - either no key or wrong key
                if is_encrypted:
                    return {
                        "error": f"❌ Memory #{memory_id} is encrypted and cannot be decrypted (missing or wrong key)"
                    }
                else:
                    return {
                        "error": f"❌ Memory #{memory_id} content could not be decoded"
                    }
            
            timestamp_iso = result[6].isoformat()
            memory = {
                "id": result[0],
                "source": result[5],
                "content": content,
                "time": format_time_ago(timestamp_iso)
            }
            
            # Include labels if present (before meta)
            if result[4]:
                memory["labels"] = result[4] if isinstance(result[4], list) else json.loads(result[4])
            
            # Add meta last
            memory["meta"] = {
                "timestamp": timestamp_iso,
                "embedding_model": result[2],
                "embedding_dims": embedding_dim,
                "encrypted": is_encrypted
            }
            
            return memory
        else:
            return {
                "error": f"❌ Memory #{memory_id} not found or access denied"
            }
    
    except Exception as e:
        return {
            "error": f"❌ Error retrieving memory: {str(e)}"
        }
    finally:
        cur.close()
        conn.close()

def random_memory(labels: str = None, source: str = None) -> dict:
    """Retrieve a random memory, optionally filtered by labels and/or source."""
    # Auto-populate from config
    embedding_model = EMBEDDING_MODEL
    namespace = NAMESPACE if NAMESPACE is not None else ""
    
    # Get embedding dimension to determine table
    embedding_dim = get_embedding_dimension()
    table_name = f"memory_{embedding_dim}"
    
    # Parse comma-separated labels into array
    label_list = []
    if labels:
        label_list = [t.strip() for t in labels.split(',') if t.strip()]
    
    # Check if encryption key is available
    encryption_available = is_encryption_enabled()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Build SQL query with filters - include enc flag
        sql = f"""SELECT id, content, embedding_model, namespace, labels, source, timestamp, enc
                  FROM {table_name}"""
        
        params = []
        where_clauses = []
        
        # ALWAYS filter by embedding model (prevents dimension mismatches)
        where_clauses.append("embedding_model = %s")
        params.append(embedding_model)
        
        # Only filter by namespace if NAMESPACE is not empty
        if namespace:
            where_clauses.append("namespace = %s")
            params.append(namespace)
        
        # If no encryption key, only return unencrypted memories
        if not encryption_available:
            where_clauses.append("enc = false")
        
        # Label filtering with fuzzy OR matching on JSONB array elements
        if label_list:
            label_conditions = []
            for label in label_list:
                # Check if ANY element in the JSONB array contains this label (fuzzy)
                label_conditions.append(f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(labels) AS label WHERE label ILIKE %s)")
                params.append(f"%{label}%")
            
            if label_conditions:
                where_clauses.append(f"({' OR '.join(label_conditions)})")
        
        # Source filtering with fuzzy matching
        if source:
            where_clauses.append("source ILIKE %s")
            params.append(f"%{source}%")
        
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
            is_encrypted = result[7] if result[7] is not None else False
            
            # Decode or decrypt content
            content = decode_or_decrypt_content(content_bytes, is_encrypted)
            if content is None:
                # Skip this memory (decryption failed)
                logger.debug(f"Skipping memory #{result[0]}: could not decrypt/decode content")
                continue
            
            timestamp_iso = result[6].isoformat()
            memory = {
                "id": result[0],
                "source": result[5],
                "content": content,
                "time": format_time_ago(timestamp_iso)
            }
            
            # Include labels if present (before meta)
            if result[4]:
                memory["labels"] = result[4] if isinstance(result[4], list) else json.loads(result[4])
            
            # Add meta last
            memory["meta"] = {
                "timestamp": timestamp_iso,
                "embedding_model": result[2],
                "embedding_dims": embedding_dim,
                "encrypted": is_encrypted
            }
            
            return memory
        
        # No valid memory found
        return {
            "error": "❌ No memories found matching the criteria"
        }
    
    except Exception as e:
        return {
            "error": f"❌ Error retrieving random memory: {str(e)}"
        }
    finally:
        cur.close()
        conn.close()

def add_labels(memory_id: int, labels: str) -> dict:
    """Add labels to an existing memory without replacing existing ones."""
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else ""
    
    # Get embedding dimension to determine table
    embedding_dim = get_embedding_dimension()
    table_name = f"memory_{embedding_dim}"
    
    # Normalize input labels (supports both comma-separated and JSON array)
    try:
        # Try parsing as JSON array first
        parsed_labels = json.loads(labels)
        if isinstance(parsed_labels, list):
            new_labels = normalize_labels(parsed_labels)
        else:
            # Not a list, treat as comma-separated string
            new_labels = normalize_labels(labels)
    except (json.JSONDecodeError, ValueError):
        # Not valid JSON, treat as comma-separated string
        new_labels = normalize_labels(labels)
    
    if not new_labels:
        return {
            "success": False,
            "error": "❌ No valid labels provided"
        }
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Fetch existing memory with namespace check
        if namespace:
            select_sql = f"""
                SELECT id, labels
                FROM {table_name}
                WHERE id = %s AND namespace = %s;
            """
            cur.execute(select_sql, (memory_id, namespace))
        else:
            select_sql = f"""
                SELECT id, labels
                FROM {table_name}
                WHERE id = %s;
            """
            cur.execute(select_sql, (memory_id,))
        
        result = cur.fetchone()
        
        if not result:
            return {
                "success": False,
                "error": f"❌ Memory #{memory_id} not found or access denied"
            }
        
        # Parse existing labels
        existing_labels = []
        if result[1]:
            existing_labels = result[1] if isinstance(result[1], list) else json.loads(result[1])
        
        # Merge labels: extend existing with new, remove duplicates (exact match)
        merged_labels = existing_labels.copy()
        for label in new_labels:
            if label not in merged_labels:
                merged_labels.append(label)
        
        # Update memory with merged labels
        update_sql = f"""
            UPDATE {table_name}
            SET labels = %s
            WHERE id = %s;
        """
        cur.execute(update_sql, (json.dumps(merged_labels), memory_id))
        conn.commit()
        
        return {
            "success": True,
            "message": f"✅ Labels added to memory #{memory_id}",
            "labels": merged_labels
        }
    
    except Exception as e:
        conn.rollback()
        return {
            "success": False,
            "error": f"❌ Error adding labels: {str(e)}"
        }
    finally:
        cur.close()
        conn.close()

def del_labels(memory_id: int, labels: str) -> dict:
    """Remove specific labels from an existing memory (exact match, case-sensitive)."""
    # Auto-populate from config
    namespace = NAMESPACE if NAMESPACE is not None else ""
    
    # Get embedding dimension to determine table
    embedding_dim = get_embedding_dimension()
    table_name = f"memory_{embedding_dim}"
    
    # Normalize input labels (supports both comma-separated and JSON array)
    try:
        # Try parsing as JSON array first
        parsed_labels = json.loads(labels)
        if isinstance(parsed_labels, list):
            new_labels = normalize_labels(parsed_labels)
        else:
            # Not a list, treat as comma-separated string
            new_labels = normalize_labels(labels)
    except (json.JSONDecodeError, ValueError):
        # Not valid JSON, treat as comma-separated string
        new_labels = normalize_labels(labels)
    
    if not new_labels:
        return {
            "success": False,
            "error": "❌ No valid labels provided"
        }
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Fetch existing memory with namespace check
        if namespace:
            select_sql = f"""
                SELECT id, labels
                FROM {table_name}
                WHERE id = %s AND namespace = %s;
            """
            cur.execute(select_sql, (memory_id, namespace))
        else:
            select_sql = f"""
                SELECT id, labels
                FROM {table_name}
                WHERE id = %s;
            """
            cur.execute(select_sql, (memory_id,))
        
        result = cur.fetchone()
        
        if not result:
            return {
                "success": False,
                "error": f"❌ Memory #{memory_id} not found or access denied"
            }
        
        # Parse existing labels
        existing_labels = []
        if result[1]:
            existing_labels = result[1] if isinstance(result[1], list) else json.loads(result[1])
        
        # Remove specified labels (exact string match, case-sensitive)
        # Silently ignore non-existent labels
        remaining_labels = [label for label in existing_labels if label not in labels_to_remove]
        
        # Update memory with remaining labels
        update_sql = f"""
            UPDATE {table_name}
            SET labels = %s
            WHERE id = %s;
        """
        cur.execute(update_sql, (json.dumps(remaining_labels), memory_id))
        conn.commit()
        
        return {
            "success": True,
            "message": f"✅ Labels removed from memory #{memory_id}",
            "labels": remaining_labels
        }
    
    except Exception as e:
        conn.rollback()
        return {
            "success": False,
            "error": f"❌ Error removing labels: {str(e)}"
        }
    finally:
        cur.close()
        conn.close()
