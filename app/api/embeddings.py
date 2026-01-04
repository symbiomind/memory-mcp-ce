"""
Memory MCP-CE API - Embeddings Endpoint

POST /api/embeddings/generate
Re-embeds memories with a new embedding model in the background.
"""

import logging
import threading
import json
from typing import Optional

from openai import OpenAI
import psycopg2.extras

from app.config import NAMESPACE
from app.database import (
    get_db_connection,
    table_exists,
    create_embedding_table,
    add_embedding_to_state,
)
from app.encryption import decode_or_decrypt_content

logger = logging.getLogger(__name__)


def _detect_embedding_dimensions(client: OpenAI, model: str, requested_dims: Optional[int] = None) -> int:
    """
    Detect embedding dimensions by calling the API with test text.
    
    Args:
        client: OpenAI client configured for the embedding endpoint
        model: Model name to test
        requested_dims: Optional specific dimension count to request (for MRL models)
    
    Returns:
        Number of dimensions in the embedding vector
    
    Raises:
        Exception if the model doesn't return valid embeddings
        ValueError if requested_dims doesn't match actual returned dimensions
    """
    try:
        # Build API call - only include dimensions if requested_dims is set
        api_kwargs = {
            "model": model,
            "input": "test",
        }
        if requested_dims is not None:
            api_kwargs["dimensions"] = requested_dims
        
        response = client.embeddings.create(**api_kwargs)
        
        if not response.data or len(response.data) == 0:
            raise ValueError(f"Model '{model}' returned no embeddings")
        
        embedding = response.data[0].embedding
        
        if not isinstance(embedding, (list, tuple)):
            raise TypeError(
                f"Expected embedding to be a list/array, got {type(embedding).__name__}"
            )
        
        actual_dims = len(embedding)
        if actual_dims == 0:
            raise ValueError("Model returned empty embedding vector")
        
        # CRITICAL: Validate dimensions match if requested_dims was specified
        if requested_dims is not None and actual_dims != requested_dims:
            raise ValueError(
                f"Re-embed failed: Model '{model}' returned {actual_dims} dimensions "
                f"but embedding_dims={requested_dims} was requested. "
                f"Either remove embedding_dims to use native dimensions, or use an MRL-capable model."
            )
        
        if requested_dims is not None:
            logger.info(f"✓ Validated embedding dimensions for {model}: {actual_dims}D (embedding_dims={requested_dims})")
        else:
            logger.info(f"✓ Detected embedding dimensions for {model}: {actual_dims}D")
        return actual_dims
        
    except Exception as e:
        raise Exception(f"Failed to detect dimensions for model '{model}': {str(e)}")


def _get_embedding(client: OpenAI, model: str, text: str, dims: Optional[int] = None) -> list[float]:
    """
    Generate embedding for text using the provided client and model.
    
    Args:
        client: OpenAI client configured for the embedding endpoint
        model: Model name
        text: Text to embed
        dims: Optional dimension count to request (for MRL models)
    """
    # Build API call - only include dimensions if dims is set
    api_kwargs = {
        "model": model,
        "input": text,
    }
    if dims is not None:
        api_kwargs["dimensions"] = dims
    
    response = client.embeddings.create(**api_kwargs)
    return response.data[0].embedding


def _do_reembedding(
    embedding_url: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    namespace: Optional[str],
    dims: int,
    table_name: str,
    requested_dims: Optional[int] = None,
) -> None:
    """
    Background worker function that re-embeds memories.
    
    This runs in a separate thread and processes memories that don't
    have embeddings for the specified model yet.
    
    Args:
        embedding_url: OpenAI-compatible endpoint URL
        embedding_model: Model name to use
        embedding_api_key: Optional API key
        namespace: If set, filter to this namespace. If None/empty, process ALL namespaces.
        dims: Target dimension count for table selection
        table_name: Embedding table name (e.g., memory_768)
        requested_dims: Optional dimension count to request from API (for MRL models)
    """
    namespace_display = namespace if namespace else "(all namespaces)"
    dims_display = f", embedding_dims={requested_dims}" if requested_dims is not None else ""
    logger.info(f"🚀 Starting re-embedding job: model={embedding_model}{dims_display}, namespace={namespace_display}, table={table_name}")
    
    # Create dedicated OpenAI client for this job
    client = OpenAI(
        base_url=embedding_url,
        api_key=embedding_api_key or "not-needed",
    )
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Build query to find memories that DON'T have embeddings for this model yet
        # The model is stored in state.embedding_tables.{table_name} array
        where_clauses = []
        params = []
        
        # Only filter by namespace if it's set (empty = all namespaces)
        if namespace:
            where_clauses.append("namespace = %s")
            params.append(namespace)
        
        # Filter for memories missing this model in their state
        where_clauses.append("""
            NOT (
                COALESCE(state->'embedding_tables'->%s, '[]'::jsonb) @> %s::jsonb
            )
        """)
        params.extend([table_name, json.dumps([embedding_model])])
        
        sql = f"""
            SELECT id, content, enc, state, namespace
            FROM memories 
            WHERE {' AND '.join(where_clauses)};
        """
        cur.execute(sql, params)
        
        memories = cur.fetchall()
        total = len(memories)
        
        if total == 0:
            logger.info(f"✅ No memories need re-embedding for model {embedding_model}")
            return
        
        logger.info(f"📊 Found {total} memories to re-embed")
        
        processed = 0
        skipped = 0
        errors = 0
        
        for row in memories:
            memory_id, content_bytes, is_encrypted, state, memory_namespace = row
            is_encrypted = is_encrypted if is_encrypted is not None else False
            
            try:
                # Decode or decrypt content
                content = decode_or_decrypt_content(bytes(content_bytes), is_encrypted)
                
                if content is None:
                    if is_encrypted:
                        logger.warning(f"⚠️ Skipping memory #{memory_id}: encrypted but ENCRYPTION_KEY not set")
                    else:
                        logger.warning(f"⚠️ Skipping memory #{memory_id}: could not decode content")
                    skipped += 1
                    continue
                
                # Generate new embedding (pass requested_dims for MRL models)
                embedding = _get_embedding(client, embedding_model, content, requested_dims)
                
                # Insert into embedding table (use memory's actual namespace)
                cur.execute(f"""
                    INSERT INTO {table_name} (memory_id, embedding, namespace, embedding_model)
                    VALUES (%s, %s::vector, %s, %s)
                    ON CONFLICT (memory_id, embedding_model) DO NOTHING;
                """, (memory_id, embedding, memory_namespace, embedding_model))
                
                # Update state.embedding_tables to include new model
                add_embedding_to_state(memory_id, table_name, embedding_model)
                
                processed += 1
                
                # Log progress periodically
                if processed % 10 == 0:
                    logger.info(f"📈 Progress: {processed}/{total} memories re-embedded")
                
            except Exception as e:
                logger.error(f"❌ Error re-embedding memory #{memory_id}: {str(e)}")
                errors += 1
                continue
        
        conn.commit()
        logger.info(f"✅ Re-embedding complete: {processed} processed, {skipped} skipped, {errors} errors")
        
    except Exception as e:
        logger.error(f"❌ Re-embedding job failed: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


async def generate_embeddings_handler(request_body: dict) -> dict:
    """
    Handle POST /api/embeddings/generate request.
    
    Request body:
    {
        "embedding_url": "http://...",      # Required: OpenAI-compatible endpoint
        "embedding_model": "model-name",    # Required: Model name
        "embedding_api_key": "sk-...",      # Optional: API key for the service
        "embedding_dims": 768,              # Optional: Request specific dimensions (for MRL models)
        "namespace": "personal"             # Optional: defaults to .env NAMESPACE
    }
    
    If embedding_dims is specified:
      - passes dimensions parameter to embedding API
      - CRITICAL: validates returned dimensions match, fails if mismatch
      - prevents wrong-sized vectors in memory_{dims} tables
    
    Returns:
        202 response dict with status and message
    
    Raises:
        ValueError if required fields are missing or dimension validation fails
    """
    # Validate required fields
    embedding_url = request_body.get("embedding_url")
    embedding_model = request_body.get("embedding_model")
    
    if not embedding_url:
        raise ValueError("Missing required field: embedding_url")
    if not embedding_model:
        raise ValueError("Missing required field: embedding_model")
    
    # Optional fields
    embedding_api_key = request_body.get("embedding_api_key")
    
    # Parse optional embedding_dims (for MRL models like Qwen)
    # None = auto-detect, int = request specific dimensions
    embedding_dims_raw = request_body.get("embedding_dims")
    embedding_dims: Optional[int] = None
    if embedding_dims_raw is not None:
        try:
            embedding_dims = int(embedding_dims_raw)
            if embedding_dims <= 0:
                raise ValueError("embedding_dims must be a positive integer")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid embedding_dims value: {embedding_dims_raw}. Must be a positive integer.")
    
    # Namespace handling:
    # - If provided in request → use that value
    # - If not provided → use .env NAMESPACE (could be empty string meaning "all namespaces")
    # - Empty string / None = process ALL namespaces
    namespace = request_body.get("namespace")
    if namespace is None:
        namespace = NAMESPACE  # From .env - could be "" (all) or a specific value
    
    # Create client to detect dimensions
    client = OpenAI(
        base_url=embedding_url,
        api_key=embedding_api_key or "not-needed",
    )
    
    # Detect/validate embedding dimensions
    # This will FAIL if embedding_dims is set but model doesn't return matching dimensions
    dims = _detect_embedding_dimensions(client, embedding_model, embedding_dims)
    table_name = f"memory_{dims}"
    
    # Ensure embedding table exists
    if not table_exists(table_name):
        logger.info(f"📦 Creating new embedding table: {table_name}")
        create_embedding_table(dims)
    
    # Start background thread (pass embedding_dims for consistent generation)
    thread = threading.Thread(
        target=_do_reembedding,
        args=(embedding_url, embedding_model, embedding_api_key, namespace, dims, table_name, embedding_dims),
        daemon=True,
    )
    thread.start()
    
    # Display-friendly namespace for response
    namespace_display = namespace if namespace else "(all namespaces)"
    
    # Build response
    response = {
        "status": "processing",
        "message": f"Re-embedding started in background for model '{embedding_model}' ({dims}D)",
        "namespace": namespace_display,
        "embedding_table": table_name,
    }
    
    # Include embedding_dims in response if it was specified
    if embedding_dims is not None:
        response["embedding_dims"] = embedding_dims
    
    return response
