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


def _detect_embedding_dimensions(client: OpenAI, model: str) -> int:
    """
    Detect embedding dimensions by calling the API with test text.
    
    Args:
        client: OpenAI client configured for the embedding endpoint
        model: Model name to test
    
    Returns:
        Number of dimensions in the embedding vector
    
    Raises:
        Exception if the model doesn't return valid embeddings
    """
    try:
        response = client.embeddings.create(
            model=model,
            input="test",
        )
        
        if not response.data or len(response.data) == 0:
            raise ValueError(f"Model '{model}' returned no embeddings")
        
        embedding = response.data[0].embedding
        
        if not isinstance(embedding, (list, tuple)):
            raise TypeError(
                f"Expected embedding to be a list/array, got {type(embedding).__name__}"
            )
        
        dims = len(embedding)
        if dims == 0:
            raise ValueError("Model returned empty embedding vector")
        
        logger.info(f"✓ Detected embedding dimensions for {model}: {dims}D")
        return dims
        
    except Exception as e:
        raise Exception(f"Failed to detect dimensions for model '{model}': {str(e)}")


def _get_embedding(client: OpenAI, model: str, text: str) -> list[float]:
    """Generate embedding for text using the provided client and model."""
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


def _do_reembedding(
    embedding_url: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    namespace: Optional[str],
    dims: int,
    table_name: str,
) -> None:
    """
    Background worker function that re-embeds memories.
    
    This runs in a separate thread and processes memories that don't
    have embeddings for the specified model yet.
    
    Args:
        namespace: If set, filter to this namespace. If None/empty, process ALL namespaces.
    """
    namespace_display = namespace if namespace else "(all namespaces)"
    logger.info(f"🚀 Starting re-embedding job: model={embedding_model}, namespace={namespace_display}, table={table_name}")
    
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
                
                # Generate new embedding
                embedding = _get_embedding(client, embedding_model, content)
                
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
        "namespace": "personal"             # Optional: defaults to .env NAMESPACE
    }
    
    Returns:
        202 response dict with status and message
    
    Raises:
        ValueError if required fields are missing
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
    
    # Detect embedding dimensions
    dims = _detect_embedding_dimensions(client, embedding_model)
    table_name = f"memory_{dims}"
    
    # Ensure embedding table exists
    if not table_exists(table_name):
        logger.info(f"📦 Creating new embedding table: {table_name}")
        create_embedding_table(dims)
    
    # Start background thread
    thread = threading.Thread(
        target=_do_reembedding,
        args=(embedding_url, embedding_model, embedding_api_key, namespace, dims, table_name),
        daemon=True,
    )
    thread.start()
    
    # Display-friendly namespace for response
    namespace_display = namespace if namespace else "(all namespaces)"
    
    return {
        "status": "processing",
        "message": f"Re-embedding started in background for model '{embedding_model}' ({dims}D)",
        "namespace": namespace_display,
        "embedding_table": table_name,
    }
