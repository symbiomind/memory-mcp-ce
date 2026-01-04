
from openai import OpenAI
from app.config import EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_DIMS

# Module-level cache for embedding dimension validation
# Prevents redundant API calls on every retrieve_memories query
_validated_embedding_model = None
_validated_embedding_dims = None

client = OpenAI(
    base_url=EMBEDDING_URL,
    api_key=EMBEDDING_API_KEY or "dummy-key", # a dummy key is required for the client to work
    #default_headers={
    #    "HTTP-Referer": "https://your-mcp-ce-saas.com",  # Your site URL
    #    "X-Title": "MCP-CE Memory Platform",  # Your site name
    #}
)

def get_embedding_dimension() -> int:
    """
    Detects the embedding dimension by sending a test request and validates it's actually a vector.
    
    Uses module-level caching to avoid redundant API calls on every retrieve_memories query.
    Re-validates only when EMBEDDING_MODEL changes.
    
    If EMBEDDING_DIMS is set:
      - Passes dimensions parameter to API (for MRL models like Qwen)
      - CRITICAL: Validates returned dimensions match EMBEDDING_DIMS
      - Mismatch = startup failure (prevents wrong-sized vectors in memory_{dims} tables)
    
    If EMBEDDING_DIMS is not set:
      - Uses model's native output dimensions
    """
    global _validated_embedding_model, _validated_embedding_dims
    
    # Return cached result if model hasn't changed
    if _validated_embedding_model == EMBEDDING_MODEL and _validated_embedding_dims is not None:
        return _validated_embedding_dims
    
    try:
        # Build API call - only include dimensions if EMBEDDING_DIMS is set
        api_kwargs = {
            "model": EMBEDDING_MODEL,
            "input": "test",
        }
        if EMBEDDING_DIMS is not None:
            api_kwargs["dimensions"] = EMBEDDING_DIMS
        
        response = client.embeddings.create(**api_kwargs)
        
        # Check we got data back
        if not response.data or len(response.data) == 0:
            raise ValueError(f"Model '{EMBEDDING_MODEL}' returned no embeddings")
        
        embedding = response.data[0].embedding
        
        # Validate it's actually a list/array of numbers
        if not isinstance(embedding, (list, tuple)):
            raise TypeError(
                f"Expected embedding to be a list/array, got {type(embedding).__name__}. "
                f"Is '{EMBEDDING_MODEL}' actually an embedding model?"
            )
        
        actual_dims = len(embedding)
        
        # Check dimensions are sane
        if actual_dims == 0:
            raise ValueError(f"Model returned empty embedding vector")
        
        # Verify it's actually numeric values
        if not all(isinstance(x, (int, float)) for x in embedding[:5]):  # spot check first 5
            raise TypeError(
                f"Embedding contains non-numeric values. "
                f"Is '{EMBEDDING_MODEL}' configured correctly?"
            )
        
        # CRITICAL: Validate dimensions match if EMBEDDING_DIMS was specified
        if EMBEDDING_DIMS is not None and actual_dims != EMBEDDING_DIMS:
            raise ValueError(
                f"EMBEDDING_DIMS={EMBEDDING_DIMS} requested but model '{EMBEDDING_MODEL}' "
                f"returned {actual_dims} dimensions.\n"
                f"Either remove EMBEDDING_DIMS to use native dimensions, or use an MRL-capable "
                f"model (like Qwen) that supports the requested dimension."
            )
        
        # Cache the validated result
        _validated_embedding_model = EMBEDDING_MODEL
        _validated_embedding_dims = actual_dims
        
        if EMBEDDING_DIMS is not None:
            print(f"✓ Validated embedding model: {EMBEDDING_MODEL} ({actual_dims}D vector, EMBEDDING_DIMS={EMBEDDING_DIMS})")
        else:
            print(f"✓ Validated embedding model: {EMBEDDING_MODEL} ({actual_dims}D vector)")
        return actual_dims
        
    except Exception as e:
        raise Exception(
            f"Failed to validate embedding model '{EMBEDDING_MODEL}': {str(e)}\n"
            f"Ensure EMBEDDING_URL points to a valid embedding endpoint."
        )

def get_embedding(text: str) -> list[float]:
    """
    Gets an embedding for the given text.
    
    If EMBEDDING_DIMS is set, passes dimensions parameter to API.
    """
    try:
        # Build API call - only include dimensions if EMBEDDING_DIMS is set
        api_kwargs = {
            "model": EMBEDDING_MODEL,
            "input": text,
        }
        if EMBEDDING_DIMS is not None:
            api_kwargs["dimensions"] = EMBEDDING_DIMS
        
        response = client.embeddings.create(**api_kwargs)
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Failed to get embedding: {str(e)}")
