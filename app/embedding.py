
from openai import OpenAI
from app.config import EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY

client = OpenAI(
    base_url=EMBEDDING_URL,
    api_key=EMBEDDING_API_KEY or "dummy-key", # a dummy key is required for the client to work
    #default_headers={
    #    "HTTP-Referer": "https://your-mcp-ce-saas.com",  # Your site URL
    #    "X-Title": "MCP-CE Memory Platform",  # Your site name
    #}
)

def get_embedding_dimension() -> int:
    """Detects the embedding dimension by sending a test request and validates it's actually a vector."""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input="test",
        )
        
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
        
        print(f"✓ Validated embedding model: {EMBEDDING_MODEL} ({actual_dims}D vector)")
        return actual_dims
        
    except Exception as e:
        raise Exception(
            f"Failed to validate embedding model '{EMBEDDING_MODEL}': {str(e)}\n"
            f"Ensure EMBEDDING_URL points to a valid embedding endpoint."
        )

def get_embedding(text: str) -> list[float]:
    """Gets an embedding for the given text."""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Failed to get embedding: {str(e)}")
