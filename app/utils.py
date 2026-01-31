"""
Shared utility functions for memory-mcp-ce.
"""

import re
import logging

# Get logger
logger = logging.getLogger(__name__)


def tokenize_labels(labels: list[str]) -> dict[str, int]:
    """
    Tokenize an array of labels and return frequency count.
    
    Splits labels on hyphens, underscores, and spaces, then counts
    token frequency across all labels.
    
    Args:
        labels: List of label strings (e.g., ['memory-mcp-ce', 'database schema', 'beer_rules'])
    
    Returns:
        Dictionary of token -> count (e.g., {'memory': 1, 'mcp': 1, 'ce': 1, 'database': 1, ...})
    
    Examples:
        >>> tokenize_labels(['memory-mcp', 'database schema', 'memory_mcp community'])
        {'memory': 2, 'mcp': 2, 'database': 1, 'schema': 1, 'community': 1}
        
        >>> tokenize_labels(['memory--mcp'])  # Multiple consecutive separators
        {'memory': 1, 'mcp': 1}
        
        >>> tokenize_labels(['memory_mcp-ce'])  # Mixed separators
        {'memory': 1, 'mcp': 1, 'ce': 1}
    """
    token_counts: dict[str, int] = {}
    
    for label in labels:
        # Split on hyphen, underscore, or space (handles multiple consecutive separators)
        tokens = re.split(r'[-_\s]+', label.lower())
        
        for token in tokens:
            if token:  # Skip empty strings
                token_counts[token] = token_counts.get(token, 0) + 1
    
    return token_counts


def update_label_token_popularity(namespace: str, labels: list[str], conn) -> None:
    """
    Update label token popularity counts in the database.
    
    Tokenizes the provided labels and performs a batch upsert to the label_tokens
    table. Uses unnest() for efficient single-query updates.
    
    This is a fire-and-forget operation - failures are logged as warnings but
    don't raise exceptions. The caller's primary operation should not fail
    due to token tracking issues.
    
    Args:
        namespace: The namespace for the tokens
        labels: List of label strings to tokenize and track
        conn: Database connection to use (caller manages connection lifecycle)
    
    Note:
        - Empty labels list results in no-op (returns immediately)
        - Tokenization yielding no tokens results in no-op
        - Caller is responsible for committing the transaction
    """
    if not labels:
        return
    
    token_counts = tokenize_labels(labels)
    
    if not token_counts:
        return
    
    try:
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
        
        logger.debug(f"📊 Tracked {len(tokens)} label tokens")
        cur.close()
    except Exception as e:
        logger.warning(f"⚠️ Failed to track label tokens: {e}")
