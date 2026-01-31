"""
Shared utility functions for memory-mcp-ce.
"""

import re
import logging
import pandas as pd

# Get logger
logger = logging.getLogger(__name__)


def is_date_label(label: str) -> bool:
    """
    Check if a label appears to be a date.
    
    Uses pandas.to_datetime() which handles many date formats:
    - "jan-2026", "january-2026"
    - "2026-01-31", "01-31-2026"
    - "december-2025", "dec-2025"
    - "jan-7-2026", "7-jan-2026"
    
    This is used to filter date labels from popularity tracking.
    Date labels are useful for temporal queries but pollute trending
    results (e.g., "jan-2026" dominating at 140 count).
    
    Args:
        label: The label string to check
        
    Returns:
        True if the label parses as a date, False otherwise
        
    Note:
        Single month names like "january" won't parse as dates
        (no year context), so they're safe for genuine topics
        like band names or favorite months.
    """
    try:
        pd.to_datetime(label, errors='raise')
        return True
    except (ValueError, TypeError):
        return False


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
        - Date-like labels are filtered out before tokenization (they pollute trending results)
        - Tokenization yielding no tokens results in no-op
        - Caller is responsible for committing the transaction
    """
    if not labels:
        return
    
    # Filter out date-like labels - they're useful for temporal queries
    # but pollute trending results (e.g., "jan-2026" dominating at 140 count)
    non_date_labels = [label for label in labels if not is_date_label(label)]
    
    if not non_date_labels:
        return  # All labels were dates, nothing to track
    
    token_counts = tokenize_labels(non_date_labels)
    
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
