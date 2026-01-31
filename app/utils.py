"""
Shared utility functions for memory-mcp-ce.
"""

import re


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
