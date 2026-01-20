"""
Utility functions for file operations and string sanitization
"""
import re
import os


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """
    Sanitize a chemical name or identifier to create a safe filename.

    Args:
        name: Chemical name or identifier
        max_length: Maximum length for filename (default 200)

    Returns:
        Sanitized filename string

    Examples:
        >>> sanitize_filename("carbon dioxide")
        'carbon_dioxide'
        >>> sanitize_filename("2,3-Dimethylbutane")
        '2_3_dimethylbutane'
    """
    # Convert to lowercase
    sanitized = name.lower()

    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')

    # Remove special characters, keep only alphanumeric and underscores
    sanitized = re.sub(r'[^a-z0-9_]', '_', sanitized)

    # Replace multiple underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')

    return sanitized


def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def get_cache_path(query: str, cache_dir: str = "cache") -> str:
    """
    Get the full cache file path for a query.

    Args:
        query: Chemical name or identifier
        cache_dir: Cache directory path (default "cache")

    Returns:
        Full path to cache file
    """
    filename = sanitize_filename(query) + ".json"
    return os.path.join(cache_dir, filename)
