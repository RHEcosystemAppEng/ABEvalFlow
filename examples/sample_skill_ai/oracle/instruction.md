<!-- #ai-generated-oracle -->
# URL Shortener Task

Create a Python module `shortener.py` that implements a simple URL shortener
with in-memory storage.

## Requirements

1. Implement `shorten(url: str) -> str` that:
   - Validates the URL starts with `http://` or `https://`
   - Generates a 6-character alphanumeric short ID
   - Stores the mapping and returns the short ID
   - Raises `ValueError` for invalid URLs

2. Implement `expand(short_id: str) -> str` that:
   - Looks up the short ID and returns the original URL
   - Raises `KeyError` for unknown short IDs

3. Use an in-memory dictionary for storage (no external dependencies).

## Output

Place the implementation in `/workspace/shortener.py`.
