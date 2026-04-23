# URL Shortener Skill

When asked to create a URL shortener module, follow these guidelines:

- Provide a `shorten(url: str) -> str` function that returns a shortened identifier
- Provide an `expand(short_id: str) -> str` function that returns the original URL
- Use an in-memory dictionary for storage
- Shortened IDs should be 6 alphanumeric characters
- Raise `ValueError` for invalid URLs (must start with `http://` or `https://`)
- Raise `KeyError` when expanding an unknown short ID
- Keep the implementation in a single `shortener.py` module
