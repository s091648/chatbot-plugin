"""Load LLM provider configuration from providers.toml."""

import os
import tomllib
from typing import Any


def load_providers(path: str | None = None) -> list[dict[str, Any]]:
    """Load provider definitions from providers.toml, sorted by priority.

    Args:
        path: Path to the TOML file. Defaults to providers.toml in the
              project root (3 levels up from this file).

    Returns:
        List of provider config dicts sorted by "priority" ascending.

    Raises:
        FileNotFoundError: If the TOML file does not exist.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "providers.toml",
        )

    with open(path, "rb") as f:
        data = tomllib.load(f)

    providers = data.get("providers", [])
    return sorted(providers, key=lambda p: p.get("priority", 999))
