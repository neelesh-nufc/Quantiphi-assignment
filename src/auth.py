from functools import lru_cache

import yaml

from src.config import USERS_FILE


def _normalise(value: str) -> str:
    return value.strip().lower()


@lru_cache(maxsize=1)
def _load_users() -> dict[str, list[str]]:
    """Load user -> allowed companies mapping from users.yaml (cached)."""
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    users = data.get("users", {}) or {}
    return {
        _normalise(email): [_normalise(c) for c in (companies or [])]
        for email, companies in users.items()
    }


def is_valid_user(email: str) -> bool:
    return _normalise(email) in _load_users()


def get_allowed_companies(email: str) -> list[str]:
    """Return a copy of the user's allowed companies, or [] if unknown (fail closed)."""
    return list(_load_users().get(_normalise(email), []))
