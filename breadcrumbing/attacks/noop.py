from .base import BaseAttacker


class NoOpAttacker(BaseAttacker):
    """No-op attacker that passes through all responses unchanged."""

    def hook_search(self, query: str, response: dict) -> dict:
        return response

    def hook_visit(self, url: str, response: str) -> str:
        return response
