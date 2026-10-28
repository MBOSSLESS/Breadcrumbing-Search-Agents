"""
Simple persistent cache for API calls.
Uses SQLite for efficient storage and retrieval.
"""

import hashlib
import json
import logging
import os
import pickle
import sqlite3
import time
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Persistent cache manager using SQLite."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache manager.

        Args:
            cache_dir: Cache directory; defaults to BREADCRUMBING_CACHE_DIR or ~/.cache/breadcrumbing.
        """
        if cache_dir is None:
            cache_dir = os.getenv(
                "BREADCRUMBING_CACHE_DIR", str(Path.home() / ".cache" / "breadcrumbing")
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "api_cache.db"
        self.sqlite_timeout_seconds = float(
            os.getenv("CACHE_SQLITE_TIMEOUT_SECONDS", "30")
        )
        self.sqlite_busy_timeout_ms = int(
            os.getenv("CACHE_SQLITE_BUSY_TIMEOUT_MS", "30000")
        )
        self.sqlite_max_retries = max(
            1, int(os.getenv("CACHE_SQLITE_MAX_RETRIES", "5"))
        )
        self.sqlite_retry_delay_seconds = float(
            os.getenv("CACHE_SQLITE_RETRY_DELAY_SECONDS", "0.2")
        )

        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self.sqlite_timeout_seconds)
        conn.execute(f"PRAGMA busy_timeout = {self.sqlite_busy_timeout_ms}")
        return conn

    @staticmethod
    def _is_retryable_sqlite_error(exc: sqlite3.Error) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    def _run_sqlite_operation(
        self,
        operation_name: str,
        func: Callable[[], Any],
        *,
        default: Any = None,
        raise_on_failure: bool = False,
    ) -> Any:
        last_exc: Optional[sqlite3.Error] = None
        for attempt in range(1, self.sqlite_max_retries + 1):
            try:
                return func()
            except sqlite3.Error as exc:
                last_exc = exc
                if (
                    attempt >= self.sqlite_max_retries
                    or not self._is_retryable_sqlite_error(exc)
                ):
                    break
                time.sleep(self.sqlite_retry_delay_seconds * attempt)

        logger.warning(
            "Cache %s failed for %s (%s); falling back to uncached path",
            operation_name,
            self.db_path,
            last_exc,
        )
        if raise_on_failure and last_exc is not None:
            raise last_exc
        return default

    def _init_db(self):
        """Initialize SQLite database."""

        def op() -> None:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value BLOB
                    )
                """)
                conn.commit()
            finally:
                conn.close()

        self._run_sqlite_operation("init", op)

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key from function arguments."""
        normalized_args = self._normalize_args_for_key(args)
        cache_data = {
            "func": func_name,
            "args": normalized_args,
            "kwargs": sorted(kwargs.items()) if kwargs else {},
        }
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    @classmethod
    def _normalize_args_for_key(cls, args: tuple) -> tuple:
        if not args:
            return args
        if cls._looks_like_bound_instance(args[0]):
            return args[1:]
        return args

    @staticmethod
    def _looks_like_bound_instance(value: Any) -> bool:
        scalar_types = (str, bytes, int, float, bool, type(None))
        if isinstance(value, scalar_types):
            return False
        if isinstance(value, Mapping):
            return False
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return False
        if isinstance(value, set):
            return False
        if isinstance(value, type):
            return False
        return hasattr(value, "__class__") and hasattr(value, "__dict__")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""

        def op() -> Optional[Any]:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM cache_entries WHERE key = ?", (key,))
                row = cursor.fetchone()
            finally:
                conn.close()

            if row is None:
                return None

            value = row[0]
            return pickle.loads(value)

        return self._run_sqlite_operation("get", op, default=None)

    def set(self, key: str, value: Any):
        """Store value in cache."""

        def op() -> None:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO cache_entries (key, value) VALUES (?, ?)",
                    (key, pickle.dumps(value)),
                )
                conn.commit()
            finally:
                conn.close()

        self._run_sqlite_operation("set", op)

    def delete(self, key: str):
        """Delete entry from cache."""

        def op() -> None:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
            finally:
                conn.close()

        self._run_sqlite_operation("delete", op)

    def clear(self):
        """Clear all cache entries."""

        def op() -> None:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries")
                conn.commit()
            finally:
                conn.close()

        self._run_sqlite_operation("clear", op)

    def _is_error_result(self, result: Any) -> bool:
        """
        Check if result indicates an error/failure.

        Returns:
            True if result is None or contains error indicators
        """
        if result is None:
            return True
        if isinstance(result, str):
            error_patterns = [
                "[visit] Failed",
                "[search] Failed",
                "[Cache Error]",
            ]
            return any(pattern in result for pattern in error_patterns)
        if isinstance(result, dict):
            if result.get("is_error") is True:
                return True
            content = result.get("content")
            if isinstance(content, str):
                error_patterns = [
                    "[visit] Failed",
                    "[search] Failed",
                    "[Cache Error]",
                ]
                return any(pattern in content for pattern in error_patterns)
        return False

    def cached(self, func: Callable) -> Callable:
        """Cache successful API results; retry failed requests on the next call."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self._generate_key(func.__name__, args, kwargs)
            cached_result = self.get(key)
            if cached_result is not None:
                return cached_result
            result = func(*args, **kwargs)
            if not self._is_error_result(result):
                self.set(key, result)
            return result

        return wrapper


# Initialize storage only when a tool actually accesses the cache.
class _LazyCache:
    def __init__(self):
        self._instance = None

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = CacheManager()
        return getattr(self._instance, name)


_search_cache = _LazyCache()
_visit_cache = _LazyCache()


def cached_search(func: Callable) -> Callable:
    """
    Decorator for search API calls.
    Does NOT cache failed results (None, error messages).
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return _search_cache.cached(func)(*args, **kwargs)

    return wrapper


def cached_visit(func: Callable) -> Callable:
    """
    Decorator for visit API calls.
    Does NOT cache failed results (error messages).
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return _visit_cache.cached(func)(*args, **kwargs)

    return wrapper
