"""Thread-safe bounded LRU cache for Layer 1 AST diff inspection (100% pure Python stdlib)."""

from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional


def _update_tagged_text(hasher: "hashlib._Hash", tag: bytes, value: Optional[str]) -> None:
    hasher.update(tag)
    if value is None:
        hasher.update(b"\x00")
        return
    encoded = value.encode("utf-8", errors="surrogateescape")
    hasher.update(b"\x01")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def make_cache_key(
    file_path: Optional[str],
    diff_text: Optional[str],
    original_source: Optional[str] = None,
    allow_suppression: bool = True,
    policy_fingerprint: str = "",
) -> str:
    """Compute a deterministic SHA-256 cache key for an AST diff analysis request."""
    hasher = hashlib.sha256()
    _update_tagged_text(hasher, b"fp", str(file_path or ""))
    _update_tagged_text(hasher, b"df", str(diff_text or ""))
    _update_tagged_text(hasher, b"os", original_source)
    hasher.update(b"as\x01" if allow_suppression else b"as\x00")
    _update_tagged_text(hasher, b"pf", str(policy_fingerprint or ""))
    return hasher.hexdigest()


class ASTDiffCache:
    """Thread-safe bounded LRU cache for ``analyze_code_diff`` results."""

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max(0, int(max_entries))
        self._entries: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_cache_key(
        file_path: Optional[str],
        diff_text: Optional[str],
        original_source: Optional[str] = None,
        allow_suppression: bool = True,
        policy_fingerprint: str = "",
    ) -> str:
        """Compute a deterministic SHA-256 cache key for an AST diff analysis request."""
        return make_cache_key(
            file_path=file_path,
            diff_text=diff_text,
            original_source=original_source,
            allow_suppression=allow_suppression,
            policy_fingerprint=policy_fingerprint,
        )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return a deep copy of the cached result if present, updating LRU recency."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(entry)

    def put(self, key: str, result: Dict[str, Any]) -> None:
        """Store a deep copy of ``result`` under ``key``, evicting the oldest entry if full."""
        if self._max_entries <= 0:
            return
        snapshot = copy.deepcopy(result)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = snapshot
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries and reset hit/miss counters."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics (hits, misses, size, max_entries, hit_rate)."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._entries),
                "max_entries": self._max_entries,
                "hit_rate": hit_rate,
            }


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_CACHE: Optional[ASTDiffCache] = None


def get_default_ast_cache() -> ASTDiffCache:
    """Return the process-wide default ``ASTDiffCache`` singleton."""
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_CACHE is None:
                _DEFAULT_CACHE = ASTDiffCache(max_entries=256)
    return _DEFAULT_CACHE


__all__ = [
    "ASTDiffCache",
    "get_default_ast_cache",
    "make_cache_key",
]
