import os
import pickle
import time
from copy import deepcopy
from threading import Lock
from typing import Any, Dict, Optional, Tuple


_CACHE_PREFIX = "grey:cache"
_MEMORY_CACHE: Dict[str, Tuple[float, Any]] = {}
_MEMORY_LOCK = Lock()
_REDIS_CLIENT = None
_REDIS_INIT_DONE = False
_REDIS_INIT_LOCK = Lock()


def _redis_url() -> Optional[str]:
    value = os.getenv("REDIS_URL", "").strip()
    return value or None


def _get_redis_client():
    global _REDIS_CLIENT, _REDIS_INIT_DONE
    if _REDIS_INIT_DONE:
        return _REDIS_CLIENT

    with _REDIS_INIT_LOCK:
        if _REDIS_INIT_DONE:
            return _REDIS_CLIENT

        url = _redis_url()
        if not url:
            _REDIS_INIT_DONE = True
            _REDIS_CLIENT = None
            return None

        try:
            import redis

            client = redis.Redis.from_url(
                url,
                socket_connect_timeout=0.15,
                socket_timeout=0.15,
                retry_on_timeout=False,
            )
            client.ping()
            _REDIS_CLIENT = client
        except Exception:
            _REDIS_CLIENT = None

        _REDIS_INIT_DONE = True
        return _REDIS_CLIENT


def build_cache_key(namespace: str, key: str) -> str:
    return f"{_CACHE_PREFIX}:{namespace}:{key}"


def cache_get(namespace: str, key: str):
    final_key = build_cache_key(namespace, key)

    client = _get_redis_client()
    if client is not None:
        try:
            raw = client.get(final_key)
            if raw is None:
                return None
            return pickle.loads(raw)
        except Exception:
            # Fall back to memory cache when Redis is unavailable.
            pass

    now = time.monotonic()
    with _MEMORY_LOCK:
        item = _MEMORY_CACHE.get(final_key)
        if not item:
            return None
        expires_at, payload = item
        if expires_at <= now:
            _MEMORY_CACHE.pop(final_key, None)
            return None
        return deepcopy(payload)


def cache_set(namespace: str, key: str, value: Any, ttl_seconds: int):
    ttl = max(1, int(ttl_seconds))
    final_key = build_cache_key(namespace, key)

    client = _get_redis_client()
    if client is not None:
        try:
            client.setex(final_key, ttl, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            return
        except Exception:
            pass

    with _MEMORY_LOCK:
        _MEMORY_CACHE[final_key] = (time.monotonic() + ttl, deepcopy(value))


def cache_invalidate_exact(namespace: str, key: str):
    final_key = build_cache_key(namespace, key)

    client = _get_redis_client()
    if client is not None:
        try:
            client.delete(final_key)
        except Exception:
            pass

    with _MEMORY_LOCK:
        _MEMORY_CACHE.pop(final_key, None)


def cache_invalidate_prefix(namespace: str, prefix: str = ""):
    namespaced_prefix = build_cache_key(namespace, prefix)

    client = _get_redis_client()
    if client is not None:
        try:
            cursor = 0
            pattern = f"{namespaced_prefix}*"
            while True:
                cursor, keys = client.scan(cursor=cursor, match=pattern, count=200)
                if keys:
                    client.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass

    with _MEMORY_LOCK:
        for key in list(_MEMORY_CACHE.keys()):
            if key.startswith(namespaced_prefix):
                _MEMORY_CACHE.pop(key, None)
