import redis
import hashlib
import json
import os
from fastapi import HTTPException
import urllib.parse

# You can move these to config.py if you want
REDIS_URL = os.getenv("REDIS_URL")
print(f"[Redis] REDIS_URL from env: {os.getenv('REDIS_URL')}")
try:
    if REDIS_URL:
        r = redis.from_url(REDIS_URL)
    else:
        # fallback to local dev
        r = redis.Redis(host="localhost", port=6379, db=0)
    # Test connection
    r.ping()
    print("[Redis] Connected successfully.")
except Exception as e:
    print(f"[Redis] Connection failed: {e}")
    r = None

def make_cache_key(prefix: str, data: str):
    return f"{prefix}:{hashlib.sha256(data.encode()).hexdigest()}"

def get_cache(prefix: str, data: str):
    if r is None:
        raise HTTPException(status_code=500, detail="Redis is not connected.")
    key = make_cache_key(prefix, data)
    result = r.get(key)
    if result:
        print(f"[Redis] Cache HIT for {prefix}:{key}")
        return json.loads(result)
    print(f"[Redis] Cache MISS for {prefix}:{key}")
    return None

def set_cache(prefix: str, data: str, value, expire_seconds=3600):
    if r is None:
        raise HTTPException(status_code=500, detail="Redis is not connected.")
    key = make_cache_key(prefix, data)
    r.setex(key, expire_seconds, json.dumps(value))

def check_rate_limit(key: str, limit: int, window_seconds: int):
    if r is None:
        raise HTTPException(status_code=500, detail="Redis is not connected.")
    current = r.incr(key)
    if current == 1:
        r.expire(key, window_seconds)
    if current > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
