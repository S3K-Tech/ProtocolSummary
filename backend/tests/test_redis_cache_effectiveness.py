import time

from backend.core.generator import get_chunk_previews, get_chunks_by_ids, get_or_cache_embedding
from backend.core.config import EMBEDDING_MODEL

# Example test values (adjust as needed for your data)
COLLECTION = "ps-index"
CHUNK_IDS = ["14", "15"]  # Use actual chunk IDs from your Supabase table
PREVIEW_LEN = 200
TEST_TEXT = "This is a test sentence for embedding."

print("\n--- Redis Cache Effectiveness Test ---\n")

def test_cache(func, *args, label=None):
    print(f"Testing {label or func.__name__}...")
    t1 = time.time()
    result1 = func(*args)
    t2 = time.time()
    print(f"  First call: {t2-t1:.4f} seconds")
    t3 = time.time()
    result2 = func(*args)
    t4 = time.time()
    print(f"  Second call: {t4-t3:.4f} seconds")
    if (t4-t3) < (t2-t1):
        print("  ✅ Second call was faster (cache effective)")
    else:
        print("  ⚠️  Second call was not faster (cache may not be effective)")
    print()
    return result1, result2

# Test chunk previews
try:
    test_cache(get_chunk_previews, COLLECTION, PREVIEW_LEN, label="get_chunk_previews")
except Exception as e:
    print(f"Error testing get_chunk_previews: {e}")

# Test chunk content by IDs
try:
    test_cache(get_chunks_by_ids, COLLECTION, CHUNK_IDS, label="get_chunks_by_ids")
except Exception as e:
    print(f"Error testing get_chunks_by_ids: {e}")

# Test embedding caching
try:
    test_cache(get_or_cache_embedding, TEST_TEXT, EMBEDDING_MODEL, label="get_or_cache_embedding")
except Exception as e:
    print(f"Error testing get_or_cache_embedding: {e}")

print("--- Test Complete ---\n") 