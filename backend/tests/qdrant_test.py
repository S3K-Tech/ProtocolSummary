import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
from qdrant_client import QdrantClient
import numpy as np
import os
from backend.core.config import QDRANT_URL, QDRANT_API_KEY

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test_qdrant_connection():
    """Test connection to Qdrant server."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        logging.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections.")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant: {e}")
        return False

if __name__ == "__main__":
    test_qdrant_connection()
