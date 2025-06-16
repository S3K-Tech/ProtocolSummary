import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
from qdrant_client import QdrantClient
from backend.core.config import QDRANT_URL, QDRANT_API_KEY

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def delete_all_collections():
    """Delete all collections from Qdrant."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        for collection in collections.collections:
            client.delete_collection(collection.name)
            logging.info(f"Deleted collection: {collection.name}")
        logging.info("All collections deleted successfully.")
        return True
    except Exception as e:
        logging.error(f"Error deleting collections: {e}")
        return False

if __name__ == "__main__":
    delete_all_collections()
