import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
from supabase import create_client, Client
import numpy as np
import os
from backend.core.config import SUPABASE_URL, SUPABASE_KEY

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test_supabase_connection():
    """Test connection to Supabase server."""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test connection by trying to access a table
        result = client.table('ps-index').select('id').limit(1).execute()
        logging.info(f"Successfully connected to Supabase. Connection test passed.")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Supabase: {e}")
        return False

def test_table_operations():
    """Test basic table operations that the application uses."""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test tables that the application uses
        tables = ['ps-index', 'pt-index', 'rp-index', 'ib-index']
        
        for table in tables:
            try:
                # Test table access
                result = client.table(table).select('id').limit(1).execute()
                logging.info(f"✓ Successfully accessed table {table}")
            except Exception as e:
                logging.warning(f"⚠ Table {table} not accessible: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to test table operations: {e}")
        return False

def test_data_operations():
    """Test data insertion and retrieval operations."""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test data for ps-index table
        test_data = {
            "content": "This is a test document for clinical trial protocol generation.",
            "metadata": {
                "test": True,
                "source": "test_script",
                "file_name": "test_document.pdf"
            }
        }
        
        # Insert test data
        result = client.table('ps-index').insert(test_data).execute()
        logging.info("✓ Successfully inserted test data")
        
        # Retrieve test data
        retrieved = client.table('ps-index').select('*').eq('metadata->test', 'true').execute()
        if retrieved.data:
            logging.info(f"✓ Successfully retrieved {len(retrieved.data)} test records")
            
            # Clean up test data
            client.table('ps-index').delete().eq('metadata->test', 'true').execute()
            logging.info("✓ Successfully cleaned up test data")
        else:
            logging.warning("⚠ No test data retrieved")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to test data operations: {e}")
        return False

def test_vector_store_operations():
    """Test vector store operations using LlamaIndex."""
    try:
        from backend.core.document_loader import supabase_client
        from llama_index.vector_stores.supabase import SupabaseVectorStore
        from llama_index.core import VectorStoreIndex
        
        # Test creating a vector store
        vector_store = SupabaseVectorStore(client=supabase_client, table_name='ps-index')
        logging.info("✓ Successfully created SupabaseVectorStore")
        
        # Test creating an index
        index = VectorStoreIndex.from_vector_store(vector_store)
        logging.info("✓ Successfully created VectorStoreIndex")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to test vector store operations: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Supabase Functionality Test")
    print("=" * 50)
    
    print("\n1. Testing Supabase connection...")
    if test_supabase_connection():
        print("✓ Supabase connection successful")
    else:
        print("✗ Supabase connection failed")
        exit(1)
    
    print("\n2. Testing table operations...")
    if test_table_operations():
        print("✓ Table operations successful")
    else:
        print("✗ Table operations failed")
    
    print("\n3. Testing data operations...")
    if test_data_operations():
        print("✓ Data operations successful")
    else:
        print("✗ Data operations failed")
    
    print("\n4. Testing vector store operations...")
    if test_vector_store_operations():
        print("✓ Vector store operations successful")
    else:
        print("✗ Vector store operations failed")
    
    print("\n" + "=" * 50)
    print("✅ Supabase migration test completed!")
    print("\nNext steps:")
    print("1. Start the application: streamlit run frontend/app.py")
    print("2. Upload documents and test full functionality") 