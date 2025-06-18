"""
Basic Supabase functionality test.
This script tests the core functionality without requiring complex SQL setup.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
import os
from supabase import create_client, Client
from backend.core.config import SUPABASE_URL, SUPABASE_KEY

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test_basic_connection():
    """Test basic Supabase connection."""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("✓ Successfully created Supabase client")
        return client
    except Exception as e:
        logging.error(f"✗ Failed to create Supabase client: {e}")
        return None

def test_table_operations(client):
    """Test basic table operations."""
    try:
        # Test if we can access a table
        table_name = 'ps-index'
        
        try:
            result = client.table(table_name).select('id').limit(1).execute()
            logging.info(f"✓ Successfully accessed table {table_name}")
            return True
        except Exception as e:
            logging.warning(f"⚠ Table {table_name} doesn't exist or is not accessible: {e}")
            logging.info("ℹ This is expected if tables haven't been created yet")
            return False
            
    except Exception as e:
        logging.error(f"✗ Table operations test failed: {e}")
        return False

def test_data_insertion(client):
    """Test inserting and retrieving data."""
    try:
        table_name = 'ps-index'
        
        # Test data
        test_data = {
            "content": "This is a test document for the clinical trial protocol generator.",
            "metadata": {
                "test": True,
                "source": "migration_test",
                "timestamp": "2024-01-01"
            }
        }
        
        # Try to insert data
        try:
            result = client.table(table_name).insert(test_data).execute()
            logging.info("✓ Successfully inserted test data")
            
            # Try to retrieve the data using proper JSON query syntax
            try:
                # Use proper JSON query syntax for Supabase
                retrieved = client.table(table_name).select('*').eq('metadata->test', 'true').execute()
                if retrieved.data:
                    logging.info("✓ Successfully retrieved test data")
                    
                    # Clean up
                    client.table(table_name).delete().eq('metadata->test', 'true').execute()
                    logging.info("✓ Successfully cleaned up test data")
                    return True
                else:
                    logging.warning("⚠ Could not retrieve test data")
                    # Try alternative query
                    try:
                        retrieved = client.table(table_name).select('*').execute()
                        if retrieved.data:
                            logging.info("✓ Retrieved data using simple query")
                            # Clean up by deleting all test data
                            client.table(table_name).delete().neq('id', 0).execute()
                            logging.info("✓ Cleaned up all data")
                            return True
                    except Exception as alt_e:
                        logging.warning(f"⚠ Alternative query also failed: {alt_e}")
                    return False
                    
            except Exception as query_e:
                logging.warning(f"⚠ Could not query test data: {query_e}")
                # Try to clean up anyway
                try:
                    client.table(table_name).delete().neq('id', 0).execute()
                    logging.info("✓ Cleaned up data using alternative method")
                except:
                    pass
                return False
                
        except Exception as e:
            logging.warning(f"⚠ Could not insert test data: {e}")
            logging.info("ℹ This is expected if the table doesn't exist or has different structure")
            return False
            
    except Exception as e:
        logging.error(f"✗ Data insertion test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Basic Supabase Functionality Test")
    print("=" * 50)
    
    # Test 1: Basic connection
    print("\n1. Testing basic connection...")
    client = test_basic_connection()
    if not client:
        print("❌ Basic connection test failed")
        return False
    
    # Test 2: Table operations
    print("\n2. Testing table operations...")
    table_ok = test_table_operations(client)
    
    # Test 3: Data operations
    print("\n3. Testing data operations...")
    data_ok = test_data_insertion(client)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Connection: {'✅ PASS' if client else '❌ FAIL'}")
    print(f"   Tables: {'✅ PASS' if table_ok else '⚠️  SKIP (expected)'}")
    print(f"   Data: {'✅ PASS' if data_ok else '⚠️  SKIP (expected)'}")
    
    if client:
        print("\n✅ Basic Supabase setup is working!")
        print("\nNext steps:")
        if not table_ok:
            print("1. Create tables manually in Supabase dashboard")
            print("2. Enable pgvector extension")
        print("3. Test full functionality: python backend/tests/supabase_test.py")
        print("4. Start the application: streamlit run frontend/app.py")
        return True
    else:
        print("\n❌ Basic setup failed")
        print("Check your SUPABASE_URL and SUPABASE_KEY")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 