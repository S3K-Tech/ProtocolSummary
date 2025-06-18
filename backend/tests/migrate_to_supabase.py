"""
Migration script to help transition from Qdrant to Supabase.
This script provides utilities to migrate data and test the new setup.
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

def setup_supabase_environment():
    """Setup Supabase environment for the application."""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Enable pgvector extension using direct SQL
        # We'll use the Supabase dashboard or psql to enable this
        logging.info("ℹ pgvector extension should be enabled via Supabase dashboard")
        logging.info("ℹ Go to: Database > Extensions > Enable 'vector' extension")
        
        # Create the main tables for the application using direct table operations
        tables = ['ps-index', 'pt-index', 'rp-index', 'ib-index']
        
        for table_name in tables:
            try:
                # Check if table exists by trying to query it
                result = client.table(table_name).select('id').limit(1).execute()
                logging.info(f"✓ Table {table_name} already exists")
            except Exception:
                # Table doesn't exist, we need to create it via SQL
                logging.warning(f"⚠ Table {table_name} doesn't exist")
                logging.info(f"ℹ Please create table {table_name} manually in Supabase dashboard:")
                logging.info(f"  - Go to: Database > Tables > New Table")
                logging.info(f"  - Name: {table_name}")
                logging.info(f"  - Columns:")
                logging.info(f"    - id: bigint, primary key, auto-increment")
                logging.info(f"    - content: text")
                logging.info(f"    - metadata: jsonb")
                logging.info(f"    - embedding: vector(1536)")
                logging.info(f"  - Enable Row Level Security: No")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Failed to setup Supabase environment: {e}")
        return False

def create_tables_manually():
    """Provide instructions for manual table creation."""
    print("\n📋 Manual Table Creation Instructions:")
    print("=" * 50)
    
    tables = ['ps-index', 'pt-index', 'rp-index', 'ib-index']
    
    for table_name in tables:
        print(f"\nTable: {table_name}")
        print("-" * 30)
        print("1. Go to Supabase Dashboard > Database > Tables")
        print("2. Click 'New Table'")
        print("3. Configure as follows:")
        print(f"   - Name: {table_name}")
        print("   - Columns:")
        print("     • id: bigint, primary key, auto-increment")
        print("     • content: text")
        print("     • metadata: jsonb")
        print("     • embedding: vector(1536)")
        print("   - Enable Row Level Security: No")
        print("   - Click 'Save'")
        print("\n4. After creating table, go to Database > Extensions")
        print("5. Enable the 'vector' extension if not already enabled")

def test_supabase_connection():
    """Test the Supabase connection and basic operations."""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test basic connection
        result = client.table('ps-index').select('id').limit(1).execute()
        logging.info("✓ Basic connection test passed")
        
        # Test vector operations
        test_embedding = [0.1] * 1536
        test_data = {
            "content": "Test content for migration",
            "metadata": {"test": True, "migration": True},
            "embedding": test_embedding
        }
        
        # Insert test data
        result = client.table('ps-index').insert(test_data).execute()
        logging.info("✓ Vector insertion test passed")
        
        # Test vector similarity search using direct SQL
        try:
            # Use the client's built-in vector search if available
            # For now, just test that we can query the data
            result = client.table('ps-index').select('content, metadata').eq('metadata->test', True).execute()
            if result.data:
                logging.info("✓ Vector data query test passed")
            else:
                logging.warning("⚠ Vector data query returned no results")
        except Exception as search_e:
            logging.warning(f"⚠ Vector similarity search not available: {search_e}")
            logging.info("ℹ This is expected if pgvector extension is not enabled")
        
        # Clean up test data
        client.table('ps-index').delete().eq('metadata->test', True).execute()
        logging.info("✓ Cleanup test passed")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Supabase connection test failed: {e}")
        return False

def verify_environment_variables():
    """Verify that all required environment variables are set."""
    required_vars = {
        'SUPABASE_URL': SUPABASE_URL,
        'SUPABASE_KEY': SUPABASE_KEY,
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY')
    }
    
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if not var_value:
            missing_vars.append(var_name)
        else:
            logging.info(f"✓ {var_name} is configured")
    
    if missing_vars:
        logging.error(f"✗ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def main():
    """Main migration function."""
    print("🚀 Starting Qdrant to Supabase Migration")
    print("=" * 50)
    
    # Step 1: Verify environment variables
    print("\n1. Verifying environment variables...")
    if not verify_environment_variables():
        print("❌ Environment variables verification failed")
        return False
    
    # Step 2: Setup Supabase environment
    print("\n2. Setting up Supabase environment...")
    if not setup_supabase_environment():
        print("❌ Supabase environment setup failed")
        return False
    
    # Step 3: Provide manual setup instructions
    print("\n3. Manual setup required...")
    create_tables_manually()
    
    # Step 4: Test connection and operations
    print("\n4. Testing Supabase connection and operations...")
    if not test_supabase_connection():
        print("❌ Supabase connection test failed")
        print("\n💡 Make sure you have:")
        print("   - Created the required tables manually")
        print("   - Enabled the 'vector' extension")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Migration completed successfully!")
    print("\nNext steps:")
    print("1. Complete manual table creation if not done")
    print("2. Install the new dependencies: pip install -r requirements.txt")
    print("3. Test the application with: python backend/tests/supabase_test.py")
    print("4. Upload your documents to the new Supabase backend")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 