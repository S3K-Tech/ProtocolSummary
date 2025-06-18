"""
Simple migration script for Qdrant to Supabase transition.
This script provides instructions and tests the basic setup.
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

def test_supabase_connection():
    """Test the Supabase connection."""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test basic connection by trying to access a table
        # This will fail if tables don't exist, but that's expected
        try:
            result = client.table('ps-index').select('id').limit(1).execute()
            logging.info("✓ Basic connection test passed")
            return True
        except Exception as table_error:
            logging.warning(f"⚠ Tables don't exist yet: {table_error}")
            logging.info("ℹ This is expected - you need to create tables manually")
            return True
            
    except Exception as e:
        logging.error(f"✗ Supabase connection test failed: {e}")
        return False

def print_setup_instructions():
    """Print detailed setup instructions."""
    print("\n📋 SUPABASE SETUP INSTRUCTIONS")
    print("=" * 50)
    
    print("\n1. ENABLE PGVECTOR EXTENSION:")
    print("   - Go to Supabase Dashboard > Database > Extensions")
    print("   - Find 'vector' extension and click 'Enable'")
    print("   - Wait for it to be enabled")
    
    print("\n2. CREATE TABLES:")
    tables = ['ps-index', 'pt-index', 'rp-index', 'ib-index']
    
    for table_name in tables:
        print(f"\n   Table: {table_name}")
        print("   - Go to Database > Tables > New Table")
        print("   - Configure as follows:")
        print(f"     • Name: {table_name}")
        print("     • Columns:")
        print("       - id: bigint, primary key, auto-increment")
        print("       - content: text")
        print("       - metadata: jsonb")
        print("       - embedding: vector(1536)")
        print("     • Enable Row Level Security: No")
        print("     • Click 'Save'")
    
    print("\n3. CREATE VECTOR INDEXES (Optional but recommended):")
    print("   - Go to Database > SQL Editor")
    print("   - Run this SQL for each table:")
    for table_name in tables:
        print(f"     CREATE INDEX {table_name}_embedding_idx ON \"{table_name}\" USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
    
    print("\n4. TEST THE SETUP:")
    print("   - Run: python backend/tests/supabase_test.py")

def main():
    """Main migration function."""
    print("🚀 Qdrant to Supabase Migration Helper")
    print("=" * 50)
    
    # Step 1: Verify environment variables
    print("\n1. Verifying environment variables...")
    if not verify_environment_variables():
        print("❌ Environment variables verification failed")
        print("\n💡 Make sure your .env file contains:")
        print("   SUPABASE_URL=your-supabase-url")
        print("   SUPABASE_KEY=your-supabase-anon-key")
        print("   OPENAI_API_KEY=your-openai-key")
        print("   GROQ_API_KEY=your-groq-key")
        return False
    
    # Step 2: Test basic connection
    print("\n2. Testing Supabase connection...")
    if not test_supabase_connection():
        print("❌ Supabase connection test failed")
        print("\n💡 Check your SUPABASE_URL and SUPABASE_KEY")
        return False
    
    # Step 3: Print setup instructions
    print("\n3. Manual setup required...")
    print_setup_instructions()
    
    print("\n" + "=" * 50)
    print("✅ Environment verification completed!")
    print("\nNext steps:")
    print("1. Follow the setup instructions above")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Test with: python backend/tests/supabase_test.py")
    print("4. Start the app: streamlit run frontend/app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 