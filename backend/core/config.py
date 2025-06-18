import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

def get_config():
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        'LLM_MODEL': os.getenv('LLM_MODEL', 'gpt-4-turbo'),
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_KEY': os.getenv('SUPABASE_KEY'),
        'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'gpt-4-turbo'),
        'DEFAULT_PROVIDER': os.getenv('DEFAULT_PROVIDER', 'openai'),
        'DEFAULT_TEMPERATURE': float(os.getenv('DEFAULT_TEMPERATURE', '0.2')),
        'MAX_RETRIES': int(os.getenv('MAX_RETRIES', '3')),
        'RETRY_DELAY': int(os.getenv('RETRY_DELAY', '1')),
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'LOG_FORMAT': os.getenv('LOG_FORMAT', '[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    }

# For backward compatibility
config = get_config()
OPENAI_API_KEY = config['OPENAI_API_KEY']
EMBEDDING_MODEL = config['EMBEDDING_MODEL']
LLM_MODEL = config['LLM_MODEL']
SUPABASE_URL = config['SUPABASE_URL']
SUPABASE_KEY = config['SUPABASE_KEY']
DEFAULT_MODEL = config['DEFAULT_MODEL']
DEFAULT_PROVIDER = config['DEFAULT_PROVIDER']
DEFAULT_TEMPERATURE = config['DEFAULT_TEMPERATURE']
MAX_RETRIES = config['MAX_RETRIES']
RETRY_DELAY = config['RETRY_DELAY']
LOG_LEVEL = config['LOG_LEVEL']
LOG_FORMAT = config['LOG_FORMAT']

# Get API keys and verify they exist
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Print first 8 chars of keys for debugging (but only in development)
if os.getenv("DEBUG"):
    print(f"OpenAI API Key loaded (first 8 chars): {OPENAI_API_KEY[:8]}...")
    print(f"Groq API Key loaded (first 8 chars): {GROQ_API_KEY[:8]}...")
