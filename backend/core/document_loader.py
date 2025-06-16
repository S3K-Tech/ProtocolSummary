import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import logging
import re
from typing import List, Dict, Any
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.core.config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY, EMBEDDING_MODEL
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def get_qdrant_client(max_retries=3, retry_delay=1):
    """Initialize Qdrant client with retry logic."""
    for attempt in range(max_retries):
        try:
            client = QdrantClient(
                url=QDRANT_URL, 
                api_key=QDRANT_API_KEY,
                timeout=30.0,  # Increased timeout
                prefer_grpc=True
            )
            # Test connection
            client.get_collections()
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Failed to connect to Qdrant (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                logging.error(f"Failed to connect to Qdrant after {max_retries} attempts: {e}")
                raise

# Initialize Qdrant client with retry logic
qdrant_client = get_qdrant_client()

def ensure_collection_exists(collection_name: str, vector_size: int = 1536):
    """Ensure Qdrant collection exists with correct configuration."""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        exists = any(col.name == collection_name for col in collections.collections)
        
        if not exists:
            # Create collection with proper configuration
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logging.info(f"Created collection {collection_name}")
        else:
            logging.info(f"Collection {collection_name} already exists")
            
        return True
    except Exception as e:
        logging.error(f"Error ensuring collection exists: {e}")
        return False

def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from filename."""
    section = extract_section_from_filename(filename)
    metadata = {
        "file_name": filename,
        "section": section
    }
    return metadata

def extract_section_from_filename(filename: str) -> str:
    """Extract section information from filename and content structure."""
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Try to match section patterns like "1.2", "Section 1", "Appendix A", etc.
    patterns = [
        r"(\d+\.\d+)",  # 1.2, 2.3, etc.
        r"section\s*(\d+)",  # Section 1, section 2, etc.
        r"appendix\s*([A-Z])",  # Appendix A, Appendix B, etc.
        r"chapter\s*(\d+)",  # Chapter 1, Chapter 2, etc.
        r"part\s*(\d+)",  # Part 1, Part 2, etc.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name_without_ext, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # If no pattern matches, try to extract meaningful parts from filename
    # Common clinical trial document patterns
    if "protocol" in name_without_ext.lower():
        if "summary" in name_without_ext.lower():
            return "Protocol Summary"
        elif "template" in name_without_ext.lower():
            return "Protocol Template"
        else:
            return "Protocol"
    elif "investigator" in name_without_ext.lower() and "brochure" in name_without_ext.lower():
        return "Investigator's Brochure"
    elif "reference" in name_without_ext.lower():
        return "Reference Protocol"
    elif "consent" in name_without_ext.lower():
        return "Informed Consent"
    elif "case" in name_without_ext.lower() and "report" in name_without_ext.lower():
        return "Case Report Form"
    
    # Default to filename without extension if no pattern matches
    return name_without_ext


def build_index(folder_path: str, collection_name: str) -> VectorStoreIndex:
    """
    Build a vector index from documents in the specified folder.
    
    Args:
        folder_path: Path to the folder containing documents
        collection_name: Name of the Qdrant collection to use
        
    Returns:
        VectorStoreIndex: The built index
    """
    try:
        logging.info(f"Indexing folder: {folder_path} into collection: {collection_name}")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            logging.error(f"Folder {folder_path} does not exist")
            return None
            
        # Ensure collection exists
        if not ensure_collection_exists(collection_name):
            return None
            
        files = os.listdir(folder_path)
        logging.info(f"Found {len(files)} files in folder: {folder_path}")
        
        # Load and process documents
        docs = []
        for filename in files:
            path = os.path.join(folder_path, filename)
            if filename.endswith('.pdf'):
                logging.info(f"Processing PDF: {filename}")
                text = extract_text_from_pdf(path)
            elif filename.endswith('.docx'):
                logging.info(f"Processing DOCX: {filename}")
                text = extract_text_from_docx(path)
            else:
                logging.info(f"Skipping unsupported file: {filename}")
                continue
                
            logging.info(f"Extracted text length for {filename}: {len(text)}")
            if not text or not isinstance(text, str) or not text.strip():
                logging.warning(f"WARNING: Invalid text extracted for file: {filename}")
                continue
                
            # Use simple filename-based section detection (keeps existing behavior)
            section = extract_section_from_filename(filename)
            docs.append(Document(
                text=text, 
                metadata={
                    "section": section,
                    "filename": filename
                }
            ))
            
        logging.info(f"Loaded {len(docs)} documents from {folder_path}")
        
        if not docs:
            logging.error("No valid documents found to index")
            return None
            
        # Initialize embedding model
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        logging.info(f"Using embedding model: {EMBEDDING_MODEL}")
        
        # Chunk docs with proper overlap
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        nodes = splitter.get_nodes_from_documents(docs)
        logging.info(f"Created {len(nodes)} chunks from documents")
        
        if not nodes:
            logging.error("No chunks created from documents")
            return None
            
        # Generate embeddings and upload to Qdrant
        points = []
        for i, node in enumerate(nodes):
            try:
                # Get node content and validate
                content = node.get_content()
                if not content or not isinstance(content, str) or not content.strip():
                    logging.warning(f"Skipping invalid chunk {i}")
                    continue
                    
                # Generate embedding
                try:
                    embedding = embed_model.get_text_embedding(content)
                    if not embedding:
                        logging.warning(f"Failed to generate embedding for chunk {i}")
                        continue
                except Exception as e:
                    logging.error(f"Error generating embedding for chunk {i}: {e}")
                    continue
                
                # Prepare payload
                payload = {
                    "text": content,
                    "metadata": node.metadata or {}
                }
                
                # Add to points list
                points.append(models.PointStruct(
                    id=i,
                    vector=embedding,
                    payload=payload
                ))
                
                if (i + 1) % 100 == 0:
                    logging.info(f"Processed {i + 1} chunks")
                    
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {e}")
                continue
                
        if not points:
            logging.error("No points generated for upload")
            return None
            
        # Upload points to Qdrant
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logging.info(f"Uploaded {len(points)} points to collection {collection_name}")
        except Exception as e:
            logging.error(f"Error uploading points to Qdrant: {e}")
            return None
            
        # Create index for querying
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            enable_hybrid=False,
            vector_size=1536
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        return index
        
    except Exception as e:
        logging.error(f"Error building index: {e}")
        return None

from PyPDF2 import PdfReader
import docx

def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs)

