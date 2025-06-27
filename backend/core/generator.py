import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.document_loader import build_index, supabase_client
from backend.core.config import OPENAI_API_KEY, GROQ_API_KEY, EMBEDDING_MODEL, SUPABASE_KEY, SUPABASE_URL
# Lazy-load engines to avoid auto-indexing at import time
_engine_cache = {}
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding

import logging
import os
import requests
import json
import tiktoken
import re

from backend.utils.redis_utils import get_cache, set_cache

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def get_engine(collection, folder):
    if collection not in _engine_cache:
        # Check if collection exists and has data
        try:
            result = supabase_client.table(collection).select('id').limit(1).execute()
            if result.data and len(result.data) > 0:
                # Create a simple query engine that uses direct Supabase queries
                from llama_index.core import VectorStoreIndex, Document
                from llama_index.embeddings.openai import OpenAIEmbedding
                
                embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
                dummy_doc = Document(text="dummy", metadata={})
                idx = VectorStoreIndex.from_documents([dummy_doc], embed_model=embed_model)
                _engine_cache[collection] = idx.as_query_engine()
                return _engine_cache[collection]
        except Exception as e:
            logging.error(f"Table {collection} not found or error: {e}")
        # If not exists or empty, build and upload
        idx = build_index(folder, collection)
        _engine_cache[collection] = idx.as_query_engine() if idx else None
    return _engine_cache[collection]

def get_chunk_previews(collection: str, preview_len: int = 200) -> List[Dict[str, Any]]:
    """
    Get chunk previews for a collection, with Redis caching.
    Cache key is based on collection and preview length.
    """
    if not collection:
        return []
    cache_key = f"chunk_previews:{collection}:{preview_len}"
    try:
        cached = get_cache("chunk_previews", cache_key)
        if cached:
            return cached
    except Exception as e:
        logging.warning(f"Redis unavailable for chunk previews: {e}")
    previews = []
    try:
        result = supabase_client.table(collection).select('id, content, metadata').limit(1000).execute()
        for row in result.data:
            chunk_id = row.get("id")
            content = row.get("content", "")
            metadata = row.get("metadata", {})
            previews.append({
                "id": chunk_id,
                "preview": content[:preview_len],
                "metadata": metadata
            })
    except Exception as e:
        logging.error(f"Error fetching chunk previews: {e}")
    try:
        set_cache("chunk_previews", cache_key, previews)
    except Exception as e:
        logging.warning(f"Redis unavailable for setting chunk previews: {e}")
    return previews

def get_chunks_by_ids(collection: str, chunk_ids: List[str]) -> List[str]:
    """
    Get chunks by IDs for a collection, with Redis caching.
    Cache key is based on collection and joined chunk IDs.
    """
    if not collection or not chunk_ids:
        return []
    # Sort chunk_ids to ensure consistent cache key
    sorted_ids = sorted(chunk_ids)
    cache_key = f"chunks_by_ids:{collection}:{'-'.join(sorted_ids)}"
    try:
        cached = get_cache("chunks_by_ids", cache_key)
        if cached:
            return cached
    except Exception as e:
        logging.warning(f"Redis unavailable for chunk content: {e}")
    chunks = []
    try:
        for chunk_id in chunk_ids:
            result = supabase_client.table(collection).select('content').eq('id', chunk_id).execute()
            if result.data:
                content = result.data[0].get('content', '')
                if content:
                    chunks.append(content)
    except Exception as e:
        logging.error(f"Error fetching chunks by IDs: {e}")
    try:
        set_cache("chunks_by_ids", cache_key, chunks)
    except Exception as e:
        logging.warning(f"Redis unavailable for setting chunk content: {e}")
    return chunks

import openai
import os

def call_llm(prompt, model, provider, max_tokens=800, temperature=0.2, **kwargs):
    """Calls the LLM with the correct provider (OpenAI or Groq) based on the provider argument."""
    try:
        # Count tokens before sending
        prompt_tokens = count_tokens(prompt, model)
        
        # Set token limits based on provider and model
        if provider == "groq":
            model_limits = {
                "llama-3.3-70b-versatile": 11000,
                "llama-3.1-8b-instant": 11000,
                "llama3-70b-8192": 7000,
                "llama3-8b-8192": 7000,
                "gemma2-9b-it": 7000
            }
            limit = model_limits.get(model, 8000)
            
            if prompt_tokens > limit:
                logging.error(f"Prompt too large for {model}: {prompt_tokens} tokens (limit: {limit})")
                return f"[LLM ERROR] Error code: 413 - Request too large: {prompt_tokens} tokens exceeds limit of {limit} for model {model}. Please reduce message size."
        
        logging.info(f"Sending prompt with {prompt_tokens} tokens to {provider} {model}")
        
        if provider == "groq":
            import groq
            client = groq.Client(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        elif provider == "openai":
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        else:
            logging.error(f"Unknown provider: {provider}")
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        error_msg = str(e)
        
        # Handle rate limits with automatic fallbacks
        if "rate_limit" in error_msg.lower() or "quota" in error_msg.lower() or "429" in error_msg:
            logging.warning(f"Rate limit hit for {provider}/{model}, attempting fallback...")
            
            # Try fallback models
            if provider == "groq":
                # Try smaller/faster Groq model first
                if model != "llama-3.1-8b-instant":
                    try:
                        import groq
                        client = groq.Client(api_key=GROQ_API_KEY)
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        return response.choices[0].message.content
                    except:
                        pass
                
                # Then try OpenAI if available
                if OPENAI_API_KEY:
                    try:
                        import openai
                        client = openai.OpenAI(api_key=OPENAI_API_KEY)
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        return response.choices[0].message.content
                    except:
                        pass
            
            elif provider == "openai":
                # Try smaller OpenAI model first
                if model != "gpt-4o-mini":
                    try:
                        import openai
                        client = openai.OpenAI(api_key=OPENAI_API_KEY)
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        return response.choices[0].message.content
                    except:
                        pass
                
                # Then try Groq if available
                if GROQ_API_KEY:
                    try:
                        import groq
                        client = groq.Client(api_key=GROQ_API_KEY)
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        return response.choices[0].message.content
                    except:
                        pass
            
            # If all fallbacks failed
            return "I'm currently experiencing high demand and rate limits. Please try again in a few minutes, or switch to a different AI provider in the settings."
        
        elif "413" in error_msg or "too large" in error_msg.lower():
            return "Your question or document content is too large. Please try a shorter, more focused question or select fewer document collections."
        else:
            logging.error(f"Error calling LLM: {e}")
            return "I encountered a technical issue. Please try again or switch to a different AI provider if the problem persists."

def get_top_k_chunks(collection: str, query: str, top_k: int = 1) -> List[str]:
    """Retrieve top-k relevant chunks from Supabase for a given query."""
    try:
        # Initialize embedding model
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        
        # Ensure query is not empty and is a string
        if not query or not isinstance(query, str):
            logging.error("Invalid query: must be a non-empty string")
            return []
            
        # Clean and validate query
        query = query.strip()
        if not query:
            logging.error("Query is empty after stripping")
            return []
            
        # Generate query embedding
        try:
            query_embedding = embed_model.get_text_embedding(query)
            if not query_embedding:
                logging.error("Failed to generate embedding for query")
                return []
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return []
        
        # Search using Supabase vector similarity
        try:
            # Try using direct table query first (fallback)
            result = supabase_client.table(collection).select('content').limit(top_k).execute()
            if result.data:
                texts = [row.get('content', '') for row in result.data if row.get('content')]
                logging.info(f"Using fallback search - found {len(texts)} results")
                return texts
            else:
                logging.warning(f"No results found for query in collection {collection}")
                return []
                
        except Exception as e:
            logging.error(f"Error searching Supabase: {e}")
            return []
        
    except Exception as e:
        logging.error(f"Error in get_top_k_chunks: {e}")
        return []

def get_top_k_chunks_with_scores(collection: str, query: str, top_k: int = 2) -> List[Tuple[str, str, float]]:
    """Get top-k chunks with scores."""
    try:
        # Initialize embedding model
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        
        # Ensure query is valid
        if not query or not isinstance(query, str):
            logging.error("Invalid query: must be a non-empty string")
            return []
            
        # Clean and validate query
        query = query.strip()
        if not query:
            logging.error("Query is empty after stripping")
            return []
            
        # Generate query embedding
        try:
            query_embedding = embed_model.get_text_embedding(query)
            if not query_embedding:
                logging.error("Failed to generate embedding for query")
                return []
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return []
        
        # Search using Supabase vector similarity
        try:
            # Try using direct table query first (fallback)
            result = supabase_client.table(collection).select('id, content').limit(top_k).execute()
            if result.data:
                results = []
                for row in result.data:
                    text = row.get('content', '')
                    if text:
                        # Use a default similarity score for fallback
                        results.append((str(row.get('id', '')), text, 0.7))
                logging.info(f"Using fallback search - found {len(results)} results")
                return results
            else:
                logging.warning(f"No results found for query in collection {collection}")
                return []
                
        except Exception as e:
            logging.error(f"Error searching Supabase: {e}")
            return []
        
    except Exception as e:
        logging.error(f"Error in get_top_k_chunks_with_scores: {e}")
        return []

def check_prompt_size_and_truncate(prompt: str, provider: str = "openai", model: str = "gpt-3.5-turbo") -> str:
    """Check if prompt exceeds token limits and truncate if necessary"""
    token_count = count_tokens(prompt, model)
    
    if provider == "groq":
        limits = {
            "llama-3.3-70b-versatile": 10000,
            "llama-3.1-8b-instant": 10000,
            "llama3-70b-8192": 6000,
            "llama3-8b-8192": 6000,
            "gemma2-9b-it": 6000
        }
        limit = limits.get(model, 8000)
        
        if token_count > limit:
            logging.warning(f"Prompt too large ({token_count} tokens), truncating to {limit}")
            # Simple truncation - take first portion that fits
            words = prompt.split()
            truncated = ""
            for word in words:
                test_prompt = truncated + " " + word if truncated else word
                if count_tokens(test_prompt, model) <= limit:
                    truncated = test_prompt
                else:
                    break
            return truncated + "\n\n[Note: Content was truncated due to length limits]"
    
    return prompt

def generate_section_with_user_selection(full_prompt, selected_chunks, top_k=2, model="gpt-4-turbo", provider="openai", temperature=0.2, regenerate=False): 
    import hashlib
    import json
    try:
        context_parts = []
        relevant_chunks_info = []
        ref_map = {"PS": "ps-index", "RP": "rp-index", "PT": "pt-index", "IB": "ib-index"}
        
        # Ensure full_prompt is valid
        if not full_prompt or not isinstance(full_prompt, str):
            logging.error("Invalid full_prompt provided")
            return "[ERROR] Invalid prompt provided", full_prompt, [], model, provider
            
        # Log the input parameters
        logging.info(f"Generating section with prompt: {full_prompt[:100]}...")
        logging.info(f"Selected chunks: {selected_chunks}")
        logging.info(f"Using model: {model}, provider: {provider}")
            
        for ref, collection in ref_map.items():
            chunk_ids = selected_chunks.get(ref, [])
            if chunk_ids:
                logging.info(f"Processing selected chunks for {ref} - intelligently selecting most relevant")
                
                # Smart chunk selection: pick most relevant chunks from pre-selected ones
                max_chunks_per_collection = 2 if provider == "groq" else 3
                search_query = f"{full_prompt} {ref} section"
                
                # Use semantic similarity to select best chunks from pre-selected ones
                relevant_chunks = select_most_relevant_chunks(
                    chunk_ids, collection, search_query, max_chunks_per_collection
                )
                
                if not relevant_chunks:
                    logging.warning(f"No relevant chunks found for {ref}")
                    continue
                
                logging.info(f"Selected {len(relevant_chunks)} most relevant chunks from {len(chunk_ids)} pre-selected chunks for {ref}")
                    
                MAX_CHUNK_CHARS = 1000
                for chunk_id, text, score in relevant_chunks:
                    truncated_text = text[:MAX_CHUNK_CHARS]
                    metadata = {}
                    try:
                        # Get metadata directly from Supabase
                        result = supabase_client.table(collection).select('metadata').eq('id', chunk_id).execute()
                        if result.data:
                            metadata = result.data[0].get('metadata', {})
                    except Exception as e:
                        logging.warning(f"Error getting metadata for chunk {chunk_id}: {e}")
                        
                    relevant_chunks_info.append({
                        "ref_type": ref,
                        "id": chunk_id,
                        "preview": truncated_text[:200],
                        "score": score,
                        "file_name": metadata.get("file_name", ""),
                        "source": collection,
                        "content": text
                    })
                
                # Only include the selected relevant chunks
                selected_texts = [text for _, text, _ in relevant_chunks]
                context_parts.append(f"{ref}:\n" + "\n".join([text[:MAX_CHUNK_CHARS] for text in selected_texts]) + "\n")
            else:
                logging.info(f"Searching for relevant chunks in {ref}")
                # Use a more specific query for vector search
                search_query = f"{full_prompt} {ref} section"
                results = get_top_k_chunks_with_scores(collection, search_query, top_k=top_k)
                
                if not results:
                    logging.warning(f"No search results found for {ref}")
                    continue
                    
                texts = [r[1] for r in results]
                for chunk_id, text, score in results:
                    metadata = {}
                    try:
                        # Get metadata directly from Supabase
                        result = supabase_client.table(collection).select('metadata').eq('id', chunk_id).execute()
                        if result.data:
                            metadata = result.data[0].get('metadata', {})
                    except Exception as e:
                        logging.warning(f"Error getting metadata for search result: {e}")
                        
                    relevant_chunks_info.append({
                        "ref_type": ref,
                        "id": chunk_id,
                        "preview": text[:200],
                        "score": score,
                        "file_name": metadata.get("file_name", ""),
                        "source": collection,
                        "content": text
                    })
                context_parts.append(f"{ref}:\n" + "\n".join(texts) + "\n")
                
        if not context_parts:
            logging.error("No reference content found for any section")
            return "[ERROR] No reference content found. Please ensure documents are properly indexed.", full_prompt, [], model, provider
                
        reference_content = "\n".join(context_parts)
        composed_prompt = f"{full_prompt}\n\nReference Content:\n{reference_content}\n"
        
        # Check and truncate prompt if necessary for token limits
        composed_prompt = check_prompt_size_and_truncate(composed_prompt, provider, model)
        
        # Always call LLM, do not use or set cache
        logging.info("Calling LLM with composed prompt (no cache)")
        try:
            output = call_llm(composed_prompt, model=model, provider=provider, temperature=temperature)
            if not output or output.startswith("[LLM ERROR]"):
                logging.error(f"LLM returned error or empty response: {output}")
                return "[ERROR] Failed to generate content. Please try again.", full_prompt, relevant_chunks_info, model, provider
        except Exception as e:
            logging.error(f"Exception in call_llm: {e}")
            return f"[ERROR] LLM error: {str(e)}", full_prompt, relevant_chunks_info, model, provider
        return output, composed_prompt, relevant_chunks_info, model, provider
    except Exception as e:
        logging.error(f"Exception in generate_section_with_user_selection: {e}")
        return f"[ERROR] {str(e)}", full_prompt, [], model, provider

# Chatbot functions moved to backend/core/chatbot.py for better code organization

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text for a given model"""
    try:
        # Map Groq models to OpenAI equivalents for token counting
        model_mapping = {
            "llama-3.3-70b-versatile": "gpt-3.5-turbo",
            "llama-3.1-8b-instant": "gpt-3.5-turbo", 
            "llama3-70b-8192": "gpt-3.5-turbo",
            "llama3-8b-8192": "gpt-3.5-turbo",
            "gemma2-9b-it": "gpt-3.5-turbo"
        }
        
        token_model = model_mapping.get(model, model)
        encoding = tiktoken.encoding_for_model(token_model)
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Token counting failed: {e}")
        # Fallback: estimate 4 characters per token
        return len(text) // 4

def truncate_chunks_to_limit(chunks: List[str], max_tokens: int = 8000) -> List[str]:
    """Truncate chunks to fit within token limit, keeping most relevant content"""
    if not chunks:
        return chunks
    
    # Calculate current token count
    total_text = " ".join(chunks)
    current_tokens = count_tokens(total_text)
    
    if current_tokens <= max_tokens:
        return chunks
    
    # If over limit, progressively reduce
    truncated_chunks = []
    running_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        
        # If adding this chunk would exceed limit
        if running_tokens + chunk_tokens > max_tokens:
            # Calculate remaining space
            remaining_tokens = max_tokens - running_tokens
            
            if remaining_tokens > 100:  # Only add if meaningful space left
                # Truncate this chunk to fit
                words = chunk.split()
                truncated_chunk = ""
                for word in words:
                    test_chunk = truncated_chunk + " " + word if truncated_chunk else word
                    if count_tokens(test_chunk) <= remaining_tokens:
                        truncated_chunk = test_chunk
                    else:
                        break
                if truncated_chunk:
                    truncated_chunks.append(truncated_chunk + "...")
            break
        else:
            truncated_chunks.append(chunk)
            running_tokens += chunk_tokens
    
    return truncated_chunks

# chat_with_documents function moved to backend/core/chatbot.py

def select_most_relevant_chunks(chunk_ids: List[str], collection: str, query: str, max_chunks: int = 3) -> List[Tuple[str, str, float]]:
    """Select the most relevant chunks from a list of pre-selected chunks using semantic similarity"""
    try:
        if not chunk_ids or not query:
            return []
        
        # Initialize embedding model
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        
        # Generate query embedding
        query_embedding = embed_model.get_text_embedding(query.strip())
        if not query_embedding:
            logging.error("Failed to generate embedding for relevance filtering")
            # Fallback: return first few chunks
            texts = get_chunks_by_ids(collection, chunk_ids[:max_chunks])
            return [(chunk_ids[i], text, 0.7) for i, text in enumerate(texts)]  # Use higher fallback score
        
        # Get chunks with their embeddings and calculate similarity
        scored_chunks = []
        
        for chunk_id in chunk_ids:
            try:
                # Get the chunk from Supabase with its embedding
                result = supabase_client.table(collection).select('id, content, embedding').eq('id', chunk_id).execute()
                
                if not result.data:
                    continue
                    
                row = result.data[0]
                chunk_text = row.get('content', '')
                chunk_embedding = row.get('embedding')
                
                if not chunk_text or not chunk_embedding:
                    continue
                
                # Calculate similarity score using vector if available
                if chunk_embedding:
                    # Calculate cosine similarity
                    import numpy as np
                    chunk_vector = np.array(chunk_embedding)
                    query_vector = np.array(query_embedding)
                    
                    # Cosine similarity
                    dot_product = np.dot(chunk_vector, query_vector)
                    norm_chunk = np.linalg.norm(chunk_vector)
                    norm_query = np.linalg.norm(query_vector)
                    
                    if norm_chunk > 0 and norm_query > 0:
                        cosine_similarity = dot_product / (norm_chunk * norm_query)
                        
                        # Convert from cosine similarity range [-1,1] to enhanced score [0,1]
                        # For text embeddings, cosine similarity is typically [0.2, 0.9]
                        # Normalize and enhance this range
                        normalized_similarity = max(0.0, (cosine_similarity + 1.0) / 2.0)  # Convert [-1,1] to [0,1]
                        
                        # Apply enhancement to make good matches more prominent
                        enhanced_score = normalized_similarity ** 0.6  # Moderate enhancement
                        similarity = min(0.95, enhanced_score)  # Cap at 95% to keep realistic
                    else:
                        similarity = 0.5
                else:
                    # Fallback: use default similarity score
                    similarity = 0.7  # Higher fallback score
                
                scored_chunks.append((chunk_id, chunk_text, similarity))
                    
            except Exception as e:
                logging.warning(f"Error scoring chunk {chunk_id}: {e}")
                # Include chunk with higher default score
                texts = get_chunks_by_ids(collection, [chunk_id])
                if texts:
                    scored_chunks.append((chunk_id, texts[0], 0.7))  # Higher fallback
        
        # Sort by similarity score (highest first) and return top chunks
        scored_chunks.sort(key=lambda x: x[2], reverse=True)
        return scored_chunks[:max_chunks]
    
    except Exception as e:
        logging.error(f"Error in select_most_relevant_chunks: {e}")
        # Fallback: return first few chunks with higher scores
        texts = get_chunks_by_ids(collection, chunk_ids[:max_chunks])
        return [(chunk_ids[i], text, 0.7) for i, text in enumerate(texts)]

# chat_with_documents_and_web function moved to backend/core/chatbot.py

def vector_search(query):
    cached = get_cache("vector_search", query)
    if cached:
        return cached
    # ... perform actual vector search ...
    result = do_vector_search(query)
    set_cache("vector_search", query, result)
    return result

def get_or_cache_embedding(text: str, model: str) -> list:
    """
    Get embedding for text and model, using Redis cache if available.
    Cache key is based on model and SHA256 hash of text.
    """
    if not text or not model:
        return []
    cache_key = f"embedding:{model}:{hashlib.sha256(text.encode()).hexdigest()}"
    try:
        cached = get_cache("embedding", cache_key)
        if cached:
            return cached
    except Exception as e:
        logging.warning(f"Redis unavailable for embedding: {e}")
    embed_model = OpenAIEmbedding(model=model, api_key=OPENAI_API_KEY)
    embedding = embed_model.get_text_embedding(text)
    try:
        set_cache("embedding", cache_key, embedding)
    except Exception as e:
        logging.warning(f"Redis unavailable for setting embedding: {e}")
    return embedding

def clear_cache_by_prefix(prefix: str):
    """
    Utility function to clear all Redis cache keys with a given prefix.
    Use with caution! This will remove all cached entries for the prefix.
    """
    try:
        if r is not None:
            pattern = f"{prefix}:*"
            for key in r.scan_iter(pattern):
                r.delete(key)
            logging.info(f"Cleared cache for prefix: {prefix}")
    except Exception as e:
        logging.warning(f"Failed to clear cache for prefix {prefix}: {e}")

