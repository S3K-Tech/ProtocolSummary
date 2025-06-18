"""
Chatbot functionality for the Clinical Trial Protocol Generator.
Handles document search, web search, and hybrid modes.
"""

import logging
import os
import re
import requests
from typing import List, Dict, Any, Tuple

# Import necessary modules from the main generator
from .generator import (
    count_tokens, truncate_chunks_to_limit, call_llm,
    get_top_k_chunks_with_scores, OPENAI_API_KEY
)
from supabase import create_client, Client
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def search_external(query: str, search_type: str = "medical", max_results: int = 5) -> Dict[str, Any]:
    """Search external sources using Serper API"""
    try:
        SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        if not SERPER_API_KEY:
            return {
                "results": [],
                "message": "Web search is not configured. Please add SERPER_API_KEY to your environment variables to enable web search functionality."
            }
        
        # Define search parameters based on type
        search_params = {
            "q": query,
            "num": max_results,
        }
        
        # Adjust search based on type
        if search_type == "medical":
            search_params["q"] += " clinical trial medical research"
        elif search_type == "pubmed":
            search_params["q"] += " site:pubmed.ncbi.nlm.nih.gov"
        elif search_type == "general":
            pass  # Use query as-is
        
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=search_params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Extract organic results
            organic_results = data.get("organic", [])
            for result in organic_results[:max_results]:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "source": "Google Search"
                })
            
            return {
                "results": results,
                "message": f"Found {len(results)} external sources."
            }
        else:
            logging.error(f"Serper API error: {response.status_code} - {response.text}")
            return {
                "results": [],
                "message": f"Web search service error (status {response.status_code}). Please try again later."
            }
            
    except requests.RequestException as e:
        logging.error(f"Network error in search_external: {e}")
        return {
            "results": [],
            "message": "Network error while searching. Please check your internet connection and try again."
        }
    except Exception as e:
        logging.error(f"Error in search_external: {e}")
        return {
            "results": [],
            "message": "An error occurred while searching external sources. Please try again."
        }

def summarize_search_results(results: List[Dict], query: str, model: str = "gpt-3.5-turbo", provider: str = "openai") -> str:
    """Summarize search results using LLM"""
    try:
        if not results:
            return "No relevant external sources found for your query."
        
        # Create context from search results
        context_parts = []
        for i, result in enumerate(results[:3], 1):  # Limit to top 3 for token management
            title = result.get('title', 'Unknown Title')
            snippet = result.get('snippet', 'No description available')
            source = result.get('link', 'Unknown Source')
            
            context_parts.append(f"Source {i}: {title}\nContent: {snippet}\nURL: {source}")
        
        context = "\n\n".join(context_parts)
        
        # Create summarization prompt
        prompt = f"""Based on the following external sources, provide a comprehensive answer to the question. Be specific and cite relevant information from the sources.

QUESTION: {query}

EXTERNAL SOURCES:
{context}

Instructions:
1. Provide a clear, informative answer based on the external sources
2. Use medical terminology appropriately
3. Synthesize information from multiple sources when relevant
4. If sources don't directly answer the question, mention what related information is available
5. Write naturally without reference numbers - sources will be listed separately"""

        summary = call_llm(
            prompt=prompt,
            model=model,
            provider=provider,
            max_tokens=600,
            temperature=0.2
        )
        
        return summary
        
    except Exception as e:
        logging.error(f"Error summarizing search results: {e}")
        return f"Found {len(results)} sources but couldn't generate summary."

def chat_with_documents(query: str, collections: List[str], provider: str = "openai", model: str = "gpt-3.5-turbo", chunks_per_collection: int = 1) -> Dict[str, Any]:
    """
    Chat with documents using vector search.
    
    Args:
        query: User's question
        collections: List of collection names to search
        provider: AI provider (openai or groq)
        model: Model name
        chunks_per_collection: Number of chunks to retrieve per collection
        
    Returns:
        Dictionary with answer and sources
    """
    try:
        logging.info(f"Chat with documents: {query[:100]}...")
        
        all_chunks = []
        chunk_sources = []
        
        for collection in collections:
            logging.info(f"Searching collection: {collection}")
            
            # Get relevant chunks with scores
            results = get_top_k_chunks_with_scores(collection, query, top_k=chunks_per_collection)
            
            if not results:
                logging.warning(f"No results found in collection {collection}")
                continue
                
            for chunk_id, chunk_text, score in results:
                logging.info(f"Found chunk with score {score:.3f}")
                all_chunks.append(chunk_text)
                
                # Extract source information from metadata
                try:
                    # Get row with metadata from Supabase
                    result = supabase_client.table(collection).select('id, metadata').eq('id', chunk_id).execute()
                    
                    if result.data:
                        row = result.data[0]
                        metadata = row.get("metadata", {})
                        
                        # Collection-based mapping for clean names
                        collection_map = {
                            'ps-index': 'Protocol Summary (PS)',
                            'pt-index': 'Protocol Template (PT)', 
                            'rp-index': 'Reference Protocol (RP)',
                            'ib-index': 'Investigator\'s Brochure (IB)'
                        }
                        
                        # Use collection mapping
                        if collection in collection_map:
                            source_info = collection_map[collection]
                        else:
                            source_info = f"{collection.replace('-index', '').upper()} Document"
                        
                        chunk_sources.append(source_info)
                    else:
                        doc_type = collection.replace('-index', '').upper()
                        chunk_sources.append(f"{doc_type} Document")
                        
                except Exception as e:
                    logging.error(f"Error getting source info: {e}")
                    doc_type = collection.replace('-index', '').upper()
                    chunk_sources.append(f"{doc_type} Document")
        
        if not all_chunks:
            return {"answer": "I couldn't find relevant information in the selected documents.", "sources": []}
        
        # Truncate chunks if needed
        all_chunks = truncate_chunks_to_limit(all_chunks, max_tokens=6000)
        
        # Create context from chunks
        context = "\n\n".join(all_chunks)
        
        # Create prompt
        prompt = f"""You are a helpful medical writing assistant. Answer the following question based on the provided document context. Be concise but thorough, and cite your sources when possible.

Question: {query}

Document Context:
{context}

Answer:"""
        
        # Get response from LLM
        try:
            answer = call_llm(prompt, model=model, provider=provider, max_tokens=800)
            
            # Add source information
            unique_sources = list(set(chunk_sources))
            sources_text = f"\n\nSources: {', '.join(unique_sources)}" if unique_sources else ""
            
            return {
                "answer": answer + sources_text,
                "sources": unique_sources,
                "chunks_used": len(all_chunks)
            }
            
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return {"answer": "I encountered an error while processing your question. Please try again.", "sources": []}
            
    except Exception as e:
        logging.error(f"Error in chat_with_documents: {e}")
        return {"answer": "I encountered an error while searching the documents. Please try again.", "sources": []}

def chat_with_web_search(query: str, provider: str = "openai", model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """
    Chat with web search results.
    
    Args:
        query: User's question
        provider: AI provider (openai or groq)
        model: Model name
        
    Returns:
        Dictionary with answer and sources
    """
    try:
        logging.info(f"Web search for: {query}")
        
        # Get web search results
        search_results = perform_web_search(query)
        
        if not search_results:
            return {"answer": "I couldn't find relevant information from web search.", "sources": []}
        
        # Create context from search results
        context_parts = []
        sources = []
        
        for result in search_results[:5]:  # Limit to top 5 results
            context_parts.append(f"Title: {result['title']}\nSnippet: {result['snippet']}\nURL: {result['url']}")
            sources.append(result['url'])
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful medical writing assistant. Answer the following question based on the provided web search results. Be concise but thorough, and cite your sources when possible.

Question: {query}

Web Search Results:
{context}

Answer:"""
        
        # Get response from LLM
        try:
            answer = call_llm(prompt, model=model, provider=provider, max_tokens=800)
            
            # Add source information
            sources_text = f"\n\nSources: {', '.join(sources)}" if sources else ""
            
            return {
                "answer": answer + sources_text,
                "sources": sources,
                "search_results_count": len(search_results)
            }
            
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return {"answer": "I encountered an error while processing your question. Please try again.", "sources": []}
            
    except Exception as e:
        logging.error(f"Error in chat_with_web_search: {e}")
        return {"answer": "I encountered an error while searching the web. Please try again.", "sources": []}

def chat_with_documents_and_web(query: str, collections: List[str], provider: str = "openai", model: str = "gpt-3.5-turbo", chunks_per_collection: int = 1) -> Dict[str, Any]:
    """
    Chat with both documents and web search.
    
    Args:
        query: User's question
        collections: List of collection names to search
        provider: AI provider (openai or groq)
        model: Model name
        chunks_per_collection: Number of chunks to retrieve per collection
        
    Returns:
        Dictionary with answer and sources
    """
    try:
        logging.info(f"Hybrid search for: {query}")
        
        # Get document results
        doc_result = chat_with_documents(query, collections, provider, model, chunks_per_collection)
        
        # Get web search results
        web_result = chat_with_web_search(query, provider, model)
        
        # Combine results
        combined_answer = f"""Based on your documents and web search:

**From Your Documents:**
{doc_result['answer']}

**From Web Search:**
{web_result['answer']}"""
        
        # Combine sources
        all_sources = doc_result.get('sources', []) + web_result.get('sources', [])
        unique_sources = list(set(all_sources))
        
        return {
            "answer": combined_answer,
            "sources": unique_sources,
            "document_sources": doc_result.get('sources', []),
            "web_sources": web_result.get('sources', []),
            "chunks_used": doc_result.get('chunks_used', 0),
            "search_results_count": web_result.get('search_results_count', 0)
        }
        
    except Exception as e:
        logging.error(f"Error in chat_with_documents_and_web: {e}")
        return {"answer": "I encountered an error while processing your request. Please try again.", "sources": []}

def perform_web_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Perform web search using Serper API.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, snippet, and URL
    """
    try:
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            logging.warning("SERPER_API_KEY not found, skipping web search")
            return []
        
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': max_results
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        if 'organic' in data:
            for result in data['organic'][:max_results]:
                results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    'url': result.get('link', '')
                })
        
        logging.info(f"Found {len(results)} web search results")
        return results
        
    except Exception as e:
        logging.error(f"Error performing web search: {e}")
        return [] 