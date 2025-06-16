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
from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL", "https://6a1e279b-b34d-4b90-b58f-8ec1179e0128.us-west-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    port=6333,
    api_key=QDRANT_API_KEY,
    https=True,
)

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
    """Query documents with optimized chunk management for chatbot use"""
    try:
        # For each collection, get relevant chunks with metadata
        all_chunks = []
        chunk_sources = []
        
        # Use user-specified chunks_per_collection (default = 1 for optimal performance)
        # This ensures fast responses and minimal token usage
        
        for collection in collections:
            # Use enhanced function to get chunks with metadata - already optimized for relevance
            chunks_with_scores = get_top_k_chunks_with_scores(collection, query, top_k=chunks_per_collection)
            
            for chunk_id, chunk_text, score in chunks_with_scores:
                all_chunks.append(chunk_text)
                
                # Extract source information from metadata
                try:
                    # Get point with payload from Qdrant
                    try:
                        point_id = int(chunk_id)
                    except ValueError:
                        point_id = chunk_id
                    
                    point = qdrant_client.retrieve(
                        collection_name=collection,
                        ids=[point_id],
                        with_payload=True
                    )[0]
                    
                    # Try multiple ways to extract metadata
                    metadata = point.payload.get("metadata", {})
                    
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
                    
                except Exception as e:
                    logging.error(f"Error getting source info: {e}")
                    doc_type = collection.replace('-index', '').upper()
                    chunk_sources.append(f"{doc_type} Document")
        
        if not all_chunks:
            return {"answer": "I couldn't find relevant information in the selected documents.", "sources": []}
        
        # Deduplicate sources while preserving order
        unique_sources = []
        for source in chunk_sources:
            if source not in unique_sources:
                unique_sources.append(source)
        
        # Apply much stricter token limits for chatbot to prevent token overload
        max_content_tokens = 3000 if provider == "groq" else 4000
        truncated_chunks = truncate_chunks_to_limit(all_chunks, max_content_tokens)
        
        # Log chunk statistics for debugging
        logging.info(f"Chatbot: Using {len(truncated_chunks)} chunks from {len(collections)} collections, ~{count_tokens(' '.join(truncated_chunks))} tokens")
        
        # Create a cleaner prompt for both Groq and OpenAI models
        prompt = f"""Answer this clinical trial question using the provided document context. Be professional and specific. Do NOT include reference numbers like [1], [2] in your response - the sources will be listed separately.

QUESTION: {query}

CONTEXT:
{chr(10).join([chunk for chunk in truncated_chunks])}

AVAILABLE SOURCES: {', '.join(unique_sources)}

Instructions:
1. Provide a helpful, informative answer based on the context
2. Use proper medical terminology and be precise
3. Write naturally without reference numbers in the text
4. If specific information isn't available, mention what related information is provided
5. Be as helpful as possible while staying accurate to the source material"""
        
        # Check final token count and apply emergency truncation if needed
        final_tokens = count_tokens(prompt, model)
        if final_tokens > 10000:
            logging.warning(f"Large prompt detected: {final_tokens} tokens for model {model}")
            if provider == "groq" and final_tokens > 11000:
                # Emergency truncation for Groq
                max_emergency_tokens = 2000  # Very aggressive for chatbot
                truncated_chunks = truncate_chunks_to_limit(all_chunks, max_emergency_tokens)
                prompt = f"""Answer the clinical trial question using the provided context. Be specific and professional.

QUESTION: {query}

CONTEXT: {' '.join([chunk for chunk in truncated_chunks])}

AVAILABLE SOURCES: {', '.join(unique_sources)}"""
        
        # Use existing LLM call function
        response = call_llm(
            prompt=prompt,
            model=model,
            provider=provider,
            max_tokens=800,
            temperature=0.2
        )
        
        return {
            "answer": response,
            "sources": unique_sources,
            "num_chunks": len(truncated_chunks)
        }
    
    except Exception as e:
        logging.error(f"Error in chat_with_documents: {e}")
        return {
            "answer": "I encountered an error while searching documents. Please try again.",
            "sources": []
        }

def chat_with_documents_and_web(query: str, collections: List[str], search_type: str = "documents", 
                               provider: str = "openai", model: str = "gpt-3.5-turbo", 
                               web_search_type: str = "medical", chunks_per_collection: int = 1) -> Dict[str, Any]:
    """
    Enhanced chat function supporting documents, web, and hybrid search modes
    """
    try:
        doc_result = None
        web_result = None
        
        # Document search
        if search_type in ["documents", "hybrid"] and collections:
            doc_result = chat_with_documents(query, collections, provider, model, chunks_per_collection)
        
        # Web search
        if search_type in ["web", "hybrid"]:
            search_results = search_external(
                query=query, 
                search_type=web_search_type,
                max_results=3  # Reduced for token management
            )
            
            if search_results["results"]:
                web_summary = summarize_search_results(
                    results=search_results["results"],
                    query=query,
                    model=model,
                    provider=provider
                )
                web_result = {
                    "summary": web_summary,
                    "sources": search_results["results"]
                }
        
        # Return appropriate result based on search type
        if search_type == "hybrid" and doc_result and web_result:
            # Combine results
            combined_answer = f"""**From Your Documents:**
{doc_result['answer']}

**From External Sources:**
{web_result['summary']}"""
            
            return {
                "answer": combined_answer,
                "sources": doc_result.get("sources", []),
                "web_sources": web_result["sources"],
                "search_type": "hybrid"
            }
        elif search_type == "web":
            if web_result:
                return {
                    "answer": web_result["summary"],
                    "sources": [],
                    "web_sources": web_result["sources"],
                    "search_type": "web"
                }
            else:
                # Handle case where web search failed
                error_message = search_results.get("message", "No relevant web sources found for your query.")
                return {
                    "answer": error_message,
                    "sources": [],
                    "web_sources": [],
                    "search_type": "web"
                }
        else:  # documents mode
            return {
                "answer": doc_result["answer"] if doc_result else "No results found.",
                "sources": doc_result.get("sources", []) if doc_result else [],
                "web_sources": [],
                "search_type": "documents"
            }
    
    except Exception as e:
        logging.error(f"Error in chat_with_documents_and_web: {e}")
        return {
            "answer": "Error occurred. Please try again.",
            "sources": [],
            "web_sources": [],
            "search_type": search_type
        } 