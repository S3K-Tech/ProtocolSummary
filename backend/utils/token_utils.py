import tiktoken
from typing import Dict, Tuple, Optional

# Model pricing configurations (as of January 2025)
MODEL_PRICING = {
    # OpenAI Models (per 1K tokens)
    "openai": {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        
    },
    
    # Groq Models (per 1M tokens, converted to per 1K)
    "groq": {
        # Production Models (Free tier with rate limits, paid tier pricing shown)
        "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},  # $0.59/$0.79 per 1M
        "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},     # $0.05/$0.08 per 1M
        "llama3-70b-8192": {"input": 0.00059, "output": 0.00079},          # $0.59/$0.79 per 1M
        "llama3-8b-8192": {"input": 0.00005, "output": 0.00008},           # $0.05/$0.08 per 1M
        "gemma2-9b-it": {"input": 0.0002, "output": 0.0002},               # $0.20/$0.20 per 1M
        "meta-llama/llama-guard-4-12b": {"input": 0.0002, "output": 0.0002}, # $0.20/$0.20 per 1M
        
        # Preview Models (Free tier with rate limits, estimated pricing for paid)
        "deepseek-r1-distill-llama-70b": {"input": 0.00075, "output": 0.00099}, # $0.75/$0.99 per 1M
        "qwen-qwq-32b": {"input": 0.00029, "output": 0.00039},             # $0.29/$0.39 per 1M
        "mistral-saba-24b": {"input": 0.00079, "output": 0.00079},         # $0.79/$0.79 per 1M
        "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.00011, "output": 0.00034}, # $0.11/$0.34 per 1M
        "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.0002, "output": 0.0006}, # $0.20/$0.60 per 1M
        
    }
}

# Groq free tier models (actually free with rate limits)
GROQ_FREE_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant", 
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "meta-llama/llama-guard-4-12b",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
    "mistral-saba-24b",
}

def get_provider_from_model(model: str) -> str:
    """Determine provider from model name - simplified"""
    model_lower = model.lower()
    
    # Groq models (simplified list)
    if any(groq_model in model_lower for groq_model in [
        "llama", "gemma", "deepseek", "qwen", "mistral"
    ]):
        return "groq"
    
    # OpenAI models  
    if any(openai_model in model_lower for openai_model in [
        "gpt-4", "gpt-3.5", "gpt-4o"
    ]):
        return "openai"
    
    # Default to openai for unknown models
    return "openai"

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string for a specific model."""
    try:
        # Handle Groq models - most use Llama tokenizer, map to closest OpenAI equivalent
        if get_provider_from_model(model) == "groq":
            # Map Groq models to OpenAI equivalents for tokenization
            if "8b" in model.lower() or "9b" in model.lower():
                encoding_model = "gpt-3.5-turbo"  # Smaller models
            elif "70b" in model.lower() or "32b" in model.lower():
                encoding_model = "gpt-4"  # Larger models
            else:
                encoding_model = "gpt-4"  # Default for other Groq models
        else:
            encoding_model = model
            
        encoding = tiktoken.encoding_for_model(encoding_model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate based on word count
        return max(1, int(len(text.split()) * 1.3))  # Approximate tokens per word

def get_model_pricing(model: str, provider: str = None) -> Dict[str, float]:
    """Get pricing for a specific model and provider"""
    if provider is None:
        provider = get_provider_from_model(model)
    
    provider_pricing = MODEL_PRICING.get(provider.lower(), MODEL_PRICING["openai"])
    
    # Try exact model match first
    if model in provider_pricing:
        return provider_pricing[model]
    
    # Try partial matches for Groq models with different naming
    if provider.lower() == "groq":
        for pricing_model, pricing in provider_pricing.items():
            if pricing_model.replace("-", "").replace("_", "") in model.replace("-", "").replace("_", ""):
                return pricing
    
    # Default fallback pricing
    if provider.lower() == "groq":
        return {"input": 0.0002, "output": 0.0002}  # Conservative Groq estimate
    else:
        return {"input": 0.03, "output": 0.06}  # GPT-4 pricing as fallback

def calculate_cost(input_tokens: int, output_tokens: int, 
                  model: str = "gpt-4", provider: str = None,
                  input_price_per_k: Optional[float] = None, 
                  output_price_per_k: Optional[float] = None) -> Tuple[float, Dict]:
    """
    Calculate the cost based on token counts, model, and provider.
    Returns (total_cost, cost_breakdown)
    """
    
    # Ensure token counts are integers
    input_tokens = int(input_tokens)
    output_tokens = int(output_tokens)
    
    # If manual prices provided, use them (for backward compatibility)
    if input_price_per_k is not None and output_price_per_k is not None:
        input_cost = (input_tokens / 1000) * input_price_per_k
        output_cost = (output_tokens / 1000) * output_price_per_k
        total_cost = input_cost + output_cost
        
        return round(total_cost, 6), {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "pricing_source": "manual",
            "provider": provider or get_provider_from_model(model),
            "model": model,
            "is_free_tier": False
        }
    
    # Determine provider
    if provider is None:
        provider = get_provider_from_model(model)
    
    # Check if this is a free Groq model
    is_free_tier = provider.lower() == "groq" and model in GROQ_FREE_MODELS
    
    if is_free_tier:
        return 0.0, {
            "input_cost": 0.0,
            "output_cost": 0.0, 
            "total_cost": 0.0,
            "pricing_source": "groq_free_tier",
            "provider": provider,
            "model": model,
            "is_free_tier": True,
            "note": "Free tier with rate limits"
        }
    
    # Get model-specific pricing
    pricing = get_model_pricing(model, provider)
    
    # Calculate costs
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return round(total_cost, 6), {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "pricing_source": "model_specific",
        "provider": provider,
        "model": model,
        "is_free_tier": False,
        "input_price_per_k": pricing["input"],
        "output_price_per_k": pricing["output"]
    }

def get_available_models() -> Dict[str, list]:
    """Get list of available models by provider"""
    return {
        "openai": list(MODEL_PRICING["openai"].keys()),
        "groq": list(MODEL_PRICING["groq"].keys())
    }

def is_model_deprecated(model: str) -> bool:
    """Check if a model is deprecated - simplified version"""
    # Only check for models that might still be in use but deprecated
    deprecated_models = {"gpt-3.5-turbo", "mixtral-8x7b-32768"}
    return model in deprecated_models

def get_model_replacement(model: str) -> Optional[str]:
    """Get recommended replacement for deprecated models - simplified version"""
    replacements = {
        "gpt-3.5-turbo": "gpt-4o-mini",
        "mixtral-8x7b-32768": "mistral-saba-24b"
    }
    return replacements.get(model)
