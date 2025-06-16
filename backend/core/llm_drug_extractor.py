from typing import List, Dict, Any, Optional
import json
import logging
import tiktoken
from dataclasses import dataclass, asdict
from backend.core.config import OPENAI_API_KEY, GROQ_API_KEY

@dataclass
class LLMDrugInfo:
    name: str
    dosage: str = ""
    frequency: str = ""
    route: str = ""
    form: str = ""
    indication: str = ""
    source_context: str = ""

class LLMDrugExtractor:
    """
    Pure LLM-based drug information extractor with cost tracking.
    Uses AI to identify and extract drug information from clinical protocol text.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", provider: str = "openai"):
        self.model = model
        self.provider = provider
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        # Initialize the appropriate LLM client
        if provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        elif provider == "groq":
            import groq
            self.client = groq.Client(api_key=GROQ_API_KEY)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text for cost calculation"""
        try:
            if "gpt" in self.model.lower():
                encoding = tiktoken.encoding_for_model(self.model)
            else:
                # Fallback for non-OpenAI models
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            # Rough estimate if tiktoken fails
            return len(text.split()) * 1.3
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model and provider"""
        if self.provider == "openai":
            if "gpt-4o-mini" in self.model:
                # GPT-4o-mini pricing: $0.00015/1K input, $0.0006/1K output
                input_cost = (input_tokens / 1000) * 0.00015
                output_cost = (output_tokens / 1000) * 0.0006
            elif "gpt-3.5" in self.model:
                # GPT-3.5-turbo pricing: $0.0005/1K input, $0.0015/1K output
                input_cost = (input_tokens / 1000) * 0.0005
                output_cost = (output_tokens / 1000) * 0.0015
            else:
                # Default GPT-4 pricing
                input_cost = (input_tokens / 1000) * 0.01
                output_cost = (output_tokens / 1000) * 0.03
            return input_cost + output_cost
        elif self.provider == "groq":
            # Groq is typically free or very low cost
            return 0.0
        else:
            return 0.0
    
    def extract_drugs_from_text(self, text: str, context_type: str = "generated") -> List[LLMDrugInfo]:
        """
        Extract drug information from text using pure LLM approach.
        
        Args:
            text: The text to extract drugs from
            context_type: "generated" or "reference" for context
            
        Returns:
            List of LLMDrugInfo objects with cost tracking
        """
        
        logging.info(f"🤖 Using pure LLM extraction for {context_type} text...")
        
        extraction_prompt = f"""
You are a clinical research expert. Extract ALL drug information from the following {context_type} clinical protocol text.

For each drug mentioned, provide:
1. Drug name (generic name preferred)
2. Dosage (with units like mg, g, etc.)
3. Frequency (daily, twice daily, etc.)
4. Route of administration (oral, IV, etc.)
5. Form (tablet, injection, etc.)
6. Indication/purpose if mentioned
7. The exact source context where you found this information

IMPORTANT RULES:
- Extract EVERY drug mentioned, including comparators, controls, and background medications
- If multiple dosages are mentioned for the same drug (e.g., "semaglutide 3mg, 7mg, and 14mg"), create SEPARATE entries for each dosage
- If dosage is not specified, leave it empty but still include the drug
- Be precise with dosage units and values
- Include the exact text snippet where you found each drug
- Do NOT extract placebo, control, or comparator as drugs - these are not medications

Text to analyze:
{text}

Respond with a JSON array of drug objects. Each object should have these fields:
- name: string
- dosage: string (empty if not specified)
- frequency: string (empty if not specified)  
- route: string (empty if not specified)
- form: string (empty if not specified)
- indication: string (empty if not specified)
- source_context: string (the exact sentence/phrase where this drug was mentioned)

Example format:
[
  {{
    "name": "Semaglutide",
    "dosage": "3 mg",
    "frequency": "once daily",
    "route": "oral",
    "form": "tablet",
    "indication": "type 2 diabetes",
    "source_context": "Participants will receive oral semaglutide 3 mg once daily"
  }}
]

CRITICAL: If multiple dosages for same drug, create SEPARATE entries for each dosage.
"""

        try:
            # Count input tokens
            input_tokens = self.count_tokens(extraction_prompt)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical research expert specializing in drug information extraction."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Count output tokens and calculate cost
            output_tokens = self.count_tokens(response_text)
            call_cost = self.calculate_cost(input_tokens, output_tokens)
            
            # Update totals
            self.total_tokens_used += input_tokens + output_tokens
            self.total_cost += call_cost
            
            logging.info(f"💰 LLM Call: {input_tokens + output_tokens} tokens, ${call_cost:.4f}")
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            try:
                # Try to find JSON array in the response
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text
                
                drugs_data = json.loads(json_text)
                drugs = []
                
                for drug_data in drugs_data:
                    drug = LLMDrugInfo(
                        name=drug_data.get("name", ""),
                        dosage=drug_data.get("dosage", ""),
                        frequency=drug_data.get("frequency", ""),
                        route=drug_data.get("route", ""),
                        form=drug_data.get("form", ""),
                        indication=drug_data.get("indication", ""),
                        source_context=drug_data.get("source_context", "")
                    )
                    drugs.append(drug)
                
                logging.info(f"✅ LLM extraction complete: Found {len(drugs)} drugs")
                return drugs
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response as JSON: {e}")
                logging.error(f"Response was: {response_text}")
                return []
                
        except Exception as e:
            logging.error(f"Error in LLM drug extraction: {e}")
            return []
    
    def compare_drug_lists(self, generated_drugs: List[LLMDrugInfo], reference_drugs: List[LLMDrugInfo]) -> Dict[str, Any]:
        """
        Compare two lists of drugs using intelligent matching.
        
        Args:
            generated_drugs: Drugs extracted from generated protocol
            reference_drugs: Drugs extracted from reference documents
            
        Returns:
            Comparison results dictionary
        """
        
        comparison_prompt = f"""
You are a clinical research expert. Compare these two lists of drugs and provide a detailed analysis.

GENERATED PROTOCOL DRUGS:
{json.dumps([asdict(drug) for drug in generated_drugs], indent=2)}

REFERENCE DOCUMENT DRUGS:
{json.dumps([asdict(drug) for drug in reference_drugs], indent=2)}

STEP-BY-STEP MATCHING PROCESS:
1. For each drug in the GENERATED list, find if there's a corresponding drug in the REFERENCE list
2. Match by drug NAME first, then compare DOSAGE
3. If both name and dosage match exactly, it's a PERFECT_MATCH
4. If name matches but dosage differs, it's a PARTIAL_MATCH
5. Only put drugs in MISSING if the drug name doesn't exist AT ALL in the generated list

MATCHING RULES:
- Exact name and dosage match = PERFECT_MATCH
- Same name but different dosage = PARTIAL_MATCH
- Ignore differences in frequency, route, form, or source_context for matching
- Focus ONLY on drug name and dosage for comparison
- "3 mg" and "3mg" are considered identical (ignore spacing)

For each comparison, provide:
- The matching category (PERFECT_MATCH, PARTIAL_MATCH, MISSING, ADDITIONAL)
- Clear explanation
- High confidence for exact matches

Respond with JSON in this format:
{{
  "matches": [
    {{
      "drug_name": "string",
      "generated_drug": {{"name": "...", "dosage": "...", ...}},
      "reference_drug": {{"name": "...", "dosage": "...", ...}},
      "match_type": "PERFECT_MATCH|PARTIAL_MATCH",
      "explanation": "Both drugs have identical name and dosage",
      "confidence": "high"
    }}
  ],
  "missing": [
    {{
      "drug": {{"name": "...", "dosage": "...", ...}},
      "explanation": "This drug name does not appear in the generated list"
    }}
  ],
  "additional": [
    {{
      "drug": {{"name": "...", "dosage": "...", ...}},
      "explanation": "This drug name does not appear in the reference list"
    }}
  ],
  "summary": {{
    "total_generated": {len(generated_drugs)},
    "total_reference": {len(reference_drugs)},
    "perfect_matches": 0,
    "partial_matches": 0,
    "missing_count": 0,
    "additional_count": 0,
    "accuracy_percentage": 0.0
  }}
}}

CRITICAL: Make sure to count perfect_matches and partial_matches correctly in the summary section.
"""

        try:
            # Count input tokens
            input_tokens = self.count_tokens(comparison_prompt)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical research expert specializing in drug comparison analysis."},
                    {"role": "user", "content": comparison_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Count output tokens and calculate cost
            output_tokens = self.count_tokens(response_text)
            call_cost = self.calculate_cost(input_tokens, output_tokens)
            
            # Update totals
            self.total_tokens_used += input_tokens + output_tokens
            self.total_cost += call_cost
            
            logging.info(f"💰 LLM Comparison: {input_tokens + output_tokens} tokens, ${call_cost:.4f}")
            
            try:
                # Extract JSON from response (handle cases where LLM adds extra text)
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text
                
                comparison_result = json.loads(json_text)
                return comparison_result
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM comparison response: {e}")
                logging.error(f"Response was: {response_text}")
                return self._create_fallback_comparison(generated_drugs, reference_drugs)
                
        except Exception as e:
            logging.error(f"Error in LLM drug comparison: {e}")
            return self._create_fallback_comparison(generated_drugs, reference_drugs)
    
    def _create_fallback_comparison(self, generated_drugs: List[LLMDrugInfo], reference_drugs: List[LLMDrugInfo]) -> Dict[str, Any]:
        """Create a basic comparison if LLM fails"""
        return {
            "matches": [],
            "missing": [asdict(drug) for drug in reference_drugs],
            "additional": [asdict(drug) for drug in generated_drugs],
            "summary": {
                "total_generated": len(generated_drugs),
                "total_reference": len(reference_drugs),
                "perfect_matches": 0,
                "partial_matches": 0,
                "missing_count": len(reference_drugs),
                "additional_count": len(generated_drugs),
                "accuracy_percentage": 0.0
            }
        }
    
    def analyze_drug_information(self, generated_text: str, reference_text: str) -> Dict[str, Any]:
        """
        Complete drug analysis workflow.
        
        Args:
            generated_text: The generated protocol text
            reference_text: The reference document text
            
        Returns:
            Complete analysis results
        """
        
        # Extract drugs from both texts
        generated_drugs = self.extract_drugs_from_text(generated_text, "generated")
        reference_drugs = self.extract_drugs_from_text(reference_text, "reference")
        
        # Compare the drugs
        comparison = self.compare_drug_lists(generated_drugs, reference_drugs)
        
        return {
            "generated_drugs": [asdict(drug) for drug in generated_drugs],
            "reference_drugs": [asdict(drug) for drug in reference_drugs],
            "comparison": comparison,
            "extraction_method": "pure_llm",
            "model_used": self.model,
            "cost_summary": self.get_cost_summary()
        }
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of token usage and costs for this session.
        """
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost, 4),
            "model": self.model,
            "provider": self.provider,
            "extraction_method": "Pure LLM (High Accuracy)"
        } 