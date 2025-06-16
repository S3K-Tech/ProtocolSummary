import re
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher

class ICHSectionMatcher:
    """
    Flexible section matching utility for ICH M11 protocols.
    Matches sections by name rather than number to accommodate different numbering schemes.
    """
    
    def __init__(self, rules_file_path: str = None):
        if rules_file_path is None:
            rules_file_path = Path(__file__).parent.parent / "medical_rules" / "ich_m11_rules.yaml"
        
        with open(rules_file_path, 'r') as f:
            self.rules = yaml.safe_load(f)
        
        self.sections = self.rules.get('ich_m11_sections', {})
        self.matching_config = self.rules.get('section_matching', {})
        self.synonym_mapping = self.matching_config.get('name_normalization', {}).get('synonym_mapping', {})
        
        # Build reverse lookup for fast matching
        self._build_section_lookup()
    
    def _build_section_lookup(self):
        """Build lookup tables for efficient section matching."""
        self.name_to_key = {}
        self.number_to_key = {}
        
        for section_key, section_data in self.sections.items():
            # Primary section name
            section_name = section_data.get('section_name', '')
            if section_name:
                normalized_name = self._normalize_section_name(section_name)
                self.name_to_key[normalized_name] = section_key
            
            # Alternate names
            for alt_name in section_data.get('alternate_names', []):
                normalized_alt = self._normalize_section_name(alt_name)
                self.name_to_key[normalized_alt] = section_key
            
            # Section numbers
            for section_num in section_data.get('section_numbers', []):
                self.number_to_key[str(section_num)] = section_key
    
    def _normalize_section_name(self, name: str) -> str:
        """Normalize section name for consistent matching."""
        if not name:
            return ""
        
        name = name.lower().strip()
        
        # Remove articles if configured
        if self.matching_config.get('name_normalization', {}).get('remove_articles'):
            name = re.sub(r'\b(the|a|an)\b\s*', '', name)
        
        # Remove punctuation if configured
        if self.matching_config.get('name_normalization', {}).get('remove_punctuation'):
            name = re.sub(r'[^\w\s]', '', name)
        
        # Apply synonym mapping
        for synonym, replacements in self.synonym_mapping.items():
            for replacement in replacements:
                name = name.replace(replacement, synonym)
        
        # Clean up extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def match_section(self, section_input: str) -> Optional[Tuple[str, Dict, float]]:
        """
        Match a section input to ICH M11 section.
        
        Args:
            section_input: Section title, number, or header
            
        Returns:
            Tuple of (section_key, section_data, confidence_score) or None
        """
        if not section_input:
            return None
        
        # Try exact matches first
        exact_match = self._try_exact_match(section_input)
        if exact_match:
            return exact_match
        
        # Try pattern-based extraction
        pattern_match = self._try_pattern_match(section_input)
        if pattern_match:
            return pattern_match
        
        # Try fuzzy matching as last resort
        fuzzy_match = self._try_fuzzy_match(section_input)
        if fuzzy_match:
            return fuzzy_match
        
        return None
    
    def _try_exact_match(self, section_input: str) -> Optional[Tuple[str, Dict, float]]:
        """Try exact matching against section names and numbers."""
        normalized_input = self._normalize_section_name(section_input)
        
        # Check normalized names
        if normalized_input in self.name_to_key:
            section_key = self.name_to_key[normalized_input]
            return section_key, self.sections[section_key], 1.0
        
        # Check section numbers
        if section_input.strip() in self.number_to_key:
            section_key = self.number_to_key[section_input.strip()]
            return section_key, self.sections[section_key], 1.0
        
        return None
    
    def _try_pattern_match(self, section_input: str) -> Optional[Tuple[str, Dict, float]]:
        """Try pattern-based extraction of section information."""
        patterns = self.matching_config.get('header_patterns', [])
        
        for pattern in patterns:
            match = re.match(pattern, section_input.strip())
            if match:
                groups = match.groups()
                
                # Extract section number and name based on pattern
                if len(groups) >= 2:
                    # Pattern like "4.2 Rationale for Trial Design"
                    section_num = groups[0]
                    section_name = groups[1]
                    
                    # Try matching by number first
                    if section_num in self.number_to_key:
                        section_key = self.number_to_key[section_num]
                        return section_key, self.sections[section_key], 0.9
                    
                    # Try matching by name
                    normalized_name = self._normalize_section_name(section_name)
                    if normalized_name in self.name_to_key:
                        section_key = self.name_to_key[normalized_name]
                        return section_key, self.sections[section_key], 0.8
        
        return None
    
    def _try_fuzzy_match(self, section_input: str, threshold: float = 0.6) -> Optional[Tuple[str, Dict, float]]:
        """Try fuzzy matching against section names."""
        normalized_input = self._normalize_section_name(section_input)
        
        best_match = None
        best_score = 0
        
        for normalized_name, section_key in self.name_to_key.items():
            similarity = SequenceMatcher(None, normalized_input, normalized_name).ratio()
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = (section_key, self.sections[section_key], similarity)
        
        return best_match
    
    def get_section_requirements(self, section_key: str) -> Dict:
        """Get validation requirements for a specific section."""
        return self.sections.get(section_key, {})
    
    def is_required_section(self, section_key: str) -> bool:
        """Check if a section is required by ICH M11."""
        section_data = self.sections.get(section_key, {})
        return section_data.get('ich_requirement') == 'required'
    
    def get_section_priority(self, section_key: str) -> str:
        """Get priority level of a section."""
        section_data = self.sections.get(section_key, {})
        return section_data.get('priority', 'standard')
    
    def list_all_sections(self) -> List[Dict]:
        """List all available ICH M11 sections."""
        sections_list = []
        for section_key, section_data in self.sections.items():
            sections_list.append({
                'key': section_key,
                'name': section_data.get('section_name', ''),
                'category': section_data.get('category', ''),
                'priority': section_data.get('priority', 'standard'),
                'required': section_data.get('ich_requirement') == 'required'
            })
        return sections_list
    
    def get_sections_by_category(self, category: str) -> List[str]:
        """Get all sections in a specific category."""
        return [
            key for key, data in self.sections.items()
            if data.get('category') == category
        ]
    
    def validate_section_input(self, section_input: str) -> Dict:
        """
        Validate and provide feedback on section input.
        
        Returns:
            Dict with validation results and suggestions
        """
        match_result = self.match_section(section_input)
        
        if not match_result:
            # Provide suggestions for unmatched input
            suggestions = self._get_similar_sections(section_input)
            return {
                'matched': False,
                'input': section_input,
                'suggestions': suggestions,
                'message': f"No matching ICH M11 section found for '{section_input}'"
            }
        
        section_key, section_data, confidence = match_result
        
        return {
            'matched': True,
            'input': section_input,
            'section_key': section_key,
            'section_name': section_data.get('section_name'),
            'confidence': confidence,
            'category': section_data.get('category'),
            'priority': section_data.get('priority'),
            'required': section_data.get('ich_requirement') == 'required',
            'message': f"Matched to '{section_data.get('section_name')}' with {confidence:.1%} confidence"
        }
    
    def _get_similar_sections(self, section_input: str, limit: int = 3) -> List[str]:
        """Get similar section names for suggestions."""
        normalized_input = self._normalize_section_name(section_input)
        similarities = []
        
        for section_key, section_data in self.sections.items():
            section_name = section_data.get('section_name', '')
            normalized_name = self._normalize_section_name(section_name)
            
            similarity = SequenceMatcher(None, normalized_input, normalized_name).ratio()
            similarities.append((similarity, section_name))
        
        # Sort by similarity and return top suggestions
        similarities.sort(reverse=True)
        return [name for _, name in similarities[:limit]]

# Example usage and testing
if __name__ == "__main__":
    matcher = ICHSectionMatcher()
    
    # Test various input formats
    test_inputs = [
        "4.2 Rationale for Trial Design",
        "Rationale for Trial Design",
        "Trial Design Rationale",
        "6.2 Inclusion Criteria",
        "Inclusion Criteria",
        "Eligibility Criteria",
        "Section 12.1: Safety Monitoring",
        "Background",
        "Primary Objective"
    ]
    
    print("=== ICH M11 Section Matching Test ===")
    for test_input in test_inputs:
        result = matcher.validate_section_input(test_input)
        print(f"Input: '{test_input}'")
        print(f"Result: {result['message']}")
        if result['matched']:
            print(f"  → Section Key: {result['section_key']}")
            print(f"  → Category: {result['category']}")
            print(f"  → Priority: {result['priority']}")
        print() 