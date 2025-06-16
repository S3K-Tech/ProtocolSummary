from typing import List, Dict, Any
import re
from dataclasses import dataclass
# Removed backend.core.config import get_settings as it's not used in the provided class
import medspacy
from medspacy.ner import TargetRule
import spacy # Import spacy
# Removed unused import from medspacy.visualization import visualize_ent

@dataclass
class DrugInfo:
    name: str
    dosage: str = ""
    frequency: str = ""
    route: str = ""
    form: str = ""
    source_text: str = ""

# Known drug database for free lookup (can be expanded)
KNOWN_DRUGS = {
    # Diabetes drugs
    'semaglutide', 'sitagliptin', 'metformin', 'sulfonylurea', 'ozempic', 'januvia', 'glucophage', 
    'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    
    # Cardiovascular drugs
    'atorvastatin', 'simvastatin', 'lovastatin', 'pravastatin', 'rosuvastatin', 'fluvastatin',
    'lisinopril', 'enalapril', 'captopril', 'ramipril', 'benazepril', 'fosinopril',
    'losartan', 'valsartan', 'irbesartan', 'candesartan', 'telmisartan', 'olmesartan',
    
    # Antibiotics
    'amoxicillin', 'penicillin', 'ampicillin', 'azithromycin', 'erythromycin', 'clarithromycin',
    'ciprofloxacin', 'levofloxacin', 'moxifloxacin', 'doxycycline', 'minocycline', 'tetracycline',
    
    # Pain/inflammation
    'ibuprofen', 'naproxen', 'diclofenac', 'celecoxib', 'aspirin', 'acetaminophen', 'paracetamol',
    
    # Common generics (note: placebo and control are NOT drugs, they are comparators)
    'saline'
}

# Dosage units (more comprehensive)
DOSAGE_UNITS = {'mg', 'g', 'mcg', 'µg', 'ug', 'units', 'unit', 'ml', 'mL', 'L', 'l', 'kg', 'lb', 'IU', 'mmol', 'mol', 'mg/kg', 'mg/m2', 'mg/m²', 'tablets', 'tab', 'caps', 'capsules'}

# Frequency terms
FREQUENCY_TERMS = {
    'daily', 'once daily', 'qd', 'q.d.', 'twice daily', 'bid', 'b.i.d.', 'q12h',
    'three times daily', 'tid', 't.i.d.', 'q8h', 'four times daily', 'qid', 'q.i.d.', 'q6h',
    'weekly', 'once weekly', 'monthly', 'once monthly', 'prn', 'as needed', 'when required'
}

# Route terms
ROUTE_TERMS = {
    'oral', 'orally', 'by mouth', 'po', 'per os', 'intravenous', 'iv', 'i.v.', 'intravenously',
    'subcutaneous', 'sc', 's.c.', 'subcut', 'subcutaneously', 'intramuscular', 'im', 'i.m.',
    'intramuscularly', 'topical', 'topically', 'applied to skin', 'intranasal', 'nasal', 'nasally',
    'in', 'inhalation', 'inhaled', 'by inhalation', 'rectal', 'rectally', 'pr', 'transdermal'
}

# Form terms
FORM_TERMS = {
    'tablet', 'tab', 'pill', 'capsule', 'cap', 'caplet', 'injection', 'shot', 'vial', 'ampoule',
    'ampule', 'syringe', 'solution', 'liquid', 'syrup', 'suspension', 'elixir', 'cream', 'ointment',
    'gel', 'lotion', 'paste', 'powder', 'inhaler', 'nebulizer', 'spray', 'drops', 'patch',
    'suppository', 'enema', 'film', 'strip', 'lozenge'
}

# Define custom target rules for medical entities relevant to drug info
target_rules = [
    # === DRUG NAMES ===
    # Common diabetes drugs (from your example)
    TargetRule(r"\b(semaglutide|sitagliptin|metformin|sulfonylurea|ozempic|januvia|glucophage|glipizide|glyburide)\b", "DRUG"),
    TargetRule(r"\b(oral\s+semaglutide|subcutaneous\s+semaglutide)\b", "DRUG"),
    
    # Generic drug patterns
    TargetRule(r"\b\w+mab\b", "DRUG"),  # Monoclonal antibodies (ending in -mab)
    TargetRule(r"\b\w+nib\b", "DRUG"),  # Kinase inhibitors (ending in -nib)
    TargetRule(r"\b\w+pril\b", "DRUG"), # ACE inhibitors (ending in -pril)
    TargetRule(r"\b\w+sartan\b", "DRUG"), # ARBs (ending in -sartan)
    TargetRule(r"\b\w+statin\b", "DRUG"), # Statins (ending in -statin)
    TargetRule(r"\b\w+cillin\b", "DRUG"), # Penicillins (ending in -cillin)
    TargetRule(r"\b\w+mycin\b", "DRUG"), # Antibiotics (ending in -mycin)
    
    # === DOSAGES ===
    # Enhanced dosage patterns
    TargetRule(r"\b\d+\.?\d*\s*(mg|g|mcg|µg|units?|m?L|kg|lb|IU|mmol|ml)\b", "DOSAGE"),
    TargetRule(r"\b\d+\.?\d*\s*x\s*\d+\.?\d*\s*(mg|g|mcg|µg|units?|m?L|kg|lb|IU|mmol|ml)\b", "DOSAGE"), # e.g., "3 x 267 mg"
    TargetRule(r"\b\d+\.?\d*\s*mg/kg\b", "DOSAGE"), # Weight-based dosing
    TargetRule(r"\b\d+\.?\d*\s*mg/m²\b", "DOSAGE"), # Surface area-based dosing
    
    # === FREQUENCIES ===
    # Enhanced frequency patterns
    TargetRule(r"\b(daily|once\s+daily|QD|q\.?d\.?)\b", "FREQUENCY"),
    TargetRule(r"\b(twice\s+daily|BID|b\.?i\.?d\.?|q12h|every\s+12\s+hours?)\b", "FREQUENCY"),
    TargetRule(r"\b(three\s+times?\s+daily|TID|t\.?i\.?d\.?|q8h|every\s+8\s+hours?)\b", "FREQUENCY"),
    TargetRule(r"\b(four\s+times?\s+daily|QID|q\.?i\.?d\.?|q6h|every\s+6\s+hours?)\b", "FREQUENCY"),
    TargetRule(r"\b(weekly|once\s+weekly|every\s+week)\b", "FREQUENCY"),
    TargetRule(r"\b(monthly|once\s+monthly|every\s+month)\b", "FREQUENCY"),
    TargetRule(r"\b(every\s+\d+\s+(hours?|days?|weeks?|months?))\b", "FREQUENCY"),
    TargetRule(r"\b(PRN|as\s+needed|when\s+required)\b", "FREQUENCY"),
    
    # === ROUTES ===
    # Enhanced route patterns
    TargetRule(r"\b(oral|orally|by\s+mouth|PO|per\s+os)\b", "ROUTE"),
    TargetRule(r"\b(intravenous|IV|i\.?v\.?|intravenously)\b", "ROUTE"),
    TargetRule(r"\b(subcutaneous|SC|s\.?c\.?|subcut|subcutaneously)\b", "ROUTE"),
    TargetRule(r"\b(intramuscular|IM|i\.?m\.?|intramuscularly)\b", "ROUTE"),
    TargetRule(r"\b(topical|topically|applied\s+to\s+skin)\b", "ROUTE"),
    TargetRule(r"\b(intranasal|nasal|nasally|IN)\b", "ROUTE"),
    TargetRule(r"\b(inhalation|inhaled|by\s+inhalation)\b", "ROUTE"),
    TargetRule(r"\b(rectal|rectally|PR)\b", "ROUTE"),
    TargetRule(r"\b(transdermal|through\s+skin)\b", "ROUTE"),
    
    # === FORMS ===
    # Enhanced form patterns
    TargetRule(r"\b(tablet|tab|pill|capsule|cap|caplet)\b", "FORM"),
    TargetRule(r"\b(injection|shot|vial|ampoule|ampule|syringe)\b", "FORM"),
    TargetRule(r"\b(solution|liquid|syrup|suspension|elixir)\b", "FORM"),
    TargetRule(r"\b(cream|ointment|gel|lotion|paste|powder)\b", "FORM"),
    TargetRule(r"\b(inhaler|nebulizer|spray|drops|patch)\b", "FORM"),
    TargetRule(r"\b(suppository|enema|film|strip|lozenge)\b", "FORM"),
    
    # === COMPLEX PATTERNS ===
    # Drug with dosage patterns
    TargetRule(r"\b(semaglutide|sitagliptin|metformin)\s+\d+\.?\d*\s*(mg|g|mcg|µg|units?)\b", "DRUG_WITH_DOSAGE"),
    TargetRule(r"\b\d+\.?\d*\s*(mg|g|mcg|µg|units?)\s+(semaglutide|sitagliptin|metformin)\b", "DRUG_WITH_DOSAGE"),
    
    # NOTE: Removed placebo/control patterns - these are NOT drugs!
]

class DrugInfoExtractor:
    def __init__(self, method='medspacy'):
        """
        Initialize the drug extractor with choice of method.
        
        Args:
            method: 'medspacy' (default), 'regex', or 'hybrid'
        """
        self.method = method
        
        if method in ['medspacy', 'hybrid']:
            self._initialize_medspacy()
        
    def _initialize_medspacy(self):
        """Initialize MedSpaCy components"""
        # Load the medical NLP model provided by medspacy
        # This model includes pre-trained components for medical text
        try:
            # Create a blank English spaCy model and add necessary components
            self.nlp = spacy.blank("en")
            # Add a sentencizer to detect sentence boundaries
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
            
            # Add the target rule matcher to the pipeline if it's not already there
            if "medspacy_target_matcher" not in self.nlp.pipe_names:
                 self.nlp.add_pipe("medspacy_target_matcher", last=True)
            self.nlp.get_pipe("medspacy_target_matcher").add(target_rules)
        except Exception as e:
            print(f"Error loading medspacy model or adding rules: {e}")
            self.nlp = None # Handle case where model loading fails
    
    def extract_drug_info(self, text: str) -> List[DrugInfo]:
        """
        Extract drug information from text using the selected method.
        """
        if not text or not isinstance(text, str):
            print("Invalid text provided.")
            return []
        
        if self.method == 'medspacy':
            return self._extract_with_medspacy(text)
        elif self.method == 'regex':
            return self._extract_with_regex(text)
        elif self.method == 'hybrid':
            return self._extract_with_hybrid(text)
        else:
            print(f"Unknown method: {self.method}")
            return []
    
    def _extract_with_regex(self, text: str) -> List[DrugInfo]:
        """
        Extract drug information using conservative regex patterns.
        Focus on precision over recall - better to miss some drugs than extract non-drugs.
        """
        drug_info_list = []
        processed_drugs = set()
        
        # Split text into sentences for context
        sentences = re.split(r'[.!?]+', text)
        
        # First pass: collect all potential matches for each known drug
        drug_candidates = {}  # drug_name -> list of DrugInfo objects
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_lower = sentence.lower()
            
            # Method 1: Look for known drugs first (highest confidence)
            for known_drug in KNOWN_DRUGS:
                if known_drug in sentence_lower:
                    drug_info = DrugInfo(
                        name=known_drug.title(),
                        source_text=sentence
                    )
                    
                    # Check for multiple dosages first
                    multiple_dosages = self._extract_multiple_dosages_for_drug(sentence, known_drug)
                    
                    if multiple_dosages:
                        # Create separate entries for each dosage
                        for dosage in multiple_dosages:
                            drug_info_multi = DrugInfo(
                                name=known_drug.title(),
                                dosage=dosage,
                                source_text=sentence
                            )
                            
                            # Look for frequency, route, form
                            for freq_term in FREQUENCY_TERMS:
                                if freq_term.lower() in sentence_lower:
                                    drug_info_multi.frequency = freq_term
                                    break
                            
                            for route_term in ROUTE_TERMS:
                                if route_term.lower() in sentence_lower:
                                    drug_info_multi.route = route_term
                                    break
                            
                            for form_term in FORM_TERMS:
                                if form_term.lower() in sentence_lower:
                                    drug_info_multi.form = form_term
                                    break
                            
                            # Add to candidates
                            if known_drug not in drug_candidates:
                                drug_candidates[known_drug] = []
                            drug_candidates[known_drug].append(drug_info_multi)
                    else:
                        # Single dosage extraction
                        drug_info.dosage = self._extract_dosage_for_drug(sentence, known_drug)
                        
                        # Look for frequency
                        for freq_term in FREQUENCY_TERMS:
                            if freq_term.lower() in sentence_lower:
                                drug_info.frequency = freq_term
                                break
                        
                        # Look for route
                        for route_term in ROUTE_TERMS:
                            if route_term.lower() in sentence_lower:
                                drug_info.route = route_term
                                break
                        
                        # Look for form
                        for form_term in FORM_TERMS:
                            if form_term.lower() in sentence_lower:
                                drug_info.form = form_term
                                break
                        
                        # Add to candidates
                        if known_drug not in drug_candidates:
                            drug_candidates[known_drug] = []
                        drug_candidates[known_drug].append(drug_info)
        
        # Second pass: add all candidates (since we now handle multiple dosages as separate entries)
        for drug_name, candidates in drug_candidates.items():
            if drug_name in processed_drugs:
                continue
                
            # Add all candidates (each represents a different dosage)
            for candidate in candidates:
                drug_info_list.append(candidate)
            
            processed_drugs.add(drug_name)
        
        # Continue with the existing pattern matching for other drugs
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_lower = sentence.lower()
            
            # Method 2: Look for drug-like patterns with strict validation
            # Look for drug-like patterns with flexible dosage matching
            drug_suffix_patterns = [
                # Pattern 1: Drug with suffix + dosage immediately after (more specific suffixes)
                r'\b([A-Z][a-z]{4,}(?:pril|sartan|statin|gliptin|gliflozin|mycin|cillin|formin|vastatin|dipine|azole|oxacin))\s+(\d+\.?\d*\s*(?:' + '|'.join(DOSAGE_UNITS) + r'))\b',
                # Pattern 2: Dosage + Drug with suffix (more specific suffixes)
                r'\b(\d+\.?\d*\s*(?:' + '|'.join(DOSAGE_UNITS) + r'))\s+([A-Z][a-z]{4,}(?:pril|sartan|statin|gliptin|gliflozin|mycin|cillin|formin|vastatin|dipine|azole|oxacin))\b',
                # Pattern 3: Drug with suffix ... dosage (flexible with words between, more specific suffixes)
                r'\b([A-Z][a-z]{4,}(?:pril|sartan|statin|gliptin|gliflozin|mycin|cillin|formin|vastatin|dipine|azole|oxacin))[^.!?]*?(\d+\.?\d*\s*(?:' + '|'.join(DOSAGE_UNITS) + r'))\b'
            ]
            
            for pattern in drug_suffix_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    group1 = match.group(1).strip()
                    group2 = match.group(2).strip()
                    
                    # Determine which group is the drug and which is the dosage
                    if re.match(r'\d+', group1):
                        dosage, drug_candidate = group1, group2
                    else:
                        drug_candidate, dosage = group1, group2
                    
                    # Very strict validation - only allow if it's a single word and looks like a real drug
                    excluded_words = {'provide', 'glutide', 'adequate', 'glycemic', 'control', 'oral', 'daily', 'twice', 'once', 'prior', 'stable', 'include', 'exclude', 'require', 'ensure'}
                    if (len(drug_candidate.split()) == 1 and  # Single word only
                        len(drug_candidate) >= 6 and  # At least 6 characters for real drug names
                        drug_candidate.lower() not in processed_drugs and  # Not already processed
                        drug_candidate.lower() not in excluded_words and  # Not common words
                        drug_candidate.isalpha()):  # Only alphabetic characters
                        
                        drug_info = DrugInfo(
                            name=drug_candidate,
                            dosage=dosage,
                            source_text=sentence
                        )
                        
                        # Look for additional info
                        for freq_term in FREQUENCY_TERMS:
                            if freq_term.lower() in sentence_lower:
                                drug_info.frequency = freq_term
                                break
                        
                        for route_term in ROUTE_TERMS:
                            if route_term.lower() in sentence_lower:
                                drug_info.route = route_term
                                break
                        
                        for form_term in FORM_TERMS:
                            if form_term.lower() in sentence_lower:
                                drug_info.form = form_term
                                break
                        
                        drug_info_list.append(drug_info)
                        processed_drugs.add(drug_candidate.lower())
        
        return drug_info_list
    
    def _extract_with_hybrid(self, text: str) -> List[DrugInfo]:
        """
        Combine MedSpaCy and regex methods for maximum accuracy.
        """
        # Get results from both methods
        medspacy_results = self._extract_with_medspacy(text) if self.nlp else []
        regex_results = self._extract_with_regex(text)
        
        # Combine and deduplicate results
        combined_drugs = {}
        
        # Add MedSpaCy results first (usually more accurate)
        for drug in medspacy_results:
            key = drug.name.lower()
            combined_drugs[key] = drug
        
        # Add regex results, but enhance existing entries or add new ones
        for drug in regex_results:
            key = drug.name.lower()
            if key in combined_drugs:
                # Enhance existing entry with missing information
                existing = combined_drugs[key]
                if not existing.dosage and drug.dosage:
                    existing.dosage = drug.dosage
                if not existing.frequency and drug.frequency:
                    existing.frequency = drug.frequency
                if not existing.route and drug.route:
                    existing.route = drug.route
                if not existing.form and drug.form:
                    existing.form = drug.form
            else:
                # Add new drug found by regex
                combined_drugs[key] = drug
        
        result = list(combined_drugs.values())
        return result
    
    def _extract_with_medspacy(self, text: str) -> List[DrugInfo]:
        """
        Extract drug information from text using MedSpaCy with enhanced pattern matching.
        Returns a list of DrugInfo objects.
        """
        if not self.nlp or not text or not isinstance(text, str):
            print("MedSpacy model not loaded or invalid text provided.")
            return []
            
        doc = self.nlp(text)
        drug_info_list = []
        
        # First pass: collect all entities by type
        entities_by_type = {
            'DRUG': [],
            'DOSAGE': [],
            'FREQUENCY': [],
            'ROUTE': [],
            'FORM': [],
            'DRUG_WITH_DOSAGE': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities_by_type:
                entities_by_type[ent.label_].append(ent)
        
        # Second pass: process drugs and find associated information
        processed_drugs = set()  # Avoid duplicates
        
        # Process direct drug entities
        for drug_ent in entities_by_type['DRUG']:
            drug_name = drug_ent.text.strip()
            if drug_name.lower() in processed_drugs:
                continue
            
            drug_info = DrugInfo(
                name=drug_name,
                source_text=drug_ent.sent.text.strip()
            )
            
            # Find associated information within the same sentence
            sentence_start = drug_ent.sent.start_char
            sentence_end = drug_ent.sent.end_char
            
            # Look for dosage, frequency, route, form in the same sentence
            for ent_type, entities in entities_by_type.items():
                if ent_type == 'DRUG':
                    continue
                    
                for ent in entities:
                    # Check if entity is in the same sentence
                    if sentence_start <= ent.start_char <= sentence_end:
                        if ent_type == 'DOSAGE' and not drug_info.dosage:
                            drug_info.dosage = ent.text.strip()
                        elif ent_type == 'FREQUENCY' and not drug_info.frequency:
                            drug_info.frequency = ent.text.strip()
                        elif ent_type == 'ROUTE' and not drug_info.route:
                            drug_info.route = ent.text.strip()
                        elif ent_type == 'FORM' and not drug_info.form:
                            drug_info.form = ent.text.strip()
            
            drug_info_list.append(drug_info)
            processed_drugs.add(drug_name.lower())
        
        # Process complex drug patterns (DRUG_WITH_DOSAGE)
        for complex_ent in entities_by_type['DRUG_WITH_DOSAGE']:
            # Extract drug name and dosage from the complex pattern
            complex_text = complex_ent.text
            
            # Simple parsing for drug with dosage
            match = re.search(r'(\w+(?:\s+\w+)*)\s+(\d+\.?\d*\s*(?:mg|g|mcg|µg|units?))', complex_text, re.IGNORECASE)
            if match:
                drug_name = match.group(1).strip()
                dosage = match.group(2).strip()
                
                if drug_name.lower() not in processed_drugs:
                    drug_info = DrugInfo(
                        name=drug_name,
                        dosage=dosage,
                        source_text=complex_ent.sent.text.strip()
                    )
                    
                    # Look for additional info in the same sentence
                    sentence_start = complex_ent.sent.start_char
                    sentence_end = complex_ent.sent.end_char
                    
                    for ent_type, entities in entities_by_type.items():
                        if ent_type in ['DRUG', 'DOSAGE', 'DRUG_WITH_DOSAGE']:
                            continue
                            
                        for ent in entities:
                            if sentence_start <= ent.start_char <= sentence_end:
                                if ent_type == 'FREQUENCY' and not drug_info.frequency:
                                    drug_info.frequency = ent.text.strip()
                                elif ent_type == 'ROUTE' and not drug_info.route:
                                    drug_info.route = ent.text.strip()
                                elif ent_type == 'FORM' and not drug_info.form:
                                    drug_info.form = ent.text.strip()
                    
                    drug_info_list.append(drug_info)
                    processed_drugs.add(drug_name.lower())
        
        # Additional pattern matching for commonly missed combinations
        self._extract_additional_patterns(text, drug_info_list, processed_drugs)
        
        return drug_info_list
    
    def _extract_additional_patterns(self, text: str, drug_info_list: List[DrugInfo], processed_drugs: set):
        """Extract additional drug patterns that might be missed by standard NER"""
        import re
        
        # Pattern for "X mg of [drug]" or "[drug] X mg"
        dosage_drug_patterns = [
            r'(\d+\.?\d*\s*(?:mg|g|mcg|µg|units?))\s+(?:of\s+)?(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(\d+\.?\d*\s*(?:mg|g|mcg|µg|units?))',
            r'(\w+(?:\s+\w+)*)\s*[,\s]\s*(\d+\.?\d*\s*(?:mg|g|mcg|µg|units?))'
        ]
        
        for pattern in dosage_drug_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                group1, group2 = match.groups()
                
                # Determine which is drug and which is dosage
                if re.match(r'\d+', group1):
                    dosage, drug_name = group1.strip(), group2.strip()
                else:
                    drug_name, dosage = group1.strip(), group2.strip()
                
                # Filter out common non-drug words
                non_drug_words = {'the', 'of', 'with', 'in', 'for', 'by', 'to', 'and', 'or', 'dose', 'doses', 'treatment', 'therapy', 'medication', 'drug'}
                if drug_name.lower() not in non_drug_words and drug_name.lower() not in processed_drugs:
                    # Find the sentence containing this match
                    start_pos = match.start()
                    sentences = text.split('.')
                    current_pos = 0
                    source_sentence = ""
                    
                    for sentence in sentences:
                        if current_pos <= start_pos <= current_pos + len(sentence):
                            source_sentence = sentence.strip()
                            break
                        current_pos += len(sentence) + 1
                    
                    drug_info = DrugInfo(
                        name=drug_name,
                        dosage=dosage,
                        source_text=source_sentence
                    )
                    
                    drug_info_list.append(drug_info)
                    processed_drugs.add(drug_name.lower())
    
    def _extract_multiple_dosages_for_drug(self, text: str, drug_name: str) -> List[str]:
        """
        Extract multiple dosages for a specific drug from text.
        Returns a list of dosage strings if multiple dosages found, empty list otherwise.
        """
        # Pattern for multiple dosages in parentheses like "semaglutide (3 mg, 7 mg, and 14 mg)"
        multiple_dosage_pattern = rf'\b{re.escape(drug_name)}\s*\([^)]*?((?:\d+(?:\.\d+)?\s*(?:mg|g|mcg|µg|units?)[^)]*?)+)\)'
        match = re.search(multiple_dosage_pattern, text, re.IGNORECASE)
        
        if match:
            dosage_text = match.group(1)
            # Extract all individual dosages from the parentheses content
            individual_dosages = re.findall(r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|units?)', dosage_text, re.IGNORECASE)
            
            if len(individual_dosages) > 1:  # Only return if multiple dosages found
                return [f"{number} {unit}" for number, unit in individual_dosages]
        
        # Also check for patterns like "semaglutide at doses of 3mg, 7mg, and 14mg"
        doses_pattern = rf'\b{re.escape(drug_name)}\s+(?:at\s+)?(?:doses?\s+of\s+|dosages?\s+of\s+)([^.!?]*)'
        match = re.search(doses_pattern, text, re.IGNORECASE)
        
        if match:
            dosage_text = match.group(1)
            individual_dosages = re.findall(r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|units?)', dosage_text, re.IGNORECASE)
            
            if len(individual_dosages) > 1:  # Only return if multiple dosages found
                return [f"{number} {unit}" for number, unit in individual_dosages]
        
        return []  # No multiple dosages found

    def _extract_dosage_for_drug(self, text: str, drug_name: str) -> str:
        """
        Extract dosage for a specific drug from text using multiple sophisticated patterns.
        Returns the dosage string or empty string if not found.
        """
        # Skip multiple dosage check here since it's handled separately now
        
        # Create comprehensive dosage patterns
        dosage_unit_pattern = '|'.join(re.escape(unit) for unit in DOSAGE_UNITS)
        
        patterns = [
            # Pattern 1: "drug_name NUMBER UNIT" (e.g., "metformin 500 mg")
            rf'\b{re.escape(drug_name)}\s+(\d+(?:\.\d+)?)\s*({dosage_unit_pattern})\b',
            
            # Pattern 2: "NUMBER UNIT drug_name" (e.g., "500 mg metformin")
            rf'\b(\d+(?:\.\d+)?)\s*({dosage_unit_pattern})\s+{re.escape(drug_name)}\b',
            
            # Pattern 3: "drug_name ... NUMBER UNIT" (flexible, within same sentence)
            rf'\b{re.escape(drug_name)}\b[^.!?]*?(\d+(?:\.\d+)?)\s*({dosage_unit_pattern})\b',
            
            # Pattern 4: "NUMBER UNIT ... drug_name" (reverse flexible)
            rf'\b(\d+(?:\.\d+)?)\s*({dosage_unit_pattern})\b[^.!?]*?{re.escape(drug_name)}\b',
            
            # Pattern 5: Range patterns like "≥1500 mg" or "maximum 1500 mg"
            rf'\b{re.escape(drug_name)}\b[^.!?]*?(?:≥|>=|maximum|max|minimum|min)?\s*(\d+(?:\.\d+)?)\s*({dosage_unit_pattern})\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(1)
                unit = match.group(2)
                return f"{number} {unit}"
        
        return ""  # No dosage found

    def compare_drug_info(self, generated_drugs: List[DrugInfo], reference_drugs: List[DrugInfo]) -> Dict[str, Any]:
        """
        Compare extracted drug information between generated and reference lists.
        Each drug+dosage combination is treated as a separate entity.
        Returns a dictionary with comparison results (matches, missing, additional).
        """
        comparison = {
            "matches": [], # Drug+dosage combinations found in both
            "missing": [], # Drug+dosage combinations in reference but not in generated
            "additional": [] # Drug+dosage combinations in generated but not in reference
        }
        
        # Create unique keys for each drug+dosage combination
        def create_drug_key(drug: DrugInfo) -> str:
            """Create a unique key for drug+dosage combination"""
            name = drug.name.lower().strip()
            dosage = drug.dosage.lower().strip() if drug.dosage else ""
            return f"{name}|{dosage}"
        
        # Create dictionaries for easier lookup
        ref_drugs_by_key = {create_drug_key(drug): drug for drug in reference_drugs}
        gen_drugs_by_key = {create_drug_key(drug): drug for drug in generated_drugs}
        
        # Find matches (same drug+dosage combination in both)
        for ref_key, ref_drug in ref_drugs_by_key.items():
            if ref_key in gen_drugs_by_key:
                gen_drug = gen_drugs_by_key[ref_key]
                
                # Check if all details match
                details_match = (
                    ref_drug.name.lower() == gen_drug.name.lower() and
                    ref_drug.dosage.lower() == gen_drug.dosage.lower() and
                    ref_drug.frequency.lower() == gen_drug.frequency.lower() and
                    ref_drug.route.lower() == gen_drug.route.lower() and
                    ref_drug.form.lower() == gen_drug.form.lower()
                )
                
                comparison["matches"].append({
                    "name": ref_drug.name,
                    "reference": ref_drug,
                    "generated": gen_drug,
                    "details_match": details_match
                })
        
        # Find missing (in reference but not in generated)
        for ref_key, ref_drug in ref_drugs_by_key.items():
            if ref_key not in gen_drugs_by_key:
                comparison["missing"].append(ref_drug)
        
        # Find additional (in generated but not in reference)
        for gen_key, gen_drug in gen_drugs_by_key.items():
            if gen_key not in ref_drugs_by_key:
                comparison["additional"].append(gen_drug)
                
        return comparison 