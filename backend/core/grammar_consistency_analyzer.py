import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import yaml
from dataclasses import dataclass
from collections import Counter

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing utilities and config
from backend.core.config import get_config
from backend.core.utils.section_matcher import ICHSectionMatcher

# Try to import LanguageTool, handle if not installed
try:
    from language_tool_python import LanguageTool
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    print("Warning: language_tool_python not installed. Grammar checking will be limited.")

import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

@dataclass
class AnalysisIssue:
    type: str  # "grammar", "consistency", "ich_compliance", "terminology"
    severity: str  # "high", "medium", "low"
    message: str
    suggestion: str
    location: str = ""
    category: str = ""

@dataclass
class AnalysisResult:
    grammar_score: float
    consistency_score: float
    compliance_score: float
    overall_score: float
    issues: List[AnalysisIssue]
    summary: str
    recommendations: List[str]

class GrammarConsistencyAnalyzer:
    """
    Medical Grammar and Consistency Analyzer for Clinical Trial Protocols
    Integrates with existing codebase structure
    """
    
    def __init__(self):
        self.config = get_config()
        self.section_matcher = ICHSectionMatcher()
        self._load_medical_rules()
        self._initialize_grammar_tool()
    
    def _load_medical_rules(self):
        """Load medical terminology and ICH rules"""
        try:
            # Load medical terminology from medical_rules folder
            terminology_path = Path(__file__).parent / "medical_rules" / "medical_terminology.yaml"
            with open(terminology_path, 'r') as f:
                self.medical_terms = yaml.safe_load(f)
            
            # ICH rules already loaded by section_matcher
            self.ich_rules = self.section_matcher.rules
            
            # Create flat list of all medical terms for filtering
            self._create_medical_terms_set()
            
            logging.info("Medical rules loaded successfully")
        except Exception as e:
            logging.error(f"Error loading medical rules: {e}")
            self.medical_terms = {}
            self.ich_rules = {}
            self.all_medical_terms = set()
    
    def _create_medical_terms_set(self):
        """Create a flat set of all medical terms for quick lookup"""
        self.all_medical_terms = set()
        
        medical_dict = self.medical_terms.get('medical_terms', {}).get('medical_dictionary', {})
        
        # Add all terms from all categories
        for category, terms in medical_dict.items():
            if isinstance(terms, list):
                for term in terms:
                    # Add both exact case and lowercase for flexible matching
                    self.all_medical_terms.add(term.lower())
                    self.all_medical_terms.add(term)
        
        # Also add abbreviations
        abbreviations = self.medical_terms.get('medical_terms', {}).get('abbreviations', {})
        for abbrev, full_form in abbreviations.items():
            self.all_medical_terms.add(abbrev.lower())
            self.all_medical_terms.add(full_form.lower())
        
        # Add preferred terms
        preferred_terms = self.medical_terms.get('medical_terms', {}).get('study_terminology', {}).get('preferred_terms', {})
        for preferred, variants in preferred_terms.items():
            self.all_medical_terms.add(preferred.lower())
            for variant in variants:
                self.all_medical_terms.add(variant.lower())
        
        logging.info(f"Loaded {len(self.all_medical_terms)} medical terms for filtering")
    
    def _is_medical_term(self, word: str) -> bool:
        """Check if a word is a legitimate medical term"""
        if not hasattr(self, 'all_medical_terms'):
            return False
        
        # Clean the word (remove punctuation, spaces)
        clean_word = word.strip().lower().rstrip('.,;:!?()[]{}')
        
        return clean_word in self.all_medical_terms
    
    def _initialize_grammar_tool(self):
        """Initialize grammar checking tool if available"""
        if LANGUAGE_TOOL_AVAILABLE:
            try:
                self.grammar_tool = LanguageTool('en-US')
                logging.info("LanguageTool initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing LanguageTool: {e}")
                self.grammar_tool = None
        else:
            self.grammar_tool = None
    
    def analyze_section(self, text: str, section_name: str = "", reference_text: str = "") -> AnalysisResult:
        """
        Main analysis function - comprehensive medical grammar and consistency check
        
        Args:
            text: The protocol section text to analyze
            section_name: Name/title of the section (flexible format)
            reference_text: Optional reference text for comparison
            
        Returns:
            AnalysisResult with scores, issues, and recommendations
        """
        try:
            logging.info(f"Starting analysis for section: {section_name}")
            
            # Initialize issues list
            all_issues = []
            
            # 1. Grammar Analysis
            grammar_issues = self._analyze_grammar(text)
            all_issues.extend(grammar_issues)
            
            # 2. Medical Terminology Consistency
            terminology_issues = self._analyze_medical_terminology(text)
            all_issues.extend(terminology_issues)
            
            # 3. General Consistency Checks
            consistency_issues = self._analyze_consistency(text, reference_text)
            all_issues.extend(consistency_issues)
            
            # 4. ICH M11 Compliance (if section identified)
            compliance_issues = self._analyze_ich_compliance(text, section_name)
            all_issues.extend(compliance_issues)
            
            # 5. Calculate scores
            scores = self._calculate_scores(all_issues)
            
            # 6. Generate summary and recommendations
            summary = self._generate_summary(all_issues, scores)
            recommendations = self._generate_recommendations(all_issues, section_name)
            
            return AnalysisResult(
                grammar_score=scores['grammar'],
                consistency_score=scores['consistency'],
                compliance_score=scores['compliance'],
                overall_score=scores['overall'],
                issues=all_issues,
                summary=summary,
                recommendations=recommendations
            )
            
        except Exception as e:
            logging.error(f"Error in analysis: {e}")
            return AnalysisResult(
                grammar_score=0,
                consistency_score=0,
                compliance_score=0,
                overall_score=0,
                issues=[AnalysisIssue("error", "high", f"Analysis failed: {str(e)}", "Please check the input and try again")],
                summary="Analysis could not be completed due to an error",
                recommendations=["Please check the input text and try again"]
            )
    
    def _analyze_grammar(self, text: str) -> List[AnalysisIssue]:
        """Analyze grammar and writing style"""
        issues = []
        
        # LanguageTool grammar check (if available)
        if self.grammar_tool:
            try:
                grammar_errors = self.grammar_tool.check(text)
                for error in grammar_errors[:15]:  # Check more, but filter medical terms
                    # Extract the problematic text and surrounding context
                    problem_text = text[error.offset:error.offset + error.errorLength]
                    
                    # FILTER: Skip if the flagged word is a legitimate medical term
                    if self._is_medical_term(problem_text):
                        logging.debug(f"Filtered medical term: '{problem_text}'")
                        continue
                    
                    # Also check if suggested replacement is actually wrong for medical context
                    if error.replacements and len(error.replacements) > 0:
                        suggested = error.replacements[0]
                        # If it's suggesting to change a medical term to a non-medical term, skip
                        if self._is_medical_term(problem_text) and not self._is_medical_term(suggested):
                            logging.debug(f"Filtered medical term replacement: '{problem_text}' -> '{suggested}'")
                            continue
                    
                    context = self._extract_sentence_context(text, error.offset, error.errorLength)
                    
                    # Create clear before/after suggestion
                    if error.replacements:
                        replacement = error.replacements[0]
                        suggestion_text = f"Change '{problem_text}' to '{replacement}'"
                        
                        # Show context with the change
                        improved_context = context.replace(problem_text, f"**{replacement}**", 1)
                        location_text = f"📝 **Original:** {context}\n\n✅ **Suggested:** {improved_context}"
                    else:
                        suggestion_text = f"Review '{problem_text}'"
                        location_text = f"📝 **Context:** {context}"
                    
                    issues.append(AnalysisIssue(
                        type="grammar",
                        severity="medium",
                        message=error.message,
                        suggestion=suggestion_text,
                        location=location_text,
                        category="grammar"
                    ))
            except Exception as e:
                logging.warning(f"LanguageTool error: {e}")
        
        # Medical writing style checks
        issues.extend(self._check_medical_writing_style(text))
        
        return issues
    
    def _extract_sentence_context(self, text: str, offset: int, length: int) -> str:
        """Extract the sentence containing the error for better context"""
        try:
            # Find the sentence boundaries around the error
            start = max(0, offset - 100)  # Look back up to 100 chars
            end = min(len(text), offset + length + 100)  # Look ahead up to 100 chars
            
            # Extract the context
            context = text[start:end].strip()
            
            # Try to find complete sentence boundaries
            sentences = re.split(r'[.!?]+', context)
            if len(sentences) > 1:
                # Find which sentence contains our error
                error_pos = offset - start
                current_pos = 0
                for sentence in sentences:
                    if current_pos <= error_pos <= current_pos + len(sentence):
                        return sentence.strip()
                    current_pos += len(sentence) + 1  # +1 for the delimiter
            
            # Fallback: return the context with ellipsis if needed
            result = context
            if start > 0:
                result = "..." + result
            if end < len(text):
                result = result + "..."
            
            return result
            
        except Exception:
            # Fallback: just return a simple context
            start = max(0, offset - 50)
            end = min(len(text), offset + length + 50)
            return text[start:end].strip()
    
    def _check_medical_writing_style(self, text: str) -> List[AnalysisIssue]:
        """Check medical writing style guidelines"""
        issues = []
        
        # Check passive voice overuse
        passive_patterns = [
            r'\b(was|were|is|are|been|being)\s+\w+ed\b',
            r'\b(was|were|is|are|been|being)\s+\w+en\b'
        ]
        
        passive_count = 0
        for pattern in passive_patterns:
            passive_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        sentences = len(re.split(r'[.!?]+', text))
        if sentences > 0:
            passive_percentage = (passive_count / sentences) * 100
            if passive_percentage > 60:  # Too much passive voice
                issues.append(AnalysisIssue(
                    type="grammar",
                    severity="medium",
                    message=f"High passive voice usage ({passive_percentage:.1f}%)",
                    suggestion="Consider using more active voice for clarity in medical writing",
                    category="style"
                ))
        
        # Check sentence length
        sentences_text = re.split(r'[.!?]+', text)
        long_sentences = [s for s in sentences_text if len(s.split()) > 30]
        if long_sentences:
            issues.append(AnalysisIssue(
                type="grammar",
                severity="low",
                message=f"{len(long_sentences)} sentences exceed 30 words",
                suggestion="Break long sentences into shorter ones for better readability",
                category="style"
            ))
        
        return issues
    
    def _analyze_medical_terminology(self, text: str) -> List[AnalysisIssue]:
        """Analyze medical terminology consistency"""
        issues = []
        
        if not self.medical_terms:
            return issues
        
        # Check terminology consistency
        preferred_terms = self.medical_terms.get('medical_terms', {}).get('study_terminology', {}).get('preferred_terms', {})
        
        for preferred_term, variants in preferred_terms.items():
            found_variants = []
            variant_contexts = []
            
            for variant in variants:
                if variant.lower() in text.lower():
                    found_variants.append(variant)
                    # Find context for this variant
                    variant_pos = text.lower().find(variant.lower())
                    if variant_pos != -1:
                        context = self._extract_sentence_context(text, variant_pos, len(variant))
                        variant_contexts.append(f"'{variant}' in: {context}")
            
            # If multiple variants found, suggest consistency
            if len(found_variants) > 1:
                location_text = "📝 **Found inconsistencies:**\n" + "\n".join(variant_contexts)
                issues.append(AnalysisIssue(
                    type="terminology",
                    severity="medium",
                    message=f"Mixed terminology: {', '.join(found_variants)}",
                    suggestion=f"Use '{preferred_term}' consistently throughout the document",
                    location=location_text,
                    category="consistency"
                ))
        
        # Check abbreviation definitions
        abbreviations = self.medical_terms.get('medical_terms', {}).get('abbreviations', {})
        for abbrev, full_form in abbreviations.items():
            if abbrev in text and full_form not in text:
                # Check if it's the first occurrence
                abbrev_pos = text.find(abbrev)
                if abbrev_pos < 200:  # Early in text
                    context = self._extract_sentence_context(text, abbrev_pos, len(abbrev))
                    location_text = f"📝 **First use found in:** {context}\n\n✅ **Should be:** {context.replace(abbrev, f'{full_form} ({abbrev})', 1)}"
                    
                    issues.append(AnalysisIssue(
                        type="terminology",
                        severity="low",
                        message=f"'{abbrev}' used without definition",
                        suggestion=f"Define on first use: '{full_form} ({abbrev})'",
                        location=location_text,
                        category="abbreviations"
                    ))
        
        # Check dosage format
        dosage_patterns = self.medical_terms.get('dosage_patterns', {})
        if dosage_patterns:
            invalid_pattern = dosage_patterns.get('invalid_format', '')
            if invalid_pattern:
                invalid_matches = list(re.finditer(invalid_pattern, text, re.IGNORECASE))
                if invalid_matches:
                    for match in invalid_matches[:3]:  # Show first 3 instances
                        dosage_text = match.group()
                        context = self._extract_sentence_context(text, match.start(), len(dosage_text))
                        
                        # Suggest abbreviated format
                        abbreviated = dosage_text.replace("milligrams", "mg").replace("grams", "g").replace("kilograms", "kg").replace("milliliters", "mL").replace("liters", "L")
                        location_text = f"📝 **Found:** {context}\n\n✅ **Should be:** {context.replace(dosage_text, abbreviated, 1)}"
                        
                        issues.append(AnalysisIssue(
                            type="terminology",
                            severity="low",
                            message=f"Non-standard dosage format: '{dosage_text}'",
                            suggestion=f"Use abbreviated units: '{abbreviated}'",
                            location=location_text,
                            category="formatting"
                        ))
        
        return issues
    
    def _analyze_consistency(self, text: str, reference_text: str = "") -> List[AnalysisIssue]:
        """Analyze general consistency"""
        issues = []
        
        # Date format consistency
        date_patterns = {
            "dd/mm/yyyy": r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            "dd-mm-yyyy": r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            "dd mmm yyyy": r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
        }
        
        found_formats = {}
        date_examples = {}
        
        for format_name, pattern in date_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                found_formats[format_name] = len(matches)
                # Get first example with context
                first_match = matches[0]
                context = self._extract_sentence_context(text, first_match.start(), len(first_match.group()))
                date_examples[format_name] = f"'{first_match.group()}' in: {context}"
        
        if len(found_formats) > 1:
            format_list = "\n".join([f"• {fmt}: {date_examples[fmt]}" for fmt in found_formats.keys()])
            location_text = f"📝 **Different formats found:**\n{format_list}"
            
            issues.append(AnalysisIssue(
                type="consistency",
                severity="medium",
                message="Multiple date formats used",
                suggestion="Use consistent date format (recommend DD MMM YYYY, e.g., '15 Mar 2024')",
                location=location_text,
                category="formatting"
            ))
        
        # Number formatting consistency
        # Check for both "5 mg" and "5mg" patterns
        spaced_matches = list(re.finditer(r'\d+\s+mg\b', text))
        unspaced_matches = list(re.finditer(r'\d+mg\b', text))
        
        if spaced_matches and unspaced_matches:
            examples = []
            if spaced_matches:
                first_spaced = spaced_matches[0]
                spaced_context = self._extract_sentence_context(text, first_spaced.start(), len(first_spaced.group()))
                examples.append(f"• With space: '{first_spaced.group()}' in: {spaced_context}")
            
            if unspaced_matches:
                first_unspaced = unspaced_matches[0]
                unspaced_context = self._extract_sentence_context(text, first_unspaced.start(), len(first_unspaced.group()))
                examples.append(f"• Without space: '{first_unspaced.group()}' in: {unspaced_context}")
            
            location_text = f"📝 **Inconsistent spacing found:**\n" + "\n".join(examples)
            
            issues.append(AnalysisIssue(
                type="consistency",
                severity="low",
                message="Inconsistent spacing in dosage units",
                suggestion="Use consistent spacing: '5 mg' (with space) for better readability",
                location=location_text,
                category="formatting"
            ))
        
        # Enhanced consistency checks
        
        # 1. Terminology consistency - check for mixed terms
        terminology_variants = {
            "participant": ["participant", "subject", "patient"],
            "randomized": ["randomized", "randomised"],
            "placebo": ["placebo", "dummy"],
            "adverse event": ["adverse event", "AE", "side effect"],
            "efficacy": ["efficacy", "effectiveness"],
            "double-blind": ["double-blind", "double-blinded", "double blind"]
        }
        
        for preferred_term, variants in terminology_variants.items():
            found_variants = []
            for variant in variants:
                if re.search(r'\b' + re.escape(variant) + r'\b', text, re.IGNORECASE):
                    found_variants.append(variant)
            
            if len(found_variants) > 1:
                issues.append(AnalysisIssue(
                    type="consistency",
                    severity="medium",
                    message=f"Mixed terminology: {', '.join(found_variants)}",
                    suggestion=f"Use '{preferred_term}' consistently throughout the section",
                    location="",
                    category="terminology"
                ))
        
        # 2. Abbreviation consistency
        # Find abbreviations and check if they're properly defined
        abbrev_pattern = r'\b[A-Z]{2,}(?:[0-9]+)?\b'
        abbreviations = list(set(re.findall(abbrev_pattern, text)))
        
        for abbrev in abbreviations:
            # Check if abbreviation is defined (look for pattern: "Full Form (ABBREV)")
            definition_pattern = rf'\([^)]*{re.escape(abbrev)}[^)]*\)'
            if not re.search(definition_pattern, text):
                issues.append(AnalysisIssue(
                    type="consistency",
                    severity="low",
                    message=f"Abbreviation '{abbrev}' may not be properly defined",
                    suggestion=f"Define abbreviation on first use: 'Full Form ({abbrev})'",
                    location="",
                    category="abbreviations"
                ))
        
        # 3. Sentence length variation check
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            very_long = [i for i, length in enumerate(sentence_lengths) if length > 40]
            very_short = [i for i, length in enumerate(sentence_lengths) if length < 5 and length > 0]
            
            if len(very_long) > len(sentence_lengths) * 0.3:  # More than 30% very long
                issues.append(AnalysisIssue(
                    type="consistency",
                    severity="medium",
                    message=f"Many sentences are very long (avg: {avg_length:.1f} words)",
                    suggestion="Consider breaking long sentences for better readability",
                    location="",
                    category="style"
                ))
        
        # 4. Clinical language consistency
        informal_terms = ["great", "awesome", "pretty good", "kind of", "sort of", "really", "very good"]
        formal_alternatives = {
            "great": "significant/substantial",
            "awesome": "excellent/superior", 
            "pretty good": "satisfactory/adequate",
            "kind of": "somewhat/partially",
            "sort of": "somewhat/partially",
            "really": "significantly/substantially",
            "very good": "excellent/satisfactory"
        }
        
        for informal in informal_terms:
            if re.search(r'\b' + re.escape(informal) + r'\b', text, re.IGNORECASE):
                suggestion = formal_alternatives.get(informal, "more formal clinical language")
                issues.append(AnalysisIssue(
                    type="consistency",
                    severity="medium",
                    message=f"Informal language detected: '{informal}'",
                    suggestion=f"Consider using {suggestion}",
                    location="",
                    category="clinical_language"
                ))
        
        return issues
    
    def _analyze_ich_compliance(self, text: str, section_name: str) -> List[AnalysisIssue]:
        """Analyze ICH M11 compliance"""
        issues = []
        
        if not section_name:
            return issues
        
        # Try to match section to ICH M11 requirements
        match_result = self.section_matcher.match_section(section_name)
        if not match_result:
            # If no section match, apply general ICH compliance checks
            issues.extend(self._check_general_ich_requirements(text))
            return issues
        
        section_key, section_data, confidence = match_result
        
        # Check required content elements
        required_elements = section_data.get('required_content_elements', {})
        for element_name, element_config in required_elements.items():
            if element_config.get('mandatory'):
                keywords = element_config.get('keywords', [])
                if keywords:
                    found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                    if not found_keywords:
                        issues.append(AnalysisIssue(
                            type="ich_compliance",
                            severity="high",
                            message=f"Missing required element: {element_name.replace('_', ' ')}",
                            suggestion=f"Include content about {element_name.replace('_', ' ')}. Expected keywords: {', '.join(keywords)}",
                            category="ich_compliance"
                        ))
        
        # Check content length requirements
        content_reqs = section_data.get('content_requirements', {})
        if content_reqs:
            word_count = len(text.split())
            min_words = content_reqs.get('min_words', 0)
            max_words = content_reqs.get('max_words', float('inf'))
            
            if word_count < min_words:
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="medium",
                    message=f"Content too short: {word_count} words (minimum: {min_words})",
                    suggestion="Add more detailed information to meet ICH M11 requirements",
                    category="ich_compliance"
                ))
            elif word_count > max_words:
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="low",
                    message=f"Content too long: {word_count} words (maximum: {max_words})",
                    suggestion="Consider condensing the content while maintaining required information",
                    category="ich_compliance"
                ))
        
        # Enhanced regulatory language checks
        issues.extend(self._check_regulatory_language(text, section_data))
        
        # Section-specific validation
        issues.extend(self._check_section_specific_requirements(text, section_key, section_data))
        
        # General ICH compliance checks
        issues.extend(self._check_general_ich_requirements(text))
        
        return issues
    
    def _check_regulatory_language(self, text: str, section_data: Dict) -> List[AnalysisIssue]:
        """Check for appropriate regulatory language based on section type"""
        issues = []
        
        # Get section category and requirements
        section_category = section_data.get('category', '')
        section_key = section_data.get('section_key', '')
        
        # Get regulatory requirements from medical terminology
        regulatory_config = self.medical_terms.get('medical_terms', {}).get('regulatory_phrases', {})
        
        # Check if this section requires ethics/regulatory language
        ethics_sections = regulatory_config.get('ethics_and_regulatory_sections', {}).get('sections', [])
        requires_ethics_language = any(
            ethics_section in section_key.lower() or ethics_section in section_category.lower()
            for ethics_section in ethics_sections
        )
        
        if requires_ethics_language:
            # Only check for regulatory phrases in ethics/regulatory sections
            required_phrases = regulatory_config.get('ethics_and_regulatory_sections', {}).get('required', [])
            for phrase in required_phrases:
                if phrase.lower() not in text.lower():
                    issues.append(AnalysisIssue(
                        type="ich_compliance",
                        severity="medium",
                        message=f"Missing regulatory phrase: '{phrase}'",
                        suggestion=f"Consider including '{phrase}' as required for regulatory compliance",
                        category="Regulatory Language"
                    ))
        
        # Check for inappropriate language (applies to all sections)
        avoid_phrases = regulatory_config.get('avoid', [])
        for phrase in avoid_phrases:
            if phrase.lower() in text.lower():
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="high",
                    message=f"Inappropriate language: '{phrase}'",
                    suggestion=f"Remove or replace '{phrase}' with more appropriate regulatory language",
                    category="Inappropriate Language"
                ))
        
        # Check for preferred terminology
        preferred_terms = regulatory_config.get('preferred_clinical_terms', {})
        for preferred, alternatives in preferred_terms.items():
            for alternative in alternatives:
                if alternative.lower() in text.lower() and preferred.lower() not in text.lower():
                    issues.append(AnalysisIssue(
                        type="ich_compliance",
                        severity="low",
                        message=f"Consider using preferred term: '{preferred}' instead of '{alternative}'",
                        suggestion=f"Replace '{alternative}' with '{preferred}' for consistency with ICH M11",
                        category="Terminology Preference"
                    ))
        
        return issues
    
    def _check_section_specific_requirements(self, text: str, section_key: str, section_data: Dict) -> List[AnalysisIssue]:
        """Check section-specific ICH M11 requirements"""
        issues = []
        
        # Rationale for Trial Design specific checks
        if section_key == "rationale_for_trial_design":
            # Must justify intervention model
            intervention_terms = ["intervention model", "design", "randomized", "controlled", "blinded"]
            if not any(term in text.lower() for term in intervention_terms):
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="high",
                    message="Missing intervention model justification",
                    suggestion="Explain why the chosen intervention model (e.g., randomized, controlled) is appropriate",
                    category="ich_compliance"
                ))
            
            # Must justify duration
            duration_terms = ["duration", "weeks", "months", "sufficient", "adequate"]
            if not any(term in text.lower() for term in duration_terms):
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="high",
                    message="Missing duration justification",
                    suggestion="Explain why the trial duration is appropriate for the objectives",
                    category="ich_compliance"
                ))
        
        # Background section specific checks
        elif section_key == "background":
            # Must include disease/condition information
            condition_terms = ["disease", "condition", "disorder", "syndrome", "patients", "prevalence"]
            if not any(term in text.lower() for term in condition_terms):
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="high",
                    message="Missing medical condition description",
                    suggestion="Include information about the medical condition being studied",
                    category="ich_compliance"
                ))
        
        # Primary Objective specific checks
        elif section_key == "primary_objective":
            # Must be specific and measurable
            if len(text.split()) < 30:
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="medium",
                    message="Primary objective too brief",
                    suggestion="Provide a more detailed, specific, and measurable primary objective",
                    category="ich_compliance"
                ))
            
            # Should include population and intervention
            if "participants" not in text.lower() and "patients" not in text.lower() and "subjects" not in text.lower():
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="medium",
                    message="Primary objective should specify target population",
                    suggestion="Clearly identify the target population in the primary objective",
                    category="ich_compliance"
                ))
        
        return issues
    
    def _check_general_ich_requirements(self, text: str) -> List[AnalysisIssue]:
        """Check general ICH M11 compliance requirements"""
        issues = []
        
        # Check for scientific rigor
        if len(text.split()) > 50:  # Only for substantial content
            # Should use precise language
            vague_terms = ["some", "many", "few", "several", "various", "numerous"]
            found_vague = [term for term in vague_terms if f" {term} " in text.lower()]
            if found_vague:
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="low",
                    message=f"Vague language detected: {', '.join(found_vague)}",
                    suggestion="Use more precise, quantitative language where possible",
                    category="scientific_rigor"
                ))
            
            # Should avoid absolute statements without evidence
            absolute_terms = ["always", "never", "all patients", "no patients", "completely", "totally"]
            found_absolute = [term for term in absolute_terms if term in text.lower()]
            if found_absolute:
                issues.append(AnalysisIssue(
                    type="ich_compliance",
                    severity="medium",
                    message=f"Absolute statements detected: {', '.join(found_absolute)}",
                    suggestion="Qualify statements with appropriate evidence or use more measured language",
                    category="scientific_rigor"
                ))
        
        # Check for appropriate clinical terminology
        if "drug" in text.lower() and "investigational medicinal product" not in text.lower() and "IMP" not in text.lower():
            issues.append(AnalysisIssue(
                type="ich_compliance",
                severity="low",
                message="Consider using 'investigational medicinal product' instead of 'drug'",
                suggestion="Use ICH-preferred terminology: 'investigational medicinal product (IMP)'",
                category="terminology"
            ))
        
        return issues
    
    def _calculate_scores(self, issues: List[AnalysisIssue]) -> Dict[str, float]:
        """Calculate quality scores based on issues found"""
        
        # Categorize issues
        grammar_issues = [i for i in issues if i.type == "grammar"]
        consistency_issues = [i for i in issues if i.type in ["consistency", "terminology"]]
        compliance_issues = [i for i in issues if i.type == "ich_compliance"]
        
        # Calculate deductions based on severity
        severity_weights = {"high": 15, "medium": 8, "low": 3}
        
        def calculate_score(issue_list, max_deduction=80):
            total_deduction = 0
            for issue in issue_list:
                total_deduction += severity_weights.get(issue.severity, 5)
            return max(20, 100 - min(total_deduction, max_deduction))
        
        grammar_score = calculate_score(grammar_issues)
        consistency_score = calculate_score(consistency_issues)
        compliance_score = calculate_score(compliance_issues)
        
        # Overall score is weighted average
        overall_score = (grammar_score * 0.3 + consistency_score * 0.3 + compliance_score * 0.4)
        
        return {
            'grammar': grammar_score,
            'consistency': consistency_score,
            'compliance': compliance_score,
            'overall': overall_score
        }
    
    def _generate_summary(self, issues: List[AnalysisIssue], scores: Dict[str, float]) -> str:
        """Generate analysis summary"""
        total_issues = len(issues)
        high_priority = len([i for i in issues if i.severity == "high"])
        
        if total_issues == 0:
            return "✅ Excellent! No significant issues found. The section meets high standards for medical writing and ICH M11 compliance."
        
        overall_score = scores.get('overall', 0)
        
        if overall_score >= 90:
            quality = "Excellent"
        elif overall_score >= 80:
            quality = "Good"
        elif overall_score >= 70:
            quality = "Acceptable"
        else:
            quality = "Needs Improvement"
        
        summary = f"📊 Overall Quality: {quality} ({overall_score:.0f}/100)\n"
        summary += f"🔍 Found {total_issues} total issues"
        
        if high_priority > 0:
            summary += f" ({high_priority} high priority)"
        
        return summary
    
    def _generate_recommendations(self, issues: List[AnalysisIssue], section_name: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Prioritize high-severity issues
        high_issues = [i for i in issues if i.severity == "high"]
        if high_issues:
            recommendations.append(f"🔴 Address {len(high_issues)} high-priority issues first")
        
        # Category-specific recommendations
        issue_categories = {}
        for issue in issues:
            category = issue.category or issue.type
            if category not in issue_categories:
                issue_categories[category] = 0
            issue_categories[category] += 1
        
        if "grammar" in issue_categories:
            recommendations.append("📝 Review medical writing guidelines for grammar and style")
        
        if "consistency" in issue_categories:
            recommendations.append("🔄 Establish consistent terminology and formatting standards")
        
        if "ich_compliance" in issue_categories:
            recommendations.append("📋 Review ICH M11 guidelines for section-specific requirements")
        
        if "regulatory" in issue_categories:
            recommendations.append("⚖️ Ensure compliance with regulatory language requirements")
        
        # General recommendations
        recommendations.append("✅ Consider peer review by senior medical writer")
        
        return recommendations

# Convenience function for easy integration
def analyze_protocol_section(text: str, section_name: str = "", reference_text: str = "") -> Dict:
    """
    Simple function to analyze a protocol section
    Returns a dictionary with analysis results
    """
    analyzer = GrammarConsistencyAnalyzer()
    result = analyzer.analyze_section(text, section_name, reference_text)
    
    return {
        "scores": {
            "grammar": result.grammar_score,
            "consistency": result.consistency_score,
            "compliance": result.compliance_score,
            "overall": result.overall_score
        },
        "issues": [
            {
                "type": issue.type,
                "severity": issue.severity,
                "message": issue.message,
                "suggestion": issue.suggestion,
                "location": issue.location,
                "category": issue.category
            }
            for issue in result.issues
        ],
        "summary": result.summary,
        "recommendations": result.recommendations,
        "total_issues": len(result.issues)
    } 