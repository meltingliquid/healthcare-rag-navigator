import re
import spacy
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

try:
    analyzer = AnalyzerEngine(supported_languages=["en"])
except Exception:
    analyzer = AnalyzerEngine()

anonymizer = AnonymizerEngine()

# Custom Recognizer for Medical Record Number (MRN)
mrn_pattern = Pattern(name="mrn_pattern", regex=r"\b(?:[A-Z]{1,2}-?\d{4,9}|\d{5,10}|[A-Z]{2}\d{4,6})\b", score=0.6)
mrn_recognizer = PatternRecognizer(
    supported_entity="MEDICAL_RECORD_NUMBER",
    patterns=[mrn_pattern],
    context=["mrn", "medical record", "record number", "patient id", "id"]
)
analyzer.registry.add_recognizer(mrn_recognizer)

# Custom Recognizer for Insurance Plan ID
insurance_pattern = Pattern(name="insurance_pattern", regex=r"\b[A-Z]+-?\d{4,10}\b", score=0.6)
insurance_recognizer = PatternRecognizer(
    supported_entity="INSURANCE_PLAN_ID",
    patterns=[insurance_pattern],
    context=["insurance", "plan", "policy", "member", "group"]
)
analyzer.registry.add_recognizer(insurance_recognizer)

# Simple fallback for SSN since Presidio's built-in validates strict checksums (failing on mock data)
ssn_pattern = Pattern(name="ssn_pattern", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.6)
ssn_recognizer = PatternRecognizer(
    supported_entity="US_SSN",
    patterns=[ssn_pattern],
    context=["ssn", "social security"]
)
analyzer.registry.add_recognizer(ssn_recognizer)

# Simple recognizer for DOB (format MM/DD/YYYY or similar) to avoid matching words like "today"
dob_pattern = Pattern(name="dob_pattern", regex=r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", score=0.6)
dob_recognizer = PatternRecognizer(
    supported_entity="DATE_OF_BIRTH",
    patterns=[dob_pattern],
    context=["dob", "born", "birth", "date of birth"]
)
analyzer.registry.add_recognizer(dob_recognizer)

def mask_phi(text: str) -> tuple[str, dict]:
    """
    Mask PHI in text. Returns a tuple of (masked_text, deanon_map).
    The deanon_map allows restoring original terms if necessary (in-memory only).
    """
    entities_to_mask = [
        "PERSON", 
        "PHONE_NUMBER", 
        "EMAIL_ADDRESS", 
        "US_SSN", 
        "DATE_OF_BIRTH", 
        "LOCATION", 
        "MEDICAL_RECORD_NUMBER",
        "INSURANCE_PLAN_ID"
    ]
    
    results = analyzer.analyze(
        text=text,
        entities=entities_to_mask,
        language="en"
    )
    
    # Sort results by start index to avoid overlapping entity replacements messing up the process.
    # Actually presidio_analyzer/anonymizer handles this, but let's filter overlaps.
    filtered_results = []
    # If multiple recognizers match the same span, keep the highest score.
    sorted_results = sorted(results, key=lambda x: (x.start, -x.score))
    
    for res in sorted_results:
        # Check for overlap
        overlap = False
        for kept in filtered_results:
            if max(res.start, kept.start) < min(res.end, kept.end):
                overlap = True
                break
        if not overlap:
            filtered_results.append(res)
            
    operators = {}
    for ent in entities_to_mask:
        operators[ent] = OperatorConfig("replace", {"new_value": f"<{ent}>"})
        
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=filtered_results,
        operators=operators
    )
    
    masked_text = anonymized_result.text
    
    deanon_map = {}
    for result in filtered_results:
        original_text = text[result.start:result.end]
        replacement_tag = f"<{result.entity_type}>"
        deanon_map[replacement_tag] = original_text
        
    return masked_text, deanon_map


# ==============================================================================
# Testing Block (Comment out later)
# ==============================================================================
if __name__ == "__main__":
    print("--- Running Presidio PHI Masker Tests ---")
    
    test_1 = "John Smith, SSN 123-45-6789, born on 05/12/1950, called from New York. His MRN is XJ9921."
    masked_1, map_1 = mask_phi(test_1)
    print(f"\nOriginal: {test_1}")
    print(f"Masked  : {masked_1}")
    print(f"Map     : {map_1}")

    test_2 = "Patient has diabetes and takes Metformin daily. Her insurance plan ID is AETNA-12345."
    masked_2, map_2 = mask_phi(test_2)
    print(f"\nOriginal: {test_2}")
    print(f"Masked  : {masked_2}")
    print(f"Map     : {map_2}")
    
    test_3 = "Check coverage for age range 65-70 with section 40.1 of manual 100-02."
    masked_3, map_3 = mask_phi(test_3)
    print(f"\nOriginal: {test_3}")
    print(f"Masked  : {masked_3}")
    print(f"Map     : {map_3}")
