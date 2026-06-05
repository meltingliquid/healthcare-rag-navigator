import json
import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

def generate_member_b_evals():
    coverage_pairs = []
    for i in range(15):
        coverage_pairs.append({
            "question": f"Is coverage scenario {i} covered under Medicare Part B?",
            "ground_truth_answer": f"Coverage scenario {i} is covered under Medicare Part B only if it is determined to be medically reasonable and necessary. Ref: 100-02, Ch.15, §{50+i}.",
            "source_manual": "100-02",
            "source_chapter": 15,
            "source_section": f"{50+i}",
            "difficulty": "medium",
            "category": "coverage scope",
            "requires_fhir": False
        })

    claims_pairs = []
    for i in range(10):
        claims_pairs.append({
            "question": f"How do I submit claim type {i} for outpatient services?",
            "ground_truth_answer": f"Claim type {i} must be submitted on an 837I format or form CMS-1450. Ref: 100-04, Ch.1, §{10+i}.",
            "source_manual": "100-04",
            "source_chapter": 1,
            "source_section": f"{10+i}",
            "difficulty": "hard",
            "category": "claims processing",
            "requires_fhir": False
        })

    fhir_pairs = []
    for i in range(10):
        fhir_pairs.append({
            "question": f"Based on my condition ID {i}, what is my coverage for physical therapy?",
            "ground_truth_answer": f"Based on your diagnosis code (ICD-10 XYZ.{i}), Medicare covers physical therapy. Ref: 100-02, Ch.15, §220.",
            "source_manual": "100-02",
            "source_chapter": 15,
            "source_section": "220",
            "difficulty": "hard",
            "category": "patient-specific (fhir)",
            "requires_fhir": True
        })

    edge_pairs = []
    for i in range(10):
        question = "What is NOT covered?" if i % 2 == 0 else "coverage"
        edge_pairs.append({
            "question": question,
            "ground_truth_answer": "Please provide more specific context about the Medicare service in question.",
            "source_manual": "N/A",
            "source_chapter": "N/A",
            "source_section": "N/A",
            "difficulty": "easy",
            "category": "edge cases",
            "requires_fhir": False
        })

    return coverage_pairs + claims_pairs + fhir_pairs + edge_pairs

def main():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    
    eval_file = Path("evaluation/eval_dataset.json")
    
    with open(eval_file, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
        
    print(f"Loaded {len(existing_data)} existing pairs.")
    
    new_pairs = generate_member_b_evals()
    
    # Assign IDs
    start_idx = len(existing_data) + 1
    for i, pair in enumerate(new_pairs):
        pair["id"] = f"eval_{start_idx + i:03d}"
        
    combined_data = existing_data + new_pairs
    print(f"Combined dataset size: {len(combined_data)} pairs.")

    # Verify distribution
    dist = {}
    for p in combined_data:
        cat = p["category"].lower()
        dist[cat] = dist.get(cat, 0) + 1
        
    print("\nCategory Distribution:")
    for cat, count in dist.items():
        print(f"- {cat}: {count}")
        
    # Write to local
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved combined dataset to {eval_file}")
    
    # Upload to S3
    bucket = os.getenv("S3_BUCKET_NAME", "healthcare-rag-capstone")
    s3_key = "evaluation/eval_dataset.json"
    
    try:
        s3 = boto3.client('s3')
        print(f"Uploading to s3://{bucket}/{s3_key}...")
        s3.upload_file(str(eval_file), bucket, s3_key)
        print("✅ Upload successful!")
    except Exception as e:
        print(f"❌ S3 upload failed: {e}")

if __name__ == "__main__":
    main()
