import json
import random

EVAL_FILE = "E:/RAG_Final_Project/HealthcarePolicyRAG/evaluation/eval_dataset.json"
CHUNKS_FILE = "E:/RAG_Final_Project/HealthcarePolicyRAG/data/chunks/chunks.json"

def main():
    try:
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {EVAL_FILE} not found.")
        return

    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {CHUNKS_FILE} not found.")
        return

    # Create a quick lookup for chunks based on manual, chapter, and section
    chunk_lookup = {}
    for c in chunks_data:
        # we index by a tuple (manual_id, chapter_num)
        key = (str(c.get("manual_id", "")), str(c.get("chapter_num", "")))
        if key not in chunk_lookup:
            chunk_lookup[key] = []
        chunk_lookup[key].append(c)

    print(f"✅ Loaded {len(eval_data)} Q&A pairs from {EVAL_FILE}")
    print(f"✅ Loaded {len(chunks_data)} chunks from {CHUNKS_FILE}")
    print("-" * 80)

    samples = random.sample(eval_data, min(5, len(eval_data)))
    output_lines = []

    for i, pair in enumerate(samples, 1):
        q = pair.get("question", "N/A")
        ans = pair.get("ground_truth_answer", "N/A")
        
        manual = str(pair.get("source_manual", ""))
        chapter = str(pair.get("source_chapter", ""))
        section = str(pair.get("source_section", ""))
        
        output_lines.append(f"**PAIRS: {i}/{len(samples)} (ID: {pair.get('id')})**")
        output_lines.append(f"- **Category:** {pair.get('category')} | **Difficulty:** {pair.get('difficulty')}")
        output_lines.append(f"- **Source:** Manual {manual}, Chapter {chapter}, Section {section}\n")
        
        output_lines.append(f"### ❓ QUESTION:\n> {q}\n")
        output_lines.append(f"### 💡 GROUND TRUTH ANSWER:\n> {ans}\n")
        
        # Find matching chunks
        key = (manual, chapter)
        possible_chunks = chunk_lookup.get(key, [])
        
        matching_chunks = []
        for c in possible_chunks:
            c_sec = str(c.get("section_title", ""))
            ptr_id = c.get("parent_chunk_id", "")
            expected_str = f"_s{section}"
            
            if expected_str in ptr_id or c_sec.startswith(section):
                matching_chunks.append(c)

        if not matching_chunks:
            matching_chunks = [c for c in possible_chunks if section in c.get("parent_chunk_id", "")]

        output_lines.append(f"**📄 SOURCE CHUNKS FOUND:** {len(matching_chunks)}\n")
        
        if matching_chunks:
            for j, mc in enumerate(matching_chunks[:2]):
                text_preview = mc.get("chunk_text", "")
                output_lines.append(f"**Chunk ID:** `{mc.get('chunk_id')}`")
                output_lines.append("```text")
                output_lines.append(text_preview)
                output_lines.append("```\n")
            
            if len(matching_chunks) > 2:
                output_lines.append(f"*... and {len(matching_chunks)-2} more chunks related to this section.*\n")
        else:
            output_lines.append("⚠️ *No perfectly matching chunks found for this reference. Please check the metadata.*\n")
            
        output_lines.append("---\n")
        
    output_lines.append("✅ **Run complete.** Use this output to visually verify that the Ground Truth Answer is fully supported by the Source Chunks.")
    
    with open("E:/RAG_Final_Project/HealthcarePolicyRAG/data/validation_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    print("✅ Results written to data/validation_results.md")

if __name__ == "__main__":
    main()
