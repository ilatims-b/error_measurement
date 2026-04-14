import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import tiktoken
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Prompt templates
EXTRACTION_PROMPT = """
Extract all verifiable factual claims from this text chunk exactly.
Classify each claim strictly as: NUMERIC | ENTITY | DIRECTIONAL | FORWARD.
A NUMERIC claim involves values or statistics (e.g., "$90.1 billion", "1,000 employees").
An ENTITY claim involves named entities and discrete factual relationships (e.g., "Company A acquired Company B").
A DIRECTIONAL claim involves relative changes (e.g., "Operating income increased by 12%").
A FORWARD claim involves expectations or predictions (e.g., "We expect revenue to grow next quarter").

Return ONLY valid JSON in this exact structure: 
[{"subject": "...", "relation": "...", "object": "...", "type": "NUMERIC|ENTITY|DIRECTIONAL|FORWARD"}]
Do NOT return Markdown blocks (like ```json), just the raw JSON text. Do NOT include FORWARD claims in the final JSON array.
"""

VERIFICATION_PROMPT = """
Source document excerpt: {passage}

Claim ({type}): {subject} {relation} {object}

Is this claim supported by the source documentcerpt?
Answer exactly one letter:
(A) Supported
(B) Contradicted 
(C) Not verifiable

Return ONLY the letter (A, B, or C). Do not return anything else.
"""

class FactExtractionPipeline:
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1", model_name: str = "llama-3.3-70b-versatile", extraction_prompt: str = None, verification_prompt: str = None):
        """
        Initializes the pipeline to use an OpenAI-compatible API (e.g. Grok or OpenAI).
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.extraction_prompt = extraction_prompt if extraction_prompt else EXTRACTION_PROMPT
        self.verification_prompt = verification_prompt if verification_prompt else VERIFICATION_PROMPT

    def chunk_text(self, text: str, chunk_size: int = 128) -> List[str]:
        """
        Divide the generated output into non-overlapping windows of chunk_size tokens.
        """
        tokens = self.encoder.encode(text)
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(self.encoder.decode(chunk_tokens))
        return chunks

    def extract_claims(self, text_chunk: str) -> List[Dict]:
        """
        Uses the LLM API to extract claims into JSON.
        """
        raw_response = "N/A"
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.extraction_prompt},
                    {"role": "user", "content": f"Text chunk to extract claims from:\n{text_chunk}"}
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw_response = response.choices[0].message.content.strip()
            
            # Robustly extract JSON array in case Groq returns chatty markdown
            import re
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', raw_response, re.DOTALL)
            if json_match:
                raw_response = json_match.group(0)
            else:
                 # Fallback if the model returned just a single JSON object instead of an array
                 json_obj_match = re.search(r'\{.*?\}', raw_response, re.DOTALL)
                 if json_obj_match:
                     raw_response = f"[{json_obj_match.group(0)}]"

            claims = json.loads(raw_response)
            # Filter just in case the model returned a FORWARD
            return [c for c in claims if c.get("type") in ["NUMERIC", "ENTITY", "DIRECTIONAL"]]
        except Exception as e:
            logger.error(f"Failed to extract claims: {e}. Raw response: {raw_response}")
            return []    

    def verify_claim(self, claim: Dict, source_passage: str) -> str:
        """
        Uses the LLM API to verify a single claim against the source document.
        """
        try:
            prompt = self.verification_prompt.format(
                passage=source_passage,
                type=claim.get("type", "UNKNOWN"),
                subject=claim.get("subject", ""),
                relation=claim.get("relation", ""),
                object=claim.get("object", "")
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=50, # Increased for Groq compatibility if it generates preamble
            )
            answer = response.choices[0].message.content.strip().upper()
            import re
            
            # Normalize response
            match = re.search(r'\b([ABC])\b', answer)
            if match:
                return match.group(1)
            if "SUPPORTED" in answer: return "A"
            if "CONTRADICTED" in answer: return "B"
            return "C"  # Default to not verifiable
        except Exception as e:
            logger.error(f"Failed to verify claim: {e}")
            return "C"

    def process_generation_record(self, record: Dict, source_docs_map: Dict[str, str], chunk_size: int = 128) -> Dict:
        """
        Process a single record containing 3 continuations against the original document.
        """
        doc_id = record.get("document_id")
        source_mda = source_docs_map.get(doc_id, "")
        if not source_mda:
            logger.warning(f"No source text found for {doc_id}. Verification will fail.")
        
        result = record.copy()
        result["evaluation_results"] = []
        
        for cont_idx, continuationtext in enumerate(record.get("continuations", [])):
            chunks = self.chunk_text(continuationtext, chunk_size=chunk_size)
            cont_eval = {"continuation_index": cont_idx, "chunks": []}
            
            # For 4 chunks only (C1 to C4)
            for chunk_idx, chunk_text in enumerate(chunks[:4]):
                claims = self.extract_claims(chunk_text)
                
                # Minimum 3 verifiable claims or mark UNINFORMATIVE
                if len(claims) < 3:
                     cont_eval["chunks"].append({
                         "chunk_index": chunk_idx,
                         "status": "UNINFORMATIVE",
                         "num_claims": len(claims),
                         "claims": claims
                     })
                     continue
                
                chunk_evals = []
                for claim in claims:
                    status = self.verify_claim(claim, source_mda)
                    chunk_evals.append({
                        "claim": claim,
                        "verification_result": status # A, B, or C
                    })
                
                # Compute E_fact(t)
                total = len(chunk_evals)
                errors = sum(1 for e in chunk_evals if e["verification_result"] in ["B", "C"])
                
                cont_eval["chunks"].append({
                    "chunk_index": chunk_idx,
                    "status": "INFORMATIVE",
                    "num_claims": len(claims),
                    "error_rate": errors / total if total > 0 else 0,
                    "evals": chunk_evals
                })
                
            result["evaluation_results"].append(cont_eval)
            
        return result

def main():
    parser = argparse.ArgumentParser(description="Extract and verify factual claims using LLM APIs.")
    parser.add_argument("--gen-file", type=str, default="./data/generations.json", help="Input generations JSON")
    parser.add_argument("--source-file", type=str, default="./data/processed_dataset.json", help="Original source documents JSON")
    parser.add_argument("--output-file", type=str, default="./data/evaluated_generations.json", help="Output verification JSON")
    parser.add_argument("--api-key", type=str, required=True, help="API Key for OpenAI/Grok/Groq")
    parser.add_argument("--base-url", type=str, default="https://api.x.ai/v1", help="API base URL")
    parser.add_argument("--model-name", type=str, default="grok-2-latest", help="Model name for extraction/verification")
    parser.add_argument("--chunk-size", type=int, default=128, help="Token size per evaluation chunk")
    parser.add_argument("--extraction-prompt", type=str, default=None, help="Custom prompt for claim extraction")
    parser.add_argument("--verification-prompt", type=str, default=None, help="Custom prompt for claim verification")
    args = parser.parse_args()

    # Load source map
    source_docs = {}
    with open(args.source_file, "r") as f:
        for d in json.load(f):
            source_docs[d["document_id"]] = d["full_mda_text"]
                
    pipeline = FactExtractionPipeline(
        args.api_key, 
        args.base_url, 
        args.model_name,
        extraction_prompt=args.extraction_prompt,
        verification_prompt=args.verification_prompt
    )
    
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting fact extraction and verification pipeline...")
    all_evals = []
    with open(args.gen_file, "r") as in_f:
        for record in json.load(in_f):
            logger.info(f"Processing document: {record.get('document_id')}")
            eval_record = pipeline.process_generation_record(record, source_docs, chunk_size=args.chunk_size)
            all_evals.append(eval_record)

    with open(args.output_file, "w") as out_f:
        json.dump(all_evals, out_f, indent=2, ensure_ascii=False)

    logger.info(f"Pipeline complete. Output saved to {args.output_file}")

if __name__ == "__main__":
    main()
