import argparse
import json
import logging
import time
from collections import deque
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
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1", extraction_prompt: str = None, verification_prompt: str = None, max_passage_tokens: int = 5000, extraction_model_name: str = "llama-4-scout-17b", verification_model_name: str = "qwen3-32b", extraction_tpm: int = 29000, extraction_rpm: int = 28, verification_tpm: int = 5500, verification_rpm: int = 55):
        """
        Initializes the pipeline to use an OpenAI-compatible API (e.g. Grok or OpenAI).
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.extraction_prompt = extraction_prompt if extraction_prompt else EXTRACTION_PROMPT
        self.verification_prompt = verification_prompt if verification_prompt else VERIFICATION_PROMPT
        self.max_passage_tokens = max_passage_tokens
        self.extraction_model_name = extraction_model_name
        self.verification_model_name = verification_model_name
        
        self.max_rpm_verification = verification_rpm
        self.max_tpm_verification = verification_tpm
        self.max_rpm_extraction = extraction_rpm
        self.max_tpm_extraction = extraction_tpm
        
        self.extraction_request_timestamps = deque()
        self.extraction_token_usage = deque()
        
        self.verification_request_timestamps = deque()
        self.verification_token_usage = deque()

    def _wait_for_rate_limit(self, estimated_tokens: int, model_type: str = "extraction"):
        now = time.time()
        
        if model_type == "extraction":
            timestamps = self.extraction_request_timestamps
            tokens = self.extraction_token_usage
            max_rpm = self.max_rpm_extraction
            max_tpm = self.max_tpm_extraction
        else:
            timestamps = self.verification_request_timestamps
            tokens = self.verification_token_usage
            max_rpm = self.max_rpm_verification
            max_tpm = self.max_tpm_verification
            
        # clean old timestamps
        while timestamps and now - timestamps[0] > 60:
            timestamps.popleft()
        while tokens and now - tokens[0][0] > 60:
            tokens.popleft()
            
        current_tokens_in_minute = sum(t[1] for t in tokens)
        
        while len(timestamps) >= max_rpm or (current_tokens_in_minute + estimated_tokens) > max_tpm:
            now = time.time()
            sleep_times = []
            if len(timestamps) >= max_rpm:
                sleep_times.append(60 - (now - timestamps[0]))
            if (current_tokens_in_minute + estimated_tokens) > max_tpm:
                if tokens:
                    sleep_times.append(60 - (now - tokens[0][0]))
                else:
                    break # Single request exceeds TPM limit; must proceed and see if API accepts it
                    
            if not sleep_times:
                break
                
            sleep_duration = max(sleep_times)
            if sleep_duration > 0:
                logger.info(f"Rate limit approaching for {model_type}. Sleeping for {sleep_duration:.2f} seconds...")
                time.sleep(sleep_duration + 0.1) # little extra to be safe
                
            now = time.time()
            # clean again
            while timestamps and now - timestamps[0] > 60:
                timestamps.popleft()
            while tokens and now - tokens[0][0] > 60:
                tokens.popleft()
            current_tokens_in_minute = sum(t[1] for t in tokens)

        timestamps.append(time.time())
        tokens.append((time.time(), estimated_tokens))

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
        prompt_text = f"Text chunk to extract claims from:\n{text_chunk}"
        num_tokens = len(self.encoder.encode(self.extraction_prompt + prompt_text)) + 1024
        self._wait_for_rate_limit(num_tokens, model_type="extraction")
        
        raw_response = "N/A"
        try:
            response = self.client.chat.completions.create(
                model=self.extraction_model_name,
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
            MAX_PASSAGE_TOKENS = self.max_passage_tokens
            passage_tokens = self.encoder.encode(source_passage)
            if len(passage_tokens) > MAX_PASSAGE_TOKENS:
                logger.info(f"Truncating source passage from {len(passage_tokens)} to {MAX_PASSAGE_TOKENS} tokens to respect limits.")
                source_passage = self.encoder.decode(passage_tokens[:MAX_PASSAGE_TOKENS])
                
            prompt = self.verification_prompt.format(
                passage=source_passage,
                type=claim.get("type", "UNKNOWN"),
                subject=claim.get("subject", ""),
                relation=claim.get("relation", ""),
                object=claim.get("object", "")
            )
            
            num_tokens = len(self.encoder.encode(prompt)) + 50
            self._wait_for_rate_limit(num_tokens, model_type="verification")
            
            response = self.client.chat.completions.create(
                model=self.verification_model_name,
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
    parser.add_argument("--verification-model-name", type=str, default="qwen/qwen3-32b", help="Model name for extraction/verification")
    parser.add_argument("--extraction-model-name",type=str, default="meta-llama/llama-4-scout-17b-16e-instruct", help="model for claim extraction")
    parser.add_argument("--chunk-size", type=int, default=128, help="Token size per evaluation chunk")
    parser.add_argument("--extraction-prompt", type=str, default=None, help="Custom prompt for claim extraction")
    parser.add_argument("--verification-prompt", type=str, default=None, help="Custom prompt for claim verification")
    parser.add_argument("--max-passage-tokens", type=int, default=5000, help="Max tokens per passage")
    parser.add_argument("--extraction_tpm", type =int, default=29000)
    parser.add_argument("--extraction_rpm", type =int, default=28)
    parser.add_argument("--verification_tpm", type =int, default=5500)
    parser.add_argument("--verification_rpm", type =int, default=55)
    args = parser.parse_args()

    # Load source map
    source_docs = {}
    with open(args.source_file, "r") as f:
        for d in json.load(f):
            source_docs[d["document_id"]] = d["full_mda_text"]
                
    pipeline = FactExtractionPipeline(
        api_key=args.api_key, 
        base_url=args.base_url, 
        extraction_prompt=args.extraction_prompt,
        verification_prompt=args.verification_prompt,
        max_passage_tokens=args.max_passage_tokens,
        extraction_model_name=args.extraction_model_name,
        verification_model_name=args.verification_model_name,
        extraction_tpm=args.extraction_tpm,
        extraction_rpm=args.extraction_rpm,
        verification_tpm=args.verification_tpm,
        verification_rpm=args.verification_rpm,
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
