import argparse
import json
import logging
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name: str):
    logger.info(f"Loading model: {model_name} on GPU (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    logger.info("Model load complete.")
    return model, tokenizer

def run_generations(input_file: str, output_file: str, model_name: str, num_continuations: int = 3):
    in_path = Path(input_file)
    if not in_path.exists():
        logger.error(f"Input dataset {in_path} not found.")
        return

    documents = []
    with open(in_path, "r") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    
    if not documents:
        logger.warning("No documents found to process.")
        return
        
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_f = open(out_path, "w", encoding="utf-8")
    
    logger.info(f"Starting generation for {len(documents)} documents...")
    
    for doc in documents:
        seed_content = doc.get("seed_prompt", "")
        if not seed_content:
            continue
            
        doc_id = doc.get("document_id", "unknown")
        
        # System instructions to enforce identical generation pattern
        messages = [
            {"role": "system", "content": "You are a precise financial assistant. Continue the following financial document excerpt exactly as it would appear in an SEC filing, without hallucinating details. Do not refer to the prompt."},
            {"role": "user", "content": f"Excerpt:\n{seed_content}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # continuations = []
        # for i in range(num_continuations):
        #     set_seed(42 + i)  # Independent random seed per continuation
        #     logger.debug(f"Generating continuation {i+1} for {doc_id}...")
            
        #     with torch.no_grad():
        #         outputs = model.generate(
        #             **inputs,
        #             max_new_tokens=512,
        #             temperature=0.7,
        #             top_p=0.9,
        #             do_sample=True,
        #             pad_token_id=tokenizer.eos_token_id
        #         )
        logger.debug(f"Batch generating {num_continuations} continuations for {doc_id}...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=num_continuations, # Tells the GPU to batch generate X responses
                pad_token_id=tokenizer.eos_token_id
            )
        
        continuations = []
        # Loop through the batched outputs to decode them
        for i in range(num_continuations):
            # Slice off the prompt to get just the generated tokens
            generated_tokens = outputs[i][inputs.input_ids.shape[1]:]
            continuation_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            continuations.append(continuation_text)  
            # # Slice off the prompt to get just the generated tokens
            # generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            # continuation_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # continuations.append(continuation_text)
            
        # Store result
        result_record = {
            "document_id": doc_id,
            "ticker": doc.get("ticker", ""),
            "seed_prompt": seed_content,
            "continuations": continuations,
            "phi_complexity_score": doc.get("phi_complexity_score", 0.0)
        }
        out_f.write(json.dumps(result_record) + "\n")
        out_f.flush()
        
        logger.info(f"Completed {num_continuations} generations for {doc_id}.")
        
    out_f.close()
    logger.info(f"All generations finished. Results saved to {output_file}.")

def main():
    parser = argparse.ArgumentParser(description="Generate 512-token continuations using Meta-Llama-3-8B-Instruct")
    parser.add_argument("--input-file", type=str, default="./data/processed_dataset.jsonl", help="Path to input dataset JSONL")
    parser.add_argument("--output-file", type=str, default="./data/generations.jsonl", help="Path to output generations JSONL")
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model string")
    parser.add_argument("--num-continuations", type=int, default=3, help="Number of independent generations per document")
    args = parser.parse_args()
    
    run_generations(args.input_file, args.output_file, args.model_name, args.num_continuations)

if __name__ == "__main__":
    main()
