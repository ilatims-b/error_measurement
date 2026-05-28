import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default system prompt (same intent as the local-model path)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise financial assistant. Continue the following financial "
    "document excerpt exactly as it would appear in an SEC filing, without "
    "hallucinating details. DO NOT REPEAT THE PROMPT, CONTINUE DIRECTLY AFTER IT."
)

# ===========================================================================
# LOCAL HUGGINGFACE PATH (unchanged)
# ===========================================================================

def load_model_and_tokenizer(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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


def run_generations(
    input_file: str,
    output_file: str,
    model_name: str,
    num_continuations: int = 3,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate continuations using a local HuggingFace model."""
    import torch

    in_path = Path(input_file)
    if not in_path.exists():
        logger.error(f"Input dataset {in_path} not found.")
        return

    with open(in_path, "r") as f:
        documents = json.load(f)

    if not documents:
        logger.warning("No documents found to process.")
        return

    model, tokenizer = load_model_and_tokenizer(model_name)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    logger.info(f"Starting generation for {len(documents)} documents...")

    for doc in documents:
        print(f"on doc{doc['document_id']}")
        seed_content = doc.get("seed_prompt", "")
        if not seed_content:
            continue

        doc_id = doc.get("document_id", "unknown")

        # System instructions to enforce identical generation pattern
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpt:\n{seed_content}"}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        logger.debug(f"Batch generating {num_continuations} continuations for {doc_id}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_continuations,  # Tells the GPU to batch generate X responses
                pad_token_id=tokenizer.eos_token_id
            )

        continuations = []
        # Loop through the batched outputs to decode them
        for i in range(num_continuations):
            # Slice off the prompt to get just the generated tokens
            generated_tokens = outputs[i][inputs.input_ids.shape[1]:]
            continuation_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            continuations.append(continuation_text)

        # Store result
        result_record = {
            "document_id": doc_id,
            "ticker": doc.get("ticker", ""),
            "seed_prompt": seed_content,
            "continuations": continuations,
            "phi_complexity_score": doc.get("phi_complexity_score", 0.0)
        }
        all_results.append(result_record)

        logger.info(f"Completed {num_continuations} generations for {doc_id}.")

    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(all_results, out_f, indent=2, ensure_ascii=False)
    logger.info(f"All generations finished. Results saved to {output_file}.")


# ===========================================================================
# API PATH — Grok (xAI) or any OpenAI-compatible endpoint
# ===========================================================================

class _RateLimiter:
    """
    Sliding-window rate limiter tracking requests-per-minute and
    tokens-per-minute, mirroring the approach in fact_pipeline.py.
    """

    def __init__(self, max_rpm: int, max_tpm: int):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self._request_timestamps: deque = deque()
        self._token_usage: deque = deque()  # (timestamp, token_count)

    def _cleanup(self):
        now = time.time()
        while self._request_timestamps and now - self._request_timestamps[0] > 60:
            self._request_timestamps.popleft()
        while self._token_usage and now - self._token_usage[0][0] > 60:
            self._token_usage.popleft()

    def wait(self, estimated_tokens: int):
        """Block until this request can be safely made within rate limits."""
        self._cleanup()
        current_tpm = sum(t[1] for t in self._token_usage)

        while (
            len(self._request_timestamps) >= self.max_rpm
            or (current_tpm + estimated_tokens) > self.max_tpm
        ):
            now = time.time()
            sleep_times: List[float] = []

            if len(self._request_timestamps) >= self.max_rpm:
                sleep_times.append(60.0 - (now - self._request_timestamps[0]))

            if (current_tpm + estimated_tokens) > self.max_tpm:
                if self._token_usage:
                    sleep_times.append(60.0 - (now - self._token_usage[0][0]))
                else:
                    break  # Single request exceeds TPM; must attempt anyway

            if not sleep_times:
                break

            sleep_duration = max(sleep_times)
            if sleep_duration > 0:
                logger.info(
                    f"Rate limit approaching. Sleeping for {sleep_duration:.2f}s "
                    f"(rpm={len(self._request_timestamps)}/{self.max_rpm}, "
                    f"tpm={current_tpm}/{self.max_tpm})..."
                )
                time.sleep(sleep_duration + 0.1)

            self._cleanup()
            current_tpm = sum(t[1] for t in self._token_usage)

        ts = time.time()
        self._request_timestamps.append(ts)
        self._token_usage.append((ts, estimated_tokens))

    def record_actual_tokens(self, actual_tokens: int):
        """
        After a call completes, replace the estimated token entry with the
        actual usage so subsequent waits are accurate.
        """
        if self._token_usage:
            ts, _ = self._token_usage[-1]
            self._token_usage[-1] = (ts, actual_tokens)


def run_generations_api(
    input_file: str,
    output_file: str,
    model_name: str,
    api_key: str,
    base_url: str = "https://api.x.ai/v1",
    num_continuations: int = 3,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    rpm: int = 60,
    tpm: int = 100_000,
    system_prompt: Optional[str] = None,
):
    """
    Generate continuations using a Grok (xAI) or any OpenAI-compatible API.

    Rate limiting is handled by a sliding-window deque that tracks both
    requests-per-minute (RPM) and tokens-per-minute (TPM). Each document
    makes `num_continuations` separate API calls so that different random
    seeds produce diverse outputs (the chat API does not expose
    num_return_sequences).
    """
    from openai import OpenAI

    in_path = Path(input_file)
    if not in_path.exists():
        logger.error(f"Input dataset {in_path} not found.")
        return

    with open(in_path, "r") as f:
        documents = json.load(f)

    if not documents:
        logger.warning("No documents found to process.")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    rate_limiter = _RateLimiter(max_rpm=rpm, max_tpm=tpm)

    sys_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    logger.info(
        f"Starting API generation for {len(documents)} documents "
        f"using model '{model_name}' at {base_url} "
        f"(RPM={rpm}, TPM={tpm})..."
    )

    for doc in documents:
        doc_id = doc.get("document_id", "unknown")
        seed_content = doc.get("seed_prompt", "")
        if not seed_content:
            logger.warning(f"Skipping doc {doc_id}: no seed_prompt.")
            continue

        logger.info(f"Processing doc: {doc_id}")

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Excerpt:\n{seed_content}"},
        ]

        # Rough token estimate for rate-limit pre-check
        # (system + user text ≈ chars/4, plus output budget)
        estimated_prompt_tokens = (len(sys_prompt) + len(seed_content)) // 4
        estimated_total = estimated_prompt_tokens + max_new_tokens

        continuations = []
        for i in range(num_continuations):
            rate_limiter.wait(estimated_total)

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                continuation_text = response.choices[0].message.content.strip()
                continuations.append(continuation_text)

                # Update rate limiter with actual token usage if available
                if response.usage:
                    actual_tokens = response.usage.total_tokens
                    rate_limiter.record_actual_tokens(actual_tokens)
                    logger.debug(
                        f"  continuation {i+1}/{num_continuations} done "
                        f"(actual tokens: {actual_tokens})"
                    )
                else:
                    logger.debug(f"  continuation {i+1}/{num_continuations} done")

            except Exception as e:
                logger.error(
                    f"API call failed for doc {doc_id}, continuation {i+1}: {e}"
                )
                continuations.append("")  # Keep slot so indices stay consistent

        result_record = {
            "document_id": doc_id,
            "ticker": doc.get("ticker", ""),
            "seed_prompt": seed_content,
            "continuations": continuations,
            "phi_complexity_score": doc.get("phi_complexity_score", 0.0),
            "generation_backend": "api",
            "generation_model": model_name,
        }
        all_results.append(result_record)
        logger.info(f"Completed {num_continuations} API generations for {doc_id}.")

    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(all_results, out_f, indent=2, ensure_ascii=False)
    logger.info(f"All API generations finished. Results saved to {output_file}.")


# ===========================================================================
# CLI ENTRYPOINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate text continuations using either a local HuggingFace model "
            "or a Grok/OpenAI-compatible API (pass --api-key to use the API path)."
        )
    )
    # --- shared args ---
    parser.add_argument("--input-file", type=str, default="./data/processed_dataset.json",
                        help="Path to input dataset JSON")
    parser.add_argument("--output-file", type=str, default="./data/generations.json",
                        help="Path to output generations JSON")
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace model string (local path) OR API model name (e.g. grok-3-mini)")
    parser.add_argument("--num-continuations", type=int, default=3,
                        help="Number of independent generations per document")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Generation top_p sampling parameter")

    # --- API-specific args ---
    api_group = parser.add_argument_group(
        "API backend (Grok / OpenAI-compatible)",
        "Pass --api-key to activate the API path instead of local HuggingFace inference."
    )
    api_group.add_argument("--api-key", type=str, default=None,
                           help="API key for Grok/OpenAI-compatible service. "
                                "If set, uses the API backend instead of local model.")
    api_group.add_argument("--base-url", type=str, default="https://api.x.ai/v1",
                           help="Base URL of the OpenAI-compatible API (default: Grok/xAI)")
    api_group.add_argument("--rpm", type=int, default=60,
                           help="Max requests per minute (API rate limit)")
    api_group.add_argument("--tpm", type=int, default=100_000,
                           help="Max tokens per minute (API rate limit)")
    api_group.add_argument("--system-prompt", type=str, default=None,
                           help="Override the default system prompt for API calls")

    args = parser.parse_args()

    if args.api_key:
        # ---- Grok / API path ----
        logger.info("API key provided — using API backend (Grok/OpenAI-compatible).")
        run_generations_api(
            input_file=args.input_file,
            output_file=args.output_file,
            model_name=args.model_name if args.model_name != "meta-llama/Meta-Llama-3-8B-Instruct" else "grok-3-mini",
            api_key=args.api_key,
            base_url=args.base_url,
            num_continuations=args.num_continuations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            rpm=args.rpm,
            tpm=args.tpm,
            system_prompt=args.system_prompt,
        )
    else:
        # ---- Local HuggingFace path ----
        logger.info("No API key provided — using local HuggingFace model.")
        run_generations(
            args.input_file,
            args.output_file,
            args.model_name,
            args.num_continuations,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )


if __name__ == "__main__":
    main()
