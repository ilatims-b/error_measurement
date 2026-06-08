import json
import logging
import re
import random
import time
from collections import deque
from typing import List, Optional
import openai

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Sliding-window rate limiter tracking requests-per-minute and
    tokens-per-minute to avoid hitting API limits.
    """
    def __init__(self, max_rpm: int, max_tpm: int):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self._request_timestamps = deque()
        self._token_usage = deque()  # (timestamp, token_count)

    def _cleanup(self):
        now = time.time()
        while self._request_timestamps and now - self._request_timestamps[0] > 60:
            self._request_timestamps.popleft()
        while self._token_usage and now - self._token_usage[0][0] > 60:
            self._token_usage.popleft()

    def wait(self, estimated_tokens: int):
        self._cleanup()
        current_tpm = sum(t[1] for t in self._token_usage)

        while (
            len(self._request_timestamps) >= self.max_rpm
            or (current_tpm + estimated_tokens) > self.max_tpm
        ):
            now = time.time()
            sleep_times = []

            if len(self._request_timestamps) >= self.max_rpm:
                sleep_times.append(60.0 - (now - self._request_timestamps[0]))

            if (current_tpm + estimated_tokens) > self.max_tpm:
                if self._token_usage:
                    sleep_times.append(60.0 - (now - self._token_usage[0][0]))
                else:
                    break

            if not sleep_times:
                break

            sleep_duration = max(sleep_times)
            if sleep_duration > 0:
                logger.info(f"Rate limit approaching. Sleeping for {sleep_duration:.2f}s "
                            f"(rpm={len(self._request_timestamps)}/{self.max_rpm}, "
                            f"tpm={current_tpm}/{self.max_tpm})...")
                time.sleep(sleep_duration + 0.1)

            self._cleanup()
            current_tpm = sum(t[1] for t in self._token_usage)

        ts = time.time()
        self._request_timestamps.append(ts)
        self._token_usage.append((ts, estimated_tokens))

    def record_actual_tokens(self, actual_tokens: int):
        if self._token_usage:
            ts, _ = self._token_usage[-1]
            self._token_usage[-1] = (ts, actual_tokens)


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise financial assistant. Continue the following financial "
    "document excerpt exactly as it would appear in an SEC filing, without "
    "hallucinating details. DO NOT REPEAT THE PROMPT, CONTINUE DIRECTLY AFTER IT."
)

def _call_with_retries(client, rate_limiter, est_tokens, **kwargs):
    max_retries = 10
    for attempt in range(max_retries):
        if rate_limiter:
            rate_limiter.wait(est_tokens)
        try:
            resp = client.chat.completions.create(**kwargs)
            if rate_limiter and resp.usage:
                rate_limiter.record_actual_tokens(resp.usage.total_tokens)
            return resp
        except openai.RateLimitError as e:
            msg = str(e)
            logger.warning(f"Rate limit hit. Error: {msg}")
            
            # Try to parse "Please try again in XmYs."
            match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", msg)
            wait_time = 60 # default 1 minute
            if match:
                h = int(match.group(1)) if match.group(1) else 0
                m = int(match.group(2)) if match.group(2) else 0
                s = float(match.group(3)) if match.group(3) else 0.0
                wait_time = h*3600 + m*60 + s + 5.0 # 5 seconds buffer
            
            if attempt < max_retries - 1:
                logger.info(f"Sleeping for {wait_time:.2f} seconds before retrying...")
                print(f"\nRATE LIMIT (TPD/RPM) HIT! Sleeping for {wait_time/60:.2f} minutes to cool down...")
                time.sleep(wait_time)
            else:
                raise


def generate_continuation_with_injection(client, model_name, prompt, forced_tokens=None, max_tokens=300, system_prompt=None, rate_limiter=None, **kwargs):
    """
    Two-pass generation where forced_tokens are injected in the middle of the generation.
    """
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    
    est_tokens = (len(sys_prompt) + len(prompt)) // 4 + max_tokens
    
    if not forced_tokens:
        if rate_limiter: rate_limiter.wait(est_tokens)
        resp = _call_with_retries(client, rate_limiter, est_tokens,
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Excerpt:\n{prompt}"}
            ],
            max_tokens=max_tokens,
            **kwargs
        )
        if rate_limiter and resp.usage: rate_limiter.record_actual_tokens(resp.usage.total_tokens)
        return resp.choices[0].message.content

    # Two-pass generation
    if rate_limiter: rate_limiter.wait(est_tokens // 2)
    resp1 = _call_with_retries(client, rate_limiter, est_tokens // 2,
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Excerpt:\n{prompt}"}
        ],
        max_tokens=max_tokens // 2,
        **kwargs
    )
    if rate_limiter and resp1.usage: rate_limiter.record_actual_tokens(resp1.usage.total_tokens)
    partial = resp1.choices[0].message.content
    
    if rate_limiter: rate_limiter.wait(est_tokens // 2)
    resp2 = _call_with_retries(client, rate_limiter, est_tokens // 2,
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Excerpt:\n{prompt}"},
            {"role": "assistant", "content": partial + "\n" + forced_tokens},
        ],
        max_tokens=max_tokens - (max_tokens // 2),
        **kwargs
    )
    if rate_limiter and resp2.usage: rate_limiter.record_actual_tokens(resp2.usage.total_tokens)
    return partial + "\n" + forced_tokens + "\n" + resp2.choices[0].message.content

def generate_continuation_from_partial(client, model_name, prompt, partial_text, forced_tokens, max_tokens=300, system_prompt=None, rate_limiter=None, **kwargs):
    """
    If we already have a partial generation, append forced_tokens and continue.
    """
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    
    injected_content = partial_text + " " + forced_tokens
    
    est_tokens = (len(sys_prompt) + len(prompt) + len(injected_content)) // 4 + max_tokens
    if rate_limiter: rate_limiter.wait(est_tokens)
    
    resp = _call_with_retries(client, rate_limiter, est_tokens,
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Excerpt:\n{prompt}"},
            {"role": "assistant", "content": injected_content},
        ],
        max_tokens=max_tokens,
        **kwargs
    )
    if rate_limiter and resp.usage: rate_limiter.record_actual_tokens(resp.usage.total_tokens)
    return injected_content + " " + resp.choices[0].message.content

def perturb_numeric_fact(claim_object, perturbation_factor=1.5, pre_text="", num_facts=1):
    """
    Given a claim string containing a number, perturb it by multiplying by perturbation_factor.
    num_facts determines how many numbers to perturb in the claim (if multiple exist).
    pre_text can be used to add context like "On the other hand, "
    """
    # Simple regex to find numbers that might have commas and decimals
    # Matches patterns like $6.475 or 10.4% or 1,060
    pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
    
    def replace_num(match):
        val_str = match.group(1).replace(',', '')
        try:
            val = float(val_str)
            new_val = val * perturbation_factor
            # Formatting back
            if '.' in match.group(1):
                decimals = len(match.group(1).split('.')[1])
                return f"{new_val:,.{decimals}f}"
            else:
                return f"{int(new_val):,}"
        except ValueError:
            return match.group(1)

    matches = list(re.finditer(pattern, claim_object))
    if not matches:
        return (pre_text + " " + claim_object).strip()

    # Create a new string with replacements
    offset = 0
    perturbed_claim = claim_object
    for i, match in enumerate(matches):
        if i >= num_facts:
            break
        replacement = replace_num(match)
        start = match.start() + offset
        end = match.end() + offset
        perturbed_claim = perturbed_claim[:start] + replacement + perturbed_claim[end:]
        offset += len(replacement) - (end - start)
        
    return (pre_text + " " + perturbed_claim).strip()


def perturb_directional_fact(claim_object, pre_text="", num_facts=1):
    """
    Given a claim, flip directional words (increased -> decreased, higher -> lower, etc.)
    """
    flips = {
        "increased": "decreased", "decreased": "increased",
        "increase": "decrease", "decrease": "increase",
        "higher": "lower", "lower": "higher",
        "gained": "lost", "lost": "gained",
        "up": "down", "down": "up"
    }
    
    words = claim_object.split()
    flipped_count = 0
    for i, w in enumerate(words):
        clean_w = w.lower().strip(".,;:()")
        if clean_w in flips and flipped_count < num_facts:
            replacement = flips[clean_w]
            if w.istitle():
                replacement = replacement.capitalize()
            words[i] = w.replace(clean_w, replacement).replace(clean_w.capitalize(), replacement.capitalize())
            flipped_count += 1
            
    return (pre_text + " " + " ".join(words)).strip()

def extract_mda_interval_text(mda_text, start_idx, length=50, interval_words=100, count=1):
    """
    Extracts texts from full MDA text at certain intervals to inject.
    mda_text: the full MDA string
    start_idx: the word index to start the first extraction
    length: how many words to extract per interval
    interval_words: how many words to skip between extractions
    count: how many intervals to extract (usually 1 initially)
    """
    words = mda_text.split()
    extractions = []
    
    current_idx = start_idx
    for _ in range(count):
        if current_idx >= len(words):
            break
        end_idx = min(current_idx + length, len(words))
        extractions.append(" ".join(words[current_idx:end_idx]))
        current_idx = end_idx + interval_words
        
    return extractions


def locate_fact_in_generation(generation_text, claim_dict):
    """
    Attempts to locate the claim in the generation_text.
    Returns (partial_text, match_str) where partial_text is the text up to the matched claim part,
    and match_str is the exact substring found that should be perturbed.
    If not found, returns (None, None).
    """
    ctype = claim_dict.get("type", "")
    
    if ctype == "NUMERIC":
        obj = claim_dict.get("object", "")
        # Try exact object match
        idx = generation_text.find(obj)
        if idx != -1:
            return generation_text[:idx], obj
            
        # Try finding the number inside the object
        import re
        numbers = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', obj)
        if numbers:
            for num in numbers:
                idx = generation_text.find(num)
                if idx != -1:
                    return generation_text[:idx], num
                    
    elif ctype == "DIRECTIONAL":
        rel = claim_dict.get("relation", "")
        idx = generation_text.find(rel)
        if idx != -1 and len(rel) > 3:
            return generation_text[:idx], rel
            
        subj = claim_dict.get("subject", "")
        idx = generation_text.find(subj)
        if idx != -1 and len(subj) > 3:
            return generation_text[:idx], subj
            
        obj = claim_dict.get("object", "")
        idx = generation_text.find(obj)
        if idx != -1 and len(obj) > 3:
            return generation_text[:idx], obj

    return None, None
