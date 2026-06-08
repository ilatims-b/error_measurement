import json
import logging
import re
import random

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise financial assistant. Continue the following financial "
    "document excerpt exactly as it would appear in an SEC filing, without "
    "hallucinating details. DO NOT REPEAT THE PROMPT, CONTINUE DIRECTLY AFTER IT."
)

def generate_continuation_with_injection(client, model_name, prompt, forced_tokens=None, max_tokens=300, system_prompt=None, **kwargs):
    """
    Two-pass generation where forced_tokens are injected in the middle of the generation.
    """
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    
    if not forced_tokens:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Excerpt:\n{prompt}"}
            ],
            max_tokens=max_tokens,
            **kwargs
        )
        return resp.choices[0].message.content

    # Two-pass generation
    resp1 = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Excerpt:\n{prompt}"}
        ],
        max_tokens=max_tokens // 2,
        **kwargs
    )
    partial = resp1.choices[0].message.content
    
    resp2 = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Excerpt:\n{prompt}"},
            {"role": "assistant", "content": partial + "\n" + forced_tokens},
        ],
        max_tokens=max_tokens - (max_tokens // 2),
        **kwargs
    )
    return partial + "\n" + forced_tokens + "\n" + resp2.choices[0].message.content

def generate_continuation_from_partial(client, model_name, prompt, partial_text, forced_tokens, max_tokens=300, system_prompt=None, **kwargs):
    """
    If we already have a partial generation, append forced_tokens and continue.
    """
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    
    injected_content = partial_text + " " + forced_tokens
    
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Excerpt:\n{prompt}"},
            {"role": "assistant", "content": injected_content},
        ],
        max_tokens=max_tokens,
        **kwargs
    )
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
