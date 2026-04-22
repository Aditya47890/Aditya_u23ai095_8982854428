"""
RLVER Profile Translator: Chinese → English
=============================================
This script translates all Chinese text in the RLVER training/test profile JSONL files
to English using an LLM API (OpenAI-compatible).

Usage:
    1. Set your API key and base URL below
    2. Run: python translate_profiles.py
    
The script:
    - Reads each line from the input JSONL file
    - Translates the Chinese fields (player, scene, main_cha, cha_group, task, topic) to English
    - Preserves the JSON structure and id field
    - Saves the translated output to a new file
    - Supports resuming from where it left off if interrupted
    
Supports: OpenAI API, Gemini API, any OpenAI-compatible endpoint, or local Ollama
"""

import json
import os
import time
import argparse
from pathlib import Path

# ============================================================
# CONFIGURATION — CHOOSE ONE OF THESE API OPTIONS
# ============================================================

# Option 1: OpenAI API
# API_KEY = "sk-your-openai-key-here"
# BASE_URL = "https://api.openai.com/v1"
# MODEL = "gpt-4o-mini"  # Cheap and good enough for translation

# Option 2: Google Gemini (via OpenAI-compatible endpoint)
# API_KEY = "your-gemini-key-here"
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# MODEL = "gemini-2.0-flash"

# Option 3: Local Ollama (FREE — no API key needed)
# API_KEY = "ollama"
# BASE_URL = "http://localhost:11434/v1"
# MODEL = "qwen2.5:7b"  # or llama3, gemma2, etc.

# Option 4: Groq (very fast, free tier available)
# API_KEY = "your-groq-key"
# BASE_URL = "https://api.groq.com/openai/v1"
# MODEL = "llama-3.3-70b-versatile"

# Default: OpenAI
API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("TRANSLATION_MODEL", "gpt-4o-mini")

# ============================================================

SYSTEM_PROMPT = """You are a professional Chinese-to-English translator specializing in character profiles for emotional dialogue research. 

Translate the following Chinese text to natural, fluent English. Follow these rules:
1. Preserve ALL formatting, special markers, markdown, and structural elements exactly as they are.
2. Character names should be transliterated (e.g., 小明 → Xiao Ming) NOT translated literally.
3. Cultural references should be adapted to be understandable in English while keeping the emotional essence.
4. Emotional vocabulary must be precise — this is for an empathy research system.
5. Keep section headers, bullet points, and numbered lists in the same format.
6. Do NOT add explanations or notes. Output ONLY the translated text.
7. The translation should feel natural, as if it were originally written in English.
"""


# Global delay between API calls (seconds). Adjusted via --delay CLI arg.
INTER_CALL_DELAY = 5  # Default: 5 seconds between profiles


def _parse_retry_delay(error_str: str) -> float:
    """Extract the retry delay from an API error message (e.g. 'retry in 36.2s')."""
    import re
    # Match patterns like: "retry in 36.204s", "retryDelay: 36s", "Please retry in 36s"
    m = re.search(r'retry\s*(?:in|Delay["\']?\s*[:=]\s*["\']?)\s*([\d.]+)\s*s', error_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2  # Add 2s buffer
    return 0


def api_call_with_retry(client, messages, max_retries=8):
    """Make API call with smart retry: parses the retry delay from 429 errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=6000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            is_rate_limit = any(x in error_lower for x in ['429', 'rate', 'limit', 'quota', 'resource_exhausted'])
            
            if is_rate_limit:
                # Try to parse the exact retry delay from the error message
                parsed_delay = _parse_retry_delay(error_str)
                if parsed_delay > 0:
                    wait_time = min(parsed_delay, 180)
                else:
                    wait_time = min(2 ** (attempt + 1) * 10, 180)  # 20s, 40s, 80s, 160s, 180s
                
                print(f"\n    ⏳ Rate limited! Waiting {wait_time:.0f}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                print(f"\n    ⚠ Error: {e}. Retrying in 10s...", flush=True)
                time.sleep(10)
            else:
                raise e
    raise Exception(f"Failed after {max_retries} retries")


def translate_profile(profile: dict, client, profile_idx: int) -> dict:
    """Translate all Chinese fields in a single API call to minimize rate limit hits."""
    translated = profile.copy()
    
    # Check if already translated
    cn_count = sum(1 for c in json.dumps(profile, ensure_ascii=False) if '\u4e00' <= c <= '\u9fff')
    if cn_count < 5:
        print("  Already in English, skipping.")
        return translated
    
    # Build a single prompt with ALL fields to translate (1 API call instead of 6)
    fields_to_translate = {}
    for field in ['player', 'scene', 'task', 'main_cha', 'first_talk']:
        if field in profile and profile[field]:
            cn = sum(1 for c in str(profile[field]) if '\u4e00' <= c <= '\u9fff')
            if cn > 0:
                fields_to_translate[field] = str(profile[field])
    
    # Handle cha_group separately
    if 'cha_group' in profile and isinstance(profile['cha_group'], list):
        cha_str = ', '.join(profile['cha_group'])
        cn = sum(1 for c in cha_str if '\u4e00' <= c <= '\u9fff')
        if cn > 0:
            fields_to_translate['cha_group'] = cha_str
    
    # Handle topic if present
    if 'topic' in profile and profile['topic']:
        cn = sum(1 for c in str(profile['topic']) if '\u4e00' <= c <= '\u9fff')
        if cn > 0:
            fields_to_translate['topic'] = str(profile['topic'])
    
    if not fields_to_translate:
        print("  No Chinese fields found, skipping.")
        return translated
    
    # Build combined prompt
    combined_prompt = "Translate each section below from Chinese to English. Keep the exact same section headers.\n\n"
    for field, value in fields_to_translate.items():
        combined_prompt += f"=== {field} ===\n{value}\n\n"
    combined_prompt += "Output the translated text with the same === field === headers. Do NOT add any notes or explanations."
    
    print(f"  Translating {len(fields_to_translate)} fields in one call...", end=" ", flush=True)
    
    try:
        raw_response = api_call_with_retry(client, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined_prompt}
        ])
        
        # Parse the response back into fields
        import re
        for field in fields_to_translate:
            pattern = rf'===\s*{field}\s*===\s*\n(.*?)(?=\n===|$)'
            match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Remove translation prefixes
                for prefix in ["Translation:", "Translated text:", "Here is the translation:"]:
                    if value.startswith(prefix):
                        value = value[len(prefix):].strip()
                
                if field == 'cha_group':
                    translated['cha_group'] = [n.strip() for n in value.split(',')]
                else:
                    translated[field] = value
            else:
                # Field not found in response — keep original
                print(f"\n    ⚠ Could not parse '{field}' from response", end="", flush=True)
        
        print("✓")
        
    except Exception as e:
        print(f"\n    ❌ Translation failed: {e}")
        # Return original profile on failure
        return profile
    
    return translated


def translate_file(input_path: str, output_path: str, client):
    """Translate an entire JSONL file."""
    # Load all profiles
    profiles = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))
    
    total = len(profiles)
    print(f"\n{'='*60}")
    print(f"Translating: {input_path}")
    print(f"Total profiles: {total}")
    print(f"Output: {output_path}")
    print(f"Model: {MODEL}")
    print(f"{'='*60}\n")
    
    # Check for existing progress (resume support)
    already_done = 0
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            already_done = sum(1 for line in f if line.strip())
        if already_done > 0:
            print(f"📋 Found {already_done} already translated profiles. Resuming from profile {already_done + 1}...")
    
    # Translate remaining profiles
    with open(output_path, 'a', encoding='utf-8') as out_f:
        for i, profile in enumerate(profiles):
            if i < already_done:
                continue
            
            print(f"\n[{i+1}/{total}] Profile ID: {profile.get('id', 'unknown')}")
            
            try:
                translated_profile = translate_profile(profile, client, i)
                out_f.write(json.dumps(translated_profile, ensure_ascii=False) + "\n")
                out_f.flush()
                print(f"  ✅ Done ({i+1}/{total})")
                time.sleep(INTER_CALL_DELAY)  # Rate limiting between profiles
            except Exception as e:
                print(f"  ❌ Error: {e}")
                # Write original on error so we don't lose our place
                out_f.write(json.dumps(profile, ensure_ascii=False) + "\n")
                out_f.flush()
    
    print(f"\n{'='*60}")
    print(f"✅ Translation complete! Output saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Translate RLVER profiles from Chinese to English")
    parser.add_argument("--input", type=str, default=None, help="Input JSONL file path")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path")
    parser.add_argument("--both", action="store_true", help="Translate both train and test profiles")
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides env/config)")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL (overrides env/config)")
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides env/config)")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds to wait between profiles (default: 5)")
    args = parser.parse_args()
    
    global API_KEY, BASE_URL, MODEL, INTER_CALL_DELAY
    if args.api_key:
        API_KEY = args.api_key
    if args.base_url:
        BASE_URL = args.base_url
    if args.model:
        MODEL = args.model
    INTER_CALL_DELAY = args.delay
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ ERROR: Please set your API key!")
        print("   Options:")
        print("   1. Set OPENAI_API_KEY environment variable")
        print("   2. Pass --api-key on command line")
        print("   3. Edit the API_KEY variable in this script")
        return
    
    # Initialize OpenAI client
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ ERROR: openai package not installed. Run: pip install openai")
        return
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Test connection (with retry for rate limits)
    print("🔌 Testing API connection...")
    connected = False
    for test_attempt in range(3):
        try:
            test_response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Translate to English: 你好"}],
                max_tokens=10
            )
            print(f"   ✅ Connected! Test: '你好' → '{test_response.choices[0].message.content.strip()}'")
            connected = True
            break
        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            is_rate_limit = any(x in error_lower for x in ['429', 'rate', 'limit', 'quota', 'resource_exhausted'])
            
            if is_rate_limit and test_attempt < 2:
                parsed_delay = _parse_retry_delay(error_str)
                wait_time = max(parsed_delay, 40)  # Wait at least 40s
                print(f"   ⏳ Rate limited on connection test. Waiting {wait_time:.0f}s (attempt {test_attempt+1}/3)...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ Connection failed: {e}")
                print(f"")
                print(f"   💡 TIPS:")
                print(f"   • If you see '429' or 'quota exceeded': Your daily quota is used up.")
                print(f"     Wait until it resets (midnight US Pacific time) OR use a different API.")
                print(f"   • Try Groq: --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile")
                print(f"   • Try a new Gemini key from: https://aistudio.google.com/apikey")
                print(f"   • Add --delay 8 to slow down requests further.")
                return
    
    if not connected:
        print("   ❌ Could not connect after 3 attempts. Try again later or use a different API.")
        return
    
    print(f"   ⏱  Delay between profiles: {INTER_CALL_DELAY}s")
    print(f"   📊 Using model: {MODEL}")
    print(f"   🔗 API endpoint: {BASE_URL}")
    
    # Determine which files to translate
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    if args.both or (args.input is None):
        files_to_translate = [
            (str(data_dir / "train_profile.jsonl"), str(data_dir / "train_profile_english.jsonl")),
            (str(data_dir / "test_profile.jsonl"), str(data_dir / "test_profile_english.jsonl")),
        ]
    else:
        output = args.output or (args.input.replace(".jsonl", "_english.jsonl"))
        files_to_translate = [(args.input, output)]
    
    for input_path, output_path in files_to_translate:
        if not os.path.exists(input_path):
            print(f"⚠ File not found: {input_path}, skipping...")
            continue
        translate_file(input_path, output_path, client)
    
    print("\n" + "="*60)
    print("🎉 ALL TRANSLATIONS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review a few translated profiles to check quality")
    print("2. Replace the original files:")
    print(f"   - Rename train_profile.jsonl → train_profile_chinese_backup.jsonl")
    print(f"   - Rename train_profile_english.jsonl → train_profile.jsonl")
    print(f"   - Rename test_profile.jsonl → test_profile_chinese_backup.jsonl")
    print(f"   - Rename test_profile_english.jsonl → test_profile.jsonl")
    print("3. Or update the data path in train_rlver.sh to point to the English files")


if __name__ == "__main__":
    main()
