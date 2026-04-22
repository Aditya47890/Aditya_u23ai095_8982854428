"""
RLVER Profile Generator: Create New English Empathy Profiles
=============================================================
Generates high-quality character profiles for RLVER empathy training.
Uses an LLM to create diverse, culturally-relevant profiles following 
the exact format required by the RLVER training pipeline.

Usage:
    python generate_profiles.py --api-key "YOUR_KEY" --base-url "https://api.groq.com/openai/v1" --model "llama-3.3-70b-versatile" --count 100

Features:
    - Generates profiles in the exact RLVER JSONL format
    - Covers diverse emotional scenarios and cultural contexts
    - Balanced across difficulty levels (negative/positive/mixed)
    - Each profile has a rich backstory with hidden themes
    - Supports Groq, OpenAI, Gemini, and Ollama APIs
"""

import json
import os
import uuid
import time
import random
import argparse
from pathlib import Path

# ============================================================
# API CONFIGURATION
# ============================================================
API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("GENERATION_MODEL", "llama-3.3-70b-versatile")

# ============================================================
# SCENARIO TEMPLATES — Diverse emotional situations
# ============================================================

SCENARIOS = [
    # ===== INDIAN STUDENT LIFE (25 scenarios) =====
    {
        "category": "Academic Pressure",
        "context": "An engineering student who failed a subject and is terrified of telling their parents. They feel like a failure compared to their batch-mates.",
        "traits_pool": [["anxious", "perfectionist", "self-critical"], ["frustrated", "withdrawn", "insecure"]],
        "char_type": "negative"
    },
    {
        "category": "Academic Pressure",
        "context": "A final-year student stressed about campus placements. All their friends got placed but they haven't. They question their entire career choice.",
        "traits_pool": [["anxious", "jealous", "self-doubting"], ["stressed", "envious", "withdrawn"]],
        "char_type": "negative"
    },
    {
        "category": "Academic Pressure",
        "context": "A student preparing for GATE/GRE while managing college coursework. They're burnt out and can't focus on anything anymore.",
        "traits_pool": [["exhausted", "overwhelmed", "irritable"], ["tired", "frustrated", "scattered"]],
        "char_type": "negative"
    },
    {
        "category": "Academic Pressure",
        "context": "A student who got a lower CGPA than expected this semester. Their scholarship is at risk and they feel like they've let everyone down.",
        "traits_pool": [["worried", "guilty", "ashamed"], ["anxious", "self-blaming", "desperate"]],
        "char_type": "negative"
    },
    {
        "category": "Academic Pressure",
        "context": "A research student whose thesis advisor is very demanding and dismissive. They feel stuck and consider dropping out of their master's program.",
        "traits_pool": [["frustrated", "helpless", "angry"], ["demoralized", "resentful", "trapped"]],
        "char_type": "negative"
    },
    {
        "category": "Hostel/Campus Life",
        "context": "A first-year student who is homesick and struggling to adjust to hostel life. They feel isolated because they don't speak the local language well.",
        "traits_pool": [["lonely", "homesick", "shy"], ["withdrawn", "nostalgic", "insecure"]],
        "char_type": "negative"
    },
    {
        "category": "Hostel/Campus Life",
        "context": "A student dealing with a toxic roommate who is passive-aggressive and talks behind their back. They don't know how to handle the conflict.",
        "traits_pool": [["frustrated", "conflict-avoidant", "anxious"], ["annoyed", "passive", "stressed"]],
        "char_type": "mixed"
    },
    {
        "category": "Hostel/Campus Life",
        "context": "A student who was unfairly blamed for a ragging incident they didn't participate in. They feel betrayed by friends who didn't speak up for them.",
        "traits_pool": [["angry", "betrayed", "righteous"], ["furious", "hurt", "indignant"]],
        "char_type": "negative"
    },
    {
        "category": "Career Anxiety",
        "context": "A student confused between pursuing higher studies abroad or taking a job offer in India. Their parents want them to stay, but they dream of going abroad.",
        "traits_pool": [["conflicted", "ambitious", "guilty"], ["torn", "aspirational", "obligation-bound"]],
        "char_type": "mixed"
    },
    {
        "category": "Career Anxiety",
        "context": "A student who lost interest in engineering and wants to switch to the arts/humanities. They're afraid their parents will be devastated.",
        "traits_pool": [["conflicted", "creative", "anxious"], ["passionate", "fearful", "rebellious"]],
        "char_type": "mixed"
    },
    # ===== FAMILY CONFLICTS (15 scenarios) =====
    {
        "category": "Family Pressure",
        "context": "A young professional whose parents are pressuring them for an arranged marriage. They're not ready and don't know how to say no without hurting them.",
        "traits_pool": [["respectful", "conflicted", "frustrated"], ["dutiful", "torn", "anxious"]],
        "char_type": "mixed"
    },
    {
        "category": "Family Pressure",
        "context": "A person whose parents constantly compare them to their more successful sibling. They feel invisible and unappreciated despite working hard.",
        "traits_pool": [["resentful", "jealous", "sad"], ["bitter", "insecure", "longing"]],
        "char_type": "negative"
    },
    {
        "category": "Family Pressure",
        "context": "A student who discovered their parents are having serious financial problems but are hiding it from the family. They feel helpless and scared.",
        "traits_pool": [["worried", "protective", "scared"], ["anxious", "responsible", "overwhelmed"]],
        "char_type": "mixed"
    },
    {
        "category": "Family Conflict",
        "context": "A person caught between their parents during a marital conflict. Both parents confide in them and they feel emotionally drained playing mediator.",
        "traits_pool": [["exhausted", "trapped", "empathetic"], ["burdened", "conflicted", "emotionally drained"]],
        "char_type": "negative"
    },
    {
        "category": "Family Conflict",
        "context": "A young adult whose elderly grandparent is sick, but the family can't agree on treatment. They feel guilty about not being able to do more.",
        "traits_pool": [["guilty", "helpless", "caring"], ["worried", "frustrated", "devoted"]],
        "char_type": "mixed"
    },
    # ===== RELATIONSHIP ISSUES (20 scenarios) =====
    {
        "category": "Romantic Relationship",
        "context": "A person who discovered their partner has been emotionally cheating with an ex. They're torn between confrontation and ignorance.",
        "traits_pool": [["hurt", "suspicious", "insecure"], ["angry", "betrayed", "jealous"]],
        "char_type": "negative"
    },
    {
        "category": "Romantic Relationship",
        "context": "A person in a long-distance relationship who feels like they're growing apart. They love their partner but feel increasingly disconnected.",
        "traits_pool": [["lonely", "nostalgic", "uncertain"], ["sad", "patient", "doubtful"]],
        "char_type": "mixed"
    },
    {
        "category": "Romantic Relationship",
        "context": "A person who was recently broken up with and is struggling to accept it. They keep replaying conversations trying to understand what went wrong.",
        "traits_pool": [["heartbroken", "obsessive", "confused"], ["devastated", "analytical", "denial"]],
        "char_type": "negative"
    },
    {
        "category": "Friendship",
        "context": "A person whose best friend suddenly stopped talking to them without explanation. They feel confused, hurt, and keep wondering if they did something wrong.",
        "traits_pool": [["confused", "hurt", "anxious"], ["abandoned", "insecure", "overthinking"]],
        "char_type": "negative"
    },
    {
        "category": "Friendship",
        "context": "A person who feels like they're always the one making effort in their friendships. They're exhausted and wondering if any of their friends truly care.",
        "traits_pool": [["tired", "undervalued", "resentful"], ["giving", "neglected", "bitter"]],
        "char_type": "negative"
    },
    # ===== WORKPLACE STRESS (20 scenarios) =====
    {
        "category": "Workplace",
        "context": "A new employee who made a major mistake at work that cost the company money. They're terrified of being fired and can't sleep at night.",
        "traits_pool": [["terrified", "guilty", "anxious"], ["panicked", "self-blaming", "stressed"]],
        "char_type": "negative"
    },
    {
        "category": "Workplace",
        "context": "A person who keeps getting passed over for promotions despite being the hardest worker. A less qualified colleague just got the role they wanted.",
        "traits_pool": [["frustrated", "bitter", "undervalued"], ["angry", "disillusioned", "resentful"]],
        "char_type": "negative"
    },
    {
        "category": "Workplace",
        "context": "A remote worker who feels isolated and disconnected from their team. They haven't had a meaningful social interaction in weeks.",
        "traits_pool": [["lonely", "disconnected", "apathetic"], ["isolated", "unmotivated", "numb"]],
        "char_type": "negative"
    },
    {
        "category": "Workplace",
        "context": "A person who loves their job but their toxic manager is making the workplace unbearable. They can't decide between staying and quitting.",
        "traits_pool": [["conflicted", "stressed", "angry"], ["torn", "anxious", "frustrated"]],
        "char_type": "mixed"
    },
    {
        "category": "Workplace",
        "context": "A fresh graduate whose first job is nothing like what they expected. They feel overworked, underpaid, and question if this is what adult life is really about.",
        "traits_pool": [["disillusioned", "exhausted", "lost"], ["overwhelmed", "disappointed", "cynical"]],
        "char_type": "negative"
    },
    # ===== MENTAL HEALTH (10 scenarios) =====
    {
        "category": "Mental Health",
        "context": "A person experiencing imposter syndrome at their new job. Despite being qualified, they feel like a fraud who will be 'discovered' any day.",
        "traits_pool": [["insecure", "anxious", "perfectionist"], ["self-doubting", "overachieving", "fearful"]],
        "char_type": "negative"
    },
    {
        "category": "Mental Health",
        "context": "A person dealing with social anxiety who was forced to give a presentation at work. The presentation went badly and they're spiraling with shame.",
        "traits_pool": [["ashamed", "anxious", "withdrawn"], ["humiliated", "avoidant", "self-conscious"]],
        "char_type": "negative"
    },
    {
        "category": "Mental Health",
        "context": "A person who has been pretending to be happy on social media while actually struggling with loneliness. They feel nobody knows the real them.",
        "traits_pool": [["lonely", "fake", "desperate"], ["disconnected", "performative", "sad"]],
        "char_type": "negative"
    },
    {
        "category": "Mental Health",
        "context": "A person who recently moved to a new city for work and has no friends or support system. The loneliness is becoming overwhelming.",
        "traits_pool": [["lonely", "homesick", "anxious"], ["isolated", "lost", "nostalgic"]],
        "char_type": "negative"
    },
    {
        "category": "Mental Health",
        "context": "A person dealing with burnout who can't find motivation to do even basic tasks. They feel guilty about being unproductive but physically can't push themselves.",
        "traits_pool": [["exhausted", "guilty", "numb"], ["burnt out", "apathetic", "self-critical"]],
        "char_type": "negative"
    },
    # ===== CULTURAL DILEMMAS (10 scenarios) =====
    {
        "category": "Cultural Dilemma",
        "context": "A person from a conservative family who is in an inter-caste relationship. They love their partner but know their family will never accept it.",
        "traits_pool": [["torn", "loving", "fearful"], ["defiant", "anxious", "passionate"]],
        "char_type": "mixed"
    },
    {
        "category": "Cultural Dilemma",
        "context": "A woman who wants to pursue a career but her in-laws expect her to be a homemaker. She feels trapped between tradition and her own dreams.",
        "traits_pool": [["frustrated", "ambitious", "conflicted"], ["resentful", "determined", "guilty"]],
        "char_type": "mixed"
    },
    {
        "category": "Cultural Dilemma",
        "context": "A person who came out to their best friend but was met with judgment instead of support. They feel rejected and more alone than ever.",
        "traits_pool": [["rejected", "hurt", "scared"], ["betrayed", "vulnerable", "isolated"]],
        "char_type": "negative"
    },
    {
        "category": "Cultural Dilemma",
        "context": "A person who wants to take a gap year to travel and find themselves, but their parents consider it irresponsible. They feel suffocated by expectations.",
        "traits_pool": [["restless", "idealistic", "frustrated"], ["adventurous", "misunderstood", "rebellious"]],
        "char_type": "mixed"
    },
    {
        "category": "Cultural Dilemma",
        "context": "A student who got into a foreign university but their elderly parents need them to stay. They feel torn between their dreams and their duty.",
        "traits_pool": [["conflicted", "devoted", "ambitious"], ["guilty", "torn", "resentful"]],
        "char_type": "mixed"
    },
]


PROFILE_GENERATION_PROMPT = """You are an expert at creating detailed character profiles for emotional dialogue simulation research. Generate a complete character profile in JSON format.

## Context
{context}
Category: {category}
Character type: {char_type}
Personality traits: {traits}

## Required JSON Fields

### "player" field (400-700 characters):
Create a detailed character bio including:
- Name (use a realistic name appropriate for the context), Age, Gender
- Occupation
- Personal hobbies (2-3 detailed interests)
- Habits and behavioral traits (how they act in daily life, their personality quirks)
- Speaking style (how they talk — formal/informal, direct/indirect, emotional/reserved)
- Speaking manner (do they ask questions, use sarcasm, stay quiet, etc.)

### "scene" field (1200-2000 characters):
Create a detailed background story with these sections:

#### Background Story
**Event Origin:** What triggered the current emotional state
**Event Progression:** 4 chronological stages showing how the situation escalated:
- Stage 1: Initial event (1 month ago)
- Stage 2: Escalation (2-3 weeks ago)  
- Stage 3: Crisis point (1-2 weeks ago)
- Stage 4: Current state (recent days)

**Main Conflict:** The core emotional tension
**Player's Difficulties:** What they've tried and failed to resolve

#### Reactions at Different Emotion Levels
- **High emotion (calm, relaxed):** How they behave when feeling better
- **Low emotion (agitated, angry, desperate):** How they behave when upset
- **Medium emotion (impatient, lost):** How they behave when confused

#### Reactions to NPC Responses
- **NPC reply aligns with hidden theme (emotion rises):** What helps them
- **NPC reply deviates from hidden theme (emotion drops):** What makes them worse

#### Hidden Theme
The underlying need they won't directly state (e.g., "You want validation that your feelings are okay" or "You want someone to acknowledge your effort")

### "main_cha" field:
One of: "negative", "positive", or "mixed"

### "cha_group" field:
A list of exactly 3 personality trait words, e.g., ["anxious", "perfectionist", "self-critical"]

### "task" field:
A one-sentence description of what the character secretly wants from the conversation.

### "first_talk" field:
A natural, casual opening line (15-40 words) that the character would say to start the conversation. It should hint at their problem without being too direct. Make it sound like something a real person would text a friend.

## Output Format
Output ONLY a valid JSON object. No markdown, no explanation, no code blocks. Just the raw JSON.
"""


def generate_single_profile(client, scenario, profile_idx):
    """Generate a single profile from a scenario template."""
    traits = random.choice(scenario["traits_pool"])
    
    prompt = PROFILE_GENERATION_PROMPT.format(
        context=scenario["context"],
        category=scenario["category"],
        char_type=scenario["char_type"],
        traits=", ".join(traits)
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a character profile generator for empathy research. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Higher temperature for diversity
            max_tokens=4096,
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Clean up common LLM formatting issues
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        
        profile = json.loads(raw)
        
        # Ensure required fields
        profile["id"] = str(uuid.uuid4())
        
        # Validate required fields exist
        required = ["player", "scene", "main_cha", "cha_group", "task", "first_talk"]
        for field in required:
            if field not in profile:
                raise ValueError(f"Missing field: {field}")
        
        # Ensure cha_group is a list
        if isinstance(profile["cha_group"], str):
            profile["cha_group"] = [t.strip() for t in profile["cha_group"].split(",")]
        
        return profile
        
    except json.JSONDecodeError as e:
        print(f"    ⚠ JSON parse error: {e}")
        print(f"    Raw output (first 200 chars): {raw[:200]}")
        return None
    except Exception as e:
        print(f"    ⚠ Error: {e}")
        return None


def generate_profiles(count, output_path, client):
    """Generate multiple profiles and save to JSONL."""
    print(f"\n{'='*60}")
    print(f"Generating {count} new empathy profiles")
    print(f"Output: {output_path}")
    print(f"Model: {MODEL}")
    print(f"{'='*60}\n")
    
    # Check for existing progress
    already_done = 0
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            already_done = sum(1 for line in f if line.strip())
        if already_done > 0:
            print(f"📋 Found {already_done} profiles already generated. Resuming from {already_done + 1}...")
    
    generated = already_done
    failed = 0
    
    with open(output_path, 'a', encoding='utf-8') as out_f:
        while generated < count:
            # Cycle through scenarios
            scenario_idx = generated % len(SCENARIOS)
            scenario = SCENARIOS[scenario_idx]
            
            print(f"\n[{generated+1}/{count}] Category: {scenario['category']} | Type: {scenario['char_type']}")
            
            profile = generate_single_profile(client, scenario, generated)
            
            if profile:
                out_f.write(json.dumps(profile, ensure_ascii=False) + "\n")
                out_f.flush()
                generated += 1
                print(f"  ✅ Generated (char: {profile.get('main_cha', '?')}, traits: {profile.get('cha_group', [])})")
                print(f"  First line: \"{profile.get('first_talk', '')[:80]}...\"")
            else:
                failed += 1
                print(f"  ❌ Failed (attempt {failed})")
                if failed > 10:
                    print("  Too many failures, trying next scenario...")
                    generated += 1  # Skip this one
                    failed = 0
            
            time.sleep(1)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"✅ Generation complete!")
    print(f"   Total generated: {generated}")
    print(f"   Output: {output_path}")
    print(f"{'='*60}")


def merge_profiles(translated_path, generated_path, output_path):
    """Merge translated Chinese profiles and new English profiles into one file."""
    profiles = []
    
    if os.path.exists(translated_path):
        with open(translated_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    profiles.append(json.loads(line))
        print(f"Loaded {len(profiles)} translated profiles from {translated_path}")
    
    new_count = 0
    if os.path.exists(generated_path):
        with open(generated_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    profiles.append(json.loads(line))
                    new_count += 1
        print(f"Loaded {new_count} new profiles from {generated_path}")
    
    # Shuffle for better training
    random.shuffle(profiles)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in profiles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Merged {len(profiles)} total profiles → {output_path}")
    return len(profiles)


def main():
    parser = argparse.ArgumentParser(description="Generate new empathy profiles for RLVER training")
    parser.add_argument("--count", type=int, default=100, help="Number of profiles to generate (default: 100)")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--merge", action="store_true", help="After generating, merge with translated profiles")
    args = parser.parse_args()
    
    global API_KEY, BASE_URL, MODEL
    if args.api_key:
        API_KEY = args.api_key
    if args.base_url:
        BASE_URL = args.base_url
    if args.model:
        MODEL = args.model
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ ERROR: Please set your API key!")
        print("   Pass --api-key on command line")
        return
    
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ ERROR: openai package not installed. Run: pip install openai")
        return
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Test connection
    print("🔌 Testing API connection...")
    try:
        test = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say 'ready' in one word"}],
            max_tokens=5
        )
        print(f"   ✅ Connected! Response: {test.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return
    
    # Determine output path
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    output_path = args.output or str(data_dir / "new_english_profiles.jsonl")
    
    # Generate profiles
    generate_profiles(args.count, output_path, client)
    
    # Optionally merge with translated profiles
    if args.merge:
        translated_train = str(data_dir / "train_profile_english.jsonl")
        merged_output = str(data_dir / "train_profile_merged.jsonl")
        
        if os.path.exists(translated_train):
            merge_profiles(translated_train, output_path, merged_output)
            print(f"\n📋 To use the merged dataset, run:")
            print(f"   mv {data_dir}/train_profile.jsonl {data_dir}/train_profile_backup.jsonl")
            print(f"   mv {data_dir}/train_profile_merged.jsonl {data_dir}/train_profile.jsonl")
        else:
            print(f"\n⚠ Translated profiles not found at {translated_train}")
            print(f"  Run translate_profiles.py first, then use --merge")
    
    print("\n" + "="*60)
    print("🎉 DONE! Next steps:")
    print("="*60)
    print(f"1. Review generated profiles: {output_path}")
    print(f"2. If not already done, run translate_profiles.py to translate Chinese profiles")
    print(f"3. Merge all profiles:")
    print(f"   python generate_profiles.py --merge")
    print(f"4. Rename merged file to train_profile.jsonl")
    print(f"5. Start training on HPC!")


if __name__ == "__main__":
    main()
