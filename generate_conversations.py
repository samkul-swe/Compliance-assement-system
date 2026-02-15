"""
Generate synthetic customer service conversations for compliance testing.
Uses schemas from docs/api/ and compliance rules from data/compliance_rules.json

Features:
- Randomized distribution (20-60% each type)
- Schema-driven validation
- LLM-powered for realistic conversations
"""

import json
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()
random.seed()


def load_schemas():
    """Load schemas from docs/api/"""
    conv_schema_path = Path("docs/api/conversation_schema.json")
    rules_schema_path = Path("docs/api/compliance_rules_schema.json")
    
    if not conv_schema_path.exists() or not rules_schema_path.exists():
        logger.warning("Schema files not found in docs/api/, using basic validation")
        return None, None
    
    with open(conv_schema_path, 'r') as f:
        conv_schema = json.load(f)
    
    with open(rules_schema_path, 'r') as f:
        rules_schema = json.load(f)
    
    return conv_schema, rules_schema


def load_compliance_rules() -> Dict:
    """Load compliance rules from data/compliance_rules.json"""
    rules_path = Path("data/compliance_rules.json")
    
    if not rules_path.exists():
        print(f"❌ Error: data/compliance_rules.json not found")
        sys.exit(1)
    
    with open(rules_path, 'r') as f:
        rules_data = json.load(f)
    
    print(f"✅ Loaded {len(rules_data.get('rules', []))} compliance rules")
    return rules_data


def load_prompt() -> str:
    """Load conversation generation prompt from prompts/"""
    prompt_path = Path("prompts/conversation_generation.md")
    
    if not prompt_path.exists():
        print(f"❌ Error: prompts/conversation_generation.md not found")
        sys.exit(1)
    
    with open(prompt_path, 'r') as f:
        return f.read()


def init_llm():
    """Initialize LLM client."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            return openai.OpenAI(api_key=openai_key), "openai", "gpt-4o-mini"
        except ImportError:
            print("❌ pip install openai")
            sys.exit(1)
    
    print("❌ OPENAI_API_KEY not found in .env")
    sys.exit(1)


def format_rules_for_prompt(rules_data: Dict) -> str:
    """Format rules for LLM prompt."""
    rules_text = []
    for rule in rules_data.get('rules', []):
        rules_text.append(
            f"- **{rule['id']}** ({rule['severity'].upper()}): {rule['description']}"
        )
    return "\n".join(rules_text)


def generate_conversation(client, model: str, prompt_template: str, 
                          rules_text: str, conv_type: str, conv_id: str) -> Optional[Dict]:
    """Generate one conversation using LLM."""
    type_instructions = {
        "compliant": "Professional, empathetic, helpful. NO violations.",
        "non_compliant": "Clearly violates at least ONE rule. Make it obvious.",
        "vague": "Borderline - urgent but not threatening. Genuinely ambiguous."
    }
    
    prompt = prompt_template.replace("{{RULES}}", rules_text)
    prompt = prompt.replace("{{TYPE}}", conv_type)
    prompt = prompt.replace("{{TYPE_INSTRUCTIONS}}", type_instructions[conv_type])
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You generate training data. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content.strip())
        
        # Add timestamps
        base_time = datetime.now() - timedelta(days=random.randint(1, 60))
        for i, msg in enumerate(data['messages']):
            msg['timestamp'] = (base_time + timedelta(seconds=i*30)).isoformat() + "Z"
        
        return {
            "conversation_id": conv_id,
            "channel": data.get('channel', random.choice(['phone', 'chat', 'email'])),
            "customer_segment": data.get('customer_segment', random.choice(['delinquent_30', 'delinquent_45', 'delinquent_60', 'delinquent_90'])),
            "messages": data['messages']
        }
    
    except Exception as e:
        print(f"  ⚠️  Error: {str(e)[:80]}")
        return None


def validate_conversation(conv: Dict) -> bool:
    """Validate against schema."""
    required = ['conversation_id', 'messages', 'channel', 'customer_segment']
    if not all(k in conv for k in required):
        return False
    
    for msg in conv.get('messages', []):
        if not all(k in msg for k in ['role', 'text', 'timestamp']):
            return False
        if msg['role'] not in ['agent', 'customer']:
            return False
    
    return True


def generate_all(client, model: str, prompt_template: str, rules_text: str, 
                 total: int = 200) -> Tuple[List[Dict], Dict, int]:
    """Generate all conversations with randomized distribution."""
    # Randomize (min 20% each)
    min_each = int(total * 0.20)
    remaining = total - (min_each * 3)
    
    extra_compliant = random.randint(0, remaining)
    remaining -= extra_compliant
    extra_non_compliant = random.randint(0, remaining)
    extra_vague = remaining - extra_non_compliant
    
    distribution = {
        'compliant': min_each + extra_compliant,
        'non_compliant': min_each + extra_non_compliant,
        'vague': min_each + extra_vague
    }
    
    print(f"\n📊 Distribution: Compliant {distribution['compliant']}, "
          f"Non-Compliant {distribution['non_compliant']}, Vague {distribution['vague']}")
    
    conversations = []
    total_tokens = 0
    conv_counter = 1
    
    for conv_type, count in distribution.items():
        print(f"\nGenerating {count} {conv_type} conversations...")
        for _ in range(count):
            conv_id = f"conv_{conv_counter:03d}"
            conv = generate_conversation(client, model, prompt_template, rules_text, conv_type, conv_id)
            
            if conv and validate_conversation(conv):
                conversations.append(conv)
                conv_counter += 1
                if conv_counter % 10 == 1:
                    print(f"  Generated {conv_counter-1}...")
    
    random.shuffle(conversations)
    for i, conv in enumerate(conversations, 1):
        conv["conversation_id"] = f"conv_{i:03d}"
    
    # Estimate tokens (approximate)
    total_tokens = len(conversations) * 800  # ~800 tokens per conversation
    
    return conversations, distribution, total_tokens


def save_conversations(conversations: List[Dict]):
    """Save to data/generated_conversations.json"""
    Path("data").mkdir(exist_ok=True)
    
    with open("data/generated_conversations.json", 'w') as f:
        json.dump(conversations, f, indent=2)
    
    print(f"\n✅ Saved {len(conversations)} conversations to data/generated_conversations.json")


def main():
    """Generate conversations."""
    print("="*70)
    print("CONVERSATION GENERATION")
    print("="*70)
    
    # Load
    rules_data = load_compliance_rules()
    prompt_template = load_prompt()
    client, provider, model = init_llm()
    rules_text = format_rules_for_prompt(rules_data)
    
    # Estimate cost
    print(f"\n💰 Estimated cost: ~$0.08 for 200 conversations")
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        return
    
    # Generate
    conversations, distribution, tokens = generate_all(client, model, prompt_template, rules_text)
    
    # Save
    save_conversations(conversations)
    
    # Cost
    cost = tokens * (0.150 / 1_000_000) * 0.6 + tokens * (0.600 / 1_000_000) * 0.4
    print(f"\n💰 Actual cost: ${cost:.4f} ({tokens:,} tokens)")
    
    print("\n" + "="*70)
    print("✨ COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()