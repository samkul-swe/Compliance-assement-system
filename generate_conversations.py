"""
Generate synthetic customer service conversations for compliance testing.
Uses schemas from docs/api/ and compliance rules from data/compliance_rules.json

Features:
- 50% compliant, 50% non-compliant (for testing)
- Schema-driven validation
- LLM-powered for realistic conversations
"""

import json
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()
random.seed()


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
        "non_compliant": "Clearly violates at least ONE rule. Make it obvious."
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
    """Validate against conversation schema."""
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
                 total: int = 200) -> Tuple[List[Dict], List[Dict], int]:
    """Generate conversations with 50/50 split and ground truth labels."""
    num_compliant = total // 2
    num_non_compliant = total - num_compliant
    
    print(f"\n📊 Distribution for Testing:")
    print(f"   Compliant: {num_compliant} (50%)")
    print(f"   Non-Compliant: {num_non_compliant} (50%)")
    
    conversations = []
    ground_truth = []
    conv_counter = 1
    
    # Generate compliant
    print(f"\nGenerating {num_compliant} compliant conversations...")
    for i in range(num_compliant):
        conv_id = f"conv_{conv_counter:03d}"
        conv = generate_conversation(client, model, prompt_template, rules_text, "compliant", conv_id)
        
        if conv and validate_conversation(conv):
            conversations.append(conv)
            ground_truth.append({
                "conversation_id": conv_id,
                "label": "compliant",
                "generation_type": "compliant"
            })
            conv_counter += 1
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_compliant}...")
    
    # Generate non-compliant
    print(f"\nGenerating {num_non_compliant} non-compliant conversations...")
    for i in range(num_non_compliant):
        conv_id = f"conv_{conv_counter:03d}"
        conv = generate_conversation(client, model, prompt_template, rules_text, "non_compliant", conv_id)
        
        if conv and validate_conversation(conv):
            conversations.append(conv)
            ground_truth.append({
                "conversation_id": conv_id,
                "label": "non_compliant",
                "generation_type": "non_compliant"
            })
            conv_counter += 1
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_non_compliant}...")
    
    # Shuffle conversations (but keep ground truth aligned)
    print("\nShuffling conversations...")
    combined = list(zip(conversations, ground_truth))
    random.shuffle(combined)
    conversations, ground_truth = zip(*combined)
    conversations = list(conversations)
    ground_truth = list(ground_truth)
    
    # Reassign IDs sequentially after shuffle
    for i, (conv, gt) in enumerate(zip(conversations, ground_truth), 1):
        new_id = f"conv_{i:03d}"
        conv["conversation_id"] = new_id
        gt["conversation_id"] = new_id
    
    # Estimate tokens
    total_tokens = len(conversations) * 800
    
    return conversations, ground_truth, total_tokens


def save_conversations(conversations: List[Dict]):
    """Save to data/generated_conversations.json"""
    Path("data").mkdir(exist_ok=True)
    
    with open("data/generated_conversations.json", 'w') as f:
        json.dump(conversations, f, indent=2)
    
    print(f"\n✅ Saved {len(conversations)} conversations to data/generated_conversations.json")


def save_ground_truth(ground_truth: List[Dict]):
    """Save ground truth labels to data/ground_truth.json"""
    Path("data").mkdir(exist_ok=True)
    
    with open("data/ground_truth.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"✅ Saved {len(ground_truth)} labels to data/ground_truth.json")
    print("   ⚠️  Use this for evaluation ONLY - not for training/testing!")


def main():
    """Generate conversations."""
    print("="*70)
    print("CONVERSATION GENERATION (50/50 Split for Testing)")
    print("="*70)
    
    # Load
    rules_data = load_compliance_rules()
    prompt_template = load_prompt()
    client, provider, model = init_llm()
    rules_text = format_rules_for_prompt(rules_data)
    
    # Estimate cost
    print(f"\n💰 Estimated cost: ~$0.08 for 200 conversations (100 compliant + 100 non-compliant)")
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        return
    
    # Generate
    conversations, ground_truth, tokens = generate_all(client, model, prompt_template, rules_text, total=200)
    
    # Save
    save_conversations(conversations)
    save_ground_truth(ground_truth)

    # Cost
    cost = tokens * (0.150 / 1_000_000) * 0.6 + tokens * (0.600 / 1_000_000) * 0.4
    print(f"\n💰 Actual cost: ${cost:.4f} ({tokens:,} tokens)")
    
    print("\n" + "="*70)
    print("✨ GENERATION COMPLETE")
    print("="*70)
    print("\n💡 Generated data is balanced 50/50 for testing system performance")


if __name__ == "__main__":
    main()