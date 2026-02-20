"""
Generate synthetic customer service conversations for compliance testing.

Ground truth labels now include severity:
    compliant, low, medium, high, critical

Distribution:
    20% compliant
    20% low violation
    20% medium violation
    20% high violation
    20% critical violation
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

# Severity level instructions for the LLM
# Each tells the LLM exactly what kind of conversation to generate
SEVERITY_INSTRUCTIONS = {
    "compliant": {
        "label": "compliant",
        "instruction": (
            "Generate a fully compliant conversation. "
            "The agent must be professional, empathetic, and helpful throughout. "
            "No pressure tactics, no threats, no dismissiveness. "
            "Agent acknowledges hardship if raised. Offers options and payment plans."
        )
    },
    "low": {
        "label": "low",
        "instruction": (
            "Generate a conversation with a LOW severity violation. "
            "The agent makes a subtle mistake — mild pressure or slight urgency. "
            "The agent may not even realise it is wrong. "
            "Examples: 'you really should sort this today', "
            "'I do need something from you before we finish'. "
            "No explicit threats. No abusive language. Just borderline pushiness."
        )
    },
    "medium": {
        "label": "medium",
        "instruction": (
            "Generate a conversation with a MEDIUM severity violation. "
            "The agent clearly violates a rule but not severely. "
            "Examples: demanding immediate full payment without offering options, "
            "brushing off a customer who mentions financial hardship, "
            "dismissing a complaint about the product or service. "
            "Formal warning territory."
        )
    },
    "high": {
        "label": "high",
        "instruction": (
            "Generate a conversation with a HIGH severity violation. "
            "The agent uses seriously inappropriate language. "
            "Examples: abusive or demeaning language toward the customer, "
            "seriously misrepresenting consequences of non-payment "
            "('your credit will be ruined forever', 'you'll never get a loan again'). "
            "Notice territory — clearly wrong behaviour."
        )
    },
    "critical": {
        "label": "critical",
        "instruction": (
            "Generate a conversation with a CRITICAL severity violation. "
            "The agent makes explicit threats or illegal implications. "
            "Examples: threatening legal action without authority "
            "('we will take you to court', 'our lawyers will contact you'), "
            "threatening to contact employer or family about the debt, "
            "threatening arrest or wage garnishment. "
            "Immediate escalation territory."
        )
    }
}


def load_compliance_rules() -> Dict:
    rules_path = Path("data/compliance_rules.json")
    if not rules_path.exists():
        print("❌ Error: data/compliance_rules.json not found")
        sys.exit(1)
    with open(rules_path, 'r') as f:
        rules_data = json.load(f)
    print(f"✅ Loaded {len(rules_data.get('rules', []))} compliance rules")
    return rules_data


def load_prompt() -> str:
    prompt_path = Path("prompts/conversation_generation.md")
    if not prompt_path.exists():
        print("❌ Error: prompts/conversation_generation.md not found")
        sys.exit(1)
    with open(prompt_path, 'r') as f:
        return f.read()


def init_llm():
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
    return "\n".join(
        f"- **{r['id']}** ({r['severity'].upper()}): {r['description']}"
        for r in rules_data.get('rules', [])
    )


def pick_rule_for_severity(rules_data: Dict, severity: str) -> Optional[Dict]:
    """
    Pick a random rule matching the target severity level.
    For compliant conversations, no specific rule is violated.
    """
    if severity == 'compliant':
        return None

    # Map severity labels to rule severities
    severity_map = {
        'low':      ['medium'],       # low violations often bend medium rules slightly
        'medium':   ['medium'],
        'high':     ['high'],
        'critical': ['critical']
    }

    target_severities = severity_map.get(severity, [severity])
    matching_rules = [
        r for r in rules_data.get('rules', [])
        if r['severity'] in target_severities
    ]

    if not matching_rules:
        # Fall back to any rule
        matching_rules = rules_data.get('rules', [])

    return random.choice(matching_rules) if matching_rules else None


def generate_conversation(
    client,
    model: str,
    prompt_template: str,
    rules_text: str,
    severity: str,
    conv_id: str,
    target_rule: Optional[Dict] = None
) -> Optional[Dict]:
    """Generate one conversation at a specific severity level."""
    cfg = SEVERITY_INSTRUCTIONS[severity]

    # Add specific rule context to the instruction if a target rule was provided
    rule_context = ""
    if target_rule:
        rule_context = (
            f"\n\nSpecifically violate this rule:\n"
            f"**{target_rule['id']}** ({target_rule['severity'].upper()}): "
            f"{target_rule['description']}"
        )

    prompt = prompt_template.replace("{{RULES}}", rules_text)
    prompt = prompt.replace("{{TYPE}}", severity)
    prompt = prompt.replace("{{TYPE_INSTRUCTIONS}}", cfg["instruction"] + rule_context)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You generate realistic debt collection call training data. Respond with valid JSON only."
                },
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
            msg['timestamp'] = (
                base_time + timedelta(seconds=i * 30)
            ).isoformat() + "Z"

        return {
            "conversation_id": conv_id,
            "channel": data.get('channel', random.choice(['phone', 'chat', 'email'])),
            "customer_segment": data.get(
                'customer_segment',
                random.choice(['delinquent_30', 'delinquent_45', 'delinquent_60', 'delinquent_90'])
            ),
            "messages": data['messages']
        }

    except Exception as e:
        print(f"  ⚠️  Error: {str(e)[:80]}")
        return None


def validate_conversation(conv: Dict) -> bool:
    required = ['conversation_id', 'messages', 'channel', 'customer_segment']
    if not all(k in conv for k in required):
        return False
    for msg in conv.get('messages', []):
        if not all(k in msg for k in ['role', 'text', 'timestamp']):
            return False
        if msg['role'] not in ['agent', 'customer']:
            return False
    return True


def generate_all(
    client,
    model: str,
    prompt_template: str,
    rules_text: str,
    rules_data: Dict,
    total: int = 200
) -> Tuple[List[Dict], List[Dict], int]:
    """
    Generate conversations with equal distribution across severity levels.

    total=200 → 40 per severity level (compliant/low/medium/high/critical)
    """
    severities = list(SEVERITY_INSTRUCTIONS.keys())
    per_severity = total // len(severities)
    remainder = total % len(severities)

    print(f"\n📊 Distribution:")
    for sev in severities:
        count = per_severity + (1 if severities.index(sev) < remainder else 0)
        print(f"   {sev:10}: {count}")

    conversations = []
    ground_truth = []
    conv_counter = 1

    for severity in severities:
        count = per_severity + (1 if severities.index(severity) < remainder else 0)
        print(f"\nGenerating {count} {severity} conversations...")

        for i in range(count):
            conv_id = f"conv_{conv_counter:03d}"

            # Pick a specific rule to violate for non-compliant conversations
            target_rule = pick_rule_for_severity(rules_data, severity)

            conv = generate_conversation(
                client, model, prompt_template, rules_text,
                severity, conv_id, target_rule
            )

            if conv and validate_conversation(conv):
                conversations.append(conv)
                ground_truth.append({
                    "conversation_id": conv_id,
                    "label": SEVERITY_INSTRUCTIONS[severity]["label"],
                    "violated_rule_id": target_rule['id'] if target_rule else None,
                    "violated_rule_description": target_rule['description'] if target_rule else None
                })
                conv_counter += 1

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{count}...")

    # Shuffle both together
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

    total_tokens = len(conversations) * 800

    return conversations, ground_truth, total_tokens


def save_conversations(conversations: List[Dict]):
    Path("data").mkdir(exist_ok=True)
    with open("data/conversations.json", 'w') as f:
        json.dump(conversations, f, indent=2)
    print(f"\n✅ Saved {len(conversations)} conversations to data/conversations.json")


def save_ground_truth(ground_truth: List[Dict]):
    Path("data").mkdir(exist_ok=True)
    with open("data/ground_truth.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"✅ Saved {len(ground_truth)} labels to data/ground_truth.json")

    # Print label distribution
    from collections import Counter
    label_counts = Counter(gt['label'] for gt in ground_truth)
    rule_counts  = Counter(
        gt['violated_rule_id'] for gt in ground_truth
        if gt.get('violated_rule_id')
    )

    print("\n   Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label:10}: {count}")

    print("\n   Rule distribution (violated rules):")
    for rule_id, count in sorted(rule_counts.items()):
        print(f"   {rule_id:10}: {count}")

    print("\n   ⚠️  Use ground_truth.json for evaluation ONLY — not for training")


def main():
    print("=" * 70)
    print("CONVERSATION GENERATION (Severity Labels)")
    print("=" * 70)

    rules_data = load_compliance_rules()
    prompt_template = load_prompt()
    client, provider, model = init_llm()
    rules_text = format_rules_for_prompt(rules_data)

    total = 200
    severities = list(SEVERITY_INSTRUCTIONS.keys())
    per_severity = total // len(severities)

    print(f"\n  Total conversations: {total}")
    print(f"  Severity levels:     {len(severities)}")
    print(f"  Per severity:        {per_severity}")
    print(f"\n  Estimated cost: ~$0.08")

    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        return

    conversations, ground_truth, tokens = generate_all(
        client, model, prompt_template, rules_text, rules_data, total=total
    )

    save_conversations(conversations)
    save_ground_truth(ground_truth)

    cost = tokens * (0.150 / 1_000_000) * 0.6 + tokens * (0.600 / 1_000_000) * 0.4
    print(f"\n💰 Actual cost: ${cost:.4f} ({tokens:,} tokens)")

    print("\n" + "=" * 70)
    print("✨ GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()