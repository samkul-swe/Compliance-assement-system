"""
Generate 200 synthetic customer service conversations for compliance testing.
Matches the exact schema from docs/api/conversation_schema.json and 
compliance rules from data/compliance_rules.json

75% compliant, 25% non-compliant, with some intentionally complex/ambiguous cases.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict

# Seed for reproducibility during development
random.seed(42)

# Load existing compliance rules to ensure our generator aligns
COMPLIANCE_RULES = {
    "R001": {
        "keywords": ["sue", "suing", "garnish", "wages", "legal action", "lawyer", "arrest", "prison"],
        "severity": "critical",
        "category": "forbidden_language"
    },
    "R002": {
        "keywords": ["family will find out", "tell your employer", "contact your boss", "call your family"],
        "severity": "critical", 
        "category": "forbidden_language"
    },
    "R003": {
        "keywords": ["that's not our problem", "not my problem", "your fault", "you have to"],
        "severity": "high",
        "category": "pressure_tactics"
    },
    "R004": {
        "keywords": ["pay immediately", "pay right now", "need payment in full by"],
        "severity": "medium",
        "category": "pressure_tactics"
    },
    "R005": {
        "keywords": ["credit will be ruined", "destroy your credit", "never get a loan again"],
        "severity": "high",
        "category": "misrepresentation"
    },
    "R006": {
        "keywords": ["I understand", "I'm sorry to hear", "I'll look into", "escalate", "waive"],
        "severity": "medium",
        "category": "fair_engagement",
        "positive": True  # This rule requires these phrases
    }
}

CUSTOMER_SEGMENTS = ["delinquent_30", "delinquent_45", "delinquent_60", "delinquent_90"]
CHANNELS = ["phone", "chat", "email"]

# Customer situations (aligns with customer_situation_schema.json)
SITUATIONS = {
    "product_loss": [
        "never received the product I ordered",
        "was charged for a cancelled subscription",
        "got charged twice for the same purchase",
        "the service was terminated but charges continued"
    ],
    "substandard_service": [
        "premium features don't work as advertised",
        "service is extremely slow and unreliable",
        "customer support never responds to my tickets",
        "app crashes constantly and I can't use it"
    ],
    "other": [
        "lost my job and can't make payments right now",
        "going through a divorce and need more time",
        "had unexpected medical bills this month",
        "just need to understand my payment options"
    ]
}

# Compliant agent responses - professional, empathetic
COMPLIANT_AGENT_PHRASES = [
    "I understand your situation. Let me look into this for you.",
    "I'm sorry to hear you're going through that. How can I help?",
    "Let me review your account and see what options are available.",
    "I'll escalate this to the appropriate team right away.",
    "We can work with you on a payment plan if that helps.",
    "I can waive the late fees while we sort this out.",
    "Let me connect you with someone who can help resolve this.",
    "I appreciate you reaching out. Let's find a solution together.",
    "I'll create a ticket and have someone follow up within 2 business days.",
    "Thank you for your patience. I'm here to help."
]

# Non-compliant agent responses - violations of R001-R005
NON_COMPLIANT_CRITICAL = [
    "We're going to sue you if you don't pay immediately.",
    "I'll have to garnish your wages if this isn't resolved by Friday.",
    "Your family will find out about this debt.",
    "We'll contact your employer if you don't pay.",
    "You're going to prison if you don't take care of this.",
    "We're sending this to our lawyers and you'll be sued.",
    "I'll make sure your boss knows about this.",
    "We can have you arrested for non-payment."
]

NON_COMPLIANT_HIGH = [
    "That's not our problem. You need to figure it out.",
    "This is your fault for not paying on time.",
    "Your credit will be ruined forever.",
    "You'll never get a loan again if you don't pay.",
    "I don't care about your excuses. Pay up.",
    "That's not my problem. You signed the agreement.",
    "We're going to destroy your credit rating.",
    "You have to borrow money from someone. That's on you."
]

NON_COMPLIANT_MEDIUM = [
    "You need to pay immediately. No exceptions.",
    "Pay right now or face the consequences.",
    "We need payment in full by tomorrow or else.",
    "You must pay this entire amount today.",
    "Pay immediately or we take further action.",
    "I need payment in full by end of business day."
]

# Ambiguous phrases - could be interpreted as violations depending on context/tone
AMBIGUOUS_PHRASES = [
    "We really need to resolve this soon.",
    "It would be best if you could pay as quickly as possible.",
    "I strongly encourage you to make a payment today.",
    "Time is of the essence here.",
    "This situation is becoming more serious.",
    "We're running out of options if this continues.",
    "I hope you understand the urgency of this matter.",
    "The longer this goes on, the more difficult it becomes.",
    "I'd hate to see this escalate further.",
    "You should really consider making a payment soon."
]

def generate_timestamp(base_date: datetime, offset_seconds: int) -> str:
    """Generate ISO timestamp with offset from base date."""
    return (base_date + timedelta(seconds=offset_seconds)).isoformat() + "Z"

def pick_situation() -> tuple:
    """Pick a random situation type and specific complaint."""
    situation_type = random.choice(list(SITUATIONS.keys()))
    complaint = random.choice(SITUATIONS[situation_type])
    return situation_type, complaint

def generate_compliant_conversation(conv_id: str) -> Dict:
    """Generate a clearly compliant conversation."""
    base_time = datetime.now() - timedelta(days=random.randint(1, 60))
    segment = random.choice(CUSTOMER_SEGMENTS)
    channel = random.choice(CHANNELS)
    balance = random.randint(100, 2000)
    
    situation_type, complaint = pick_situation()
    
    # Start with agent reaching out or customer initiating
    if random.random() < 0.5:
        # Agent initiates
        messages = [
            {
                "role": "agent",
                "text": f"Hi, this is calling from collections regarding your account. I see you have a balance of ${balance}. How can I help you today?",
                "timestamp": generate_timestamp(base_time, 0)
            },
            {
                "role": "customer",
                "text": f"Hi, I wanted to talk about this. {complaint}",
                "timestamp": generate_timestamp(base_time, 25)
            }
        ]
        offset = 50
    else:
        # Customer initiates
        messages = [
            {
                "role": "customer",
                "text": f"I need to discuss my balance of ${balance}. {complaint}",
                "timestamp": generate_timestamp(base_time, 0)
            },
            {
                "role": "agent",
                "text": random.choice(COMPLIANT_AGENT_PHRASES),
                "timestamp": generate_timestamp(base_time, 20)
            }
        ]
        offset = 40
    
    # Add 2-4 more exchanges - all compliant
    num_exchanges = random.randint(2, 4)
    for i in range(num_exchanges):
        customer_responses = [
            "What are my options?",
            "Can you help me set up a payment plan?",
            "How long will this take to resolve?",
            "Thank you for understanding.",
            "That would be helpful, yes.",
            "I appreciate your help with this."
        ]
        
        messages.extend([
            {
                "role": "customer",
                "text": random.choice(customer_responses),
                "timestamp": generate_timestamp(base_time, offset + i*40)
            },
            {
                "role": "agent",
                "text": random.choice(COMPLIANT_AGENT_PHRASES),
                "timestamp": generate_timestamp(base_time, offset + i*40 + 20)
            }
        ])
    
    return {
        "conversation_id": conv_id,
        "channel": channel,
        "customer_segment": segment,
        "messages": messages
    }

def generate_non_compliant_conversation(conv_id: str, severity: str = "mixed") -> Dict:
    """Generate a clearly non-compliant conversation with specified severity."""
    base_time = datetime.now() - timedelta(days=random.randint(1, 60))
    segment = random.choice(["delinquent_60", "delinquent_90"])  # Higher delinquency
    channel = random.choice(CHANNELS)
    balance = random.randint(500, 3000)
    
    situation_type, complaint = pick_situation()
    
    # Choose violation severity
    if severity == "critical":
        violation_pool = NON_COMPLIANT_CRITICAL
    elif severity == "high":
        violation_pool = NON_COMPLIANT_HIGH
    elif severity == "medium":
        violation_pool = NON_COMPLIANT_MEDIUM
    else:  # mixed
        violation_pool = NON_COMPLIANT_CRITICAL + NON_COMPLIANT_HIGH + NON_COMPLIANT_MEDIUM
    
    # Agent starts aggressively
    messages = [
        {
            "role": "agent",
            "text": f"You're past due on ${balance}. {random.choice(violation_pool)}",
            "timestamp": generate_timestamp(base_time, 0)
        },
        {
            "role": "customer",
            "text": complaint,
            "timestamp": generate_timestamp(base_time, 30)
        }
    ]
    
    # Add 1-3 more exchanges with potential additional violations
    num_exchanges = random.randint(1, 3)
    offset = 60
    
    for i in range(num_exchanges):
        customer_responses = [
            "I'm trying to work this out.",
            "I need more time.",
            "Can we set up a payment plan?",
            "This isn't fair.",
            "I'm doing my best here."
        ]
        
        # 60% chance of another violation, 40% slightly softer
        if random.random() < 0.6:
            agent_response = random.choice(violation_pool)
        else:
            agent_response = random.choice(AMBIGUOUS_PHRASES)
        
        messages.extend([
            {
                "role": "customer",
                "text": random.choice(customer_responses),
                "timestamp": generate_timestamp(base_time, offset + i*50)
            },
            {
                "role": "agent",
                "text": agent_response,
                "timestamp": generate_timestamp(base_time, offset + i*50 + 25)
            }
        ])
    
    return {
        "conversation_id": conv_id,
        "channel": channel,
        "customer_segment": segment,
        "messages": messages
    }

def generate_ambiguous_conversation(conv_id: str, lean: str = "neutral") -> Dict:
    """
    Generate an ambiguous conversation that's hard to classify.
    lean: 'compliant', 'violation', or 'neutral'
    """
    base_time = datetime.now() - timedelta(days=random.randint(1, 60))
    segment = random.choice(CUSTOMER_SEGMENTS)
    channel = random.choice(CHANNELS)
    balance = random.randint(200, 1500)
    
    situation_type, complaint = pick_situation()
    
    # Start neutrally
    messages = [
        {
            "role": "agent",
            "text": f"Hi, I'm calling about your account balance of ${balance}. Can we discuss this?",
            "timestamp": generate_timestamp(base_time, 0)
        },
        {
            "role": "customer",
            "text": complaint,
            "timestamp": generate_timestamp(base_time, 28)
        }
    ]
    
    offset = 56
    
    # Build conversation with mix of ambiguous and clearer phrases
    num_exchanges = random.randint(3, 5)
    for i in range(num_exchanges):
        customer_responses = [
            "I see. What happens next?",
            "How much time do I have?",
            "Is there any flexibility here?",
            "I'm concerned about my credit.",
            "What are you saying exactly?"
        ]
        
        # Determine agent response based on lean
        if lean == "compliant":
            # Mostly compliant with some ambiguous
            if random.random() < 0.7:
                agent_response = random.choice(COMPLIANT_AGENT_PHRASES)
            else:
                agent_response = random.choice(AMBIGUOUS_PHRASES)
        elif lean == "violation":
            # Mostly ambiguous with occasional clear violation
            if random.random() < 0.5:
                agent_response = random.choice(AMBIGUOUS_PHRASES)
            elif random.random() < 0.8:
                agent_response = random.choice(NON_COMPLIANT_MEDIUM)
            else:
                agent_response = random.choice(NON_COMPLIANT_HIGH)
        else:  # neutral
            # Mix of all types
            roll = random.random()
            if roll < 0.4:
                agent_response = random.choice(AMBIGUOUS_PHRASES)
            elif roll < 0.7:
                agent_response = random.choice(COMPLIANT_AGENT_PHRASES)
            else:
                agent_response = random.choice(NON_COMPLIANT_MEDIUM)
        
        messages.extend([
            {
                "role": "customer",
                "text": random.choice(customer_responses),
                "timestamp": generate_timestamp(base_time, offset + i*45)
            },
            {
                "role": "agent",
                "text": agent_response,
                "timestamp": generate_timestamp(base_time, offset + i*45 + 22)
            }
        ])
    
    return {
        "conversation_id": conv_id,
        "channel": channel,
        "customer_segment": segment,
        "messages": messages
    }

def generate_conversations(total: int = 200) -> List[Dict]:
    """
    Generate conversations with specified distribution:
    - 75% compliant (150)
    - 25% non-compliant (50)
    - Mix in ambiguous cases throughout
    """
    conversations = []
    
    # Calculate base distributions
    num_compliant_target = int(total * 0.75)  # 150
    num_non_compliant_target = total - num_compliant_target  # 50
    
    # Subdivide compliant group
    # 80% clearly compliant, 20% ambiguous leaning compliant
    num_clear_compliant = int(num_compliant_target * 0.80)  # 120
    num_ambiguous_compliant = num_compliant_target - num_clear_compliant  # 30
    
    # Subdivide non-compliant group
    # 60% clearly non-compliant, 40% ambiguous leaning violation
    num_clear_non_compliant = int(num_non_compliant_target * 0.60)  # 30
    num_ambiguous_violation = num_non_compliant_target - num_clear_non_compliant  # 20
    
    conv_counter = 1
    
    # Generate clearly compliant conversations (120)
    for _ in range(num_clear_compliant):
        conv_id = f"conv_{conv_counter:03d}"
        conversations.append(generate_compliant_conversation(conv_id))
        conv_counter += 1
    
    # Generate clearly non-compliant conversations (30)
    # Mix severities
    for i in range(num_clear_non_compliant):
        conv_id = f"conv_{conv_counter:03d}"
        if i < 10:
            severity = "critical"
        elif i < 20:
            severity = "high"
        else:
            severity = "medium"
        conversations.append(generate_non_compliant_conversation(conv_id, severity))
        conv_counter += 1
    
    # Generate ambiguous leaning compliant (30)
    for _ in range(num_ambiguous_compliant):
        conv_id = f"conv_{conv_counter:03d}"
        conversations.append(generate_ambiguous_conversation(conv_id, lean="compliant"))
        conv_counter += 1
    
    # Generate ambiguous leaning violation (20)
    for _ in range(num_ambiguous_violation):
        conv_id = f"conv_{conv_counter:03d}"
        conversations.append(generate_ambiguous_conversation(conv_id, lean="violation"))
        conv_counter += 1
    
    # Shuffle to mix them up
    random.shuffle(conversations)
    
    # Reassign conversation IDs sequentially after shuffle
    for i, conv in enumerate(conversations, 1):
        conv["conversation_id"] = f"conv_{i:03d}"
    
    return conversations

def save_conversations(conversations: List[Dict], output_file: str = "data/generated_conversations.json"):
    """Save conversations to JSON file matching conversation_schema.json format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated {len(conversations)} conversations")
    print(f"📁 Saved to: {output_file}")

def print_statistics(conversations: List[Dict]):
    """Print statistics about generated conversations."""
    print(f"\n📊 Generation Statistics:")
    print(f"Total conversations: {len(conversations)}")
    
    # Channel distribution
    channels = {}
    for conv in conversations:
        channel = conv['channel']
        channels[channel] = channels.get(channel, 0) + 1
    print(f"\n📱 Channel distribution:")
    for channel, count in sorted(channels.items()):
        print(f"  {channel}: {count} ({count/len(conversations)*100:.1f}%)")
    
    # Segment distribution
    segments = {}
    for conv in conversations:
        segment = conv['customer_segment']
        segments[segment] = segments.get(segment, 0) + 1
    print(f"\n👥 Customer segment distribution:")
    for segment, count in sorted(segments.items()):
        print(f"  {segment}: {count} ({count/len(conversations)*100:.1f}%)")
    
    # Message statistics
    message_counts = [len(conv['messages']) for conv in conversations]
    print(f"\n💬 Message statistics:")
    print(f"  Min messages per conversation: {min(message_counts)}")
    print(f"  Max messages per conversation: {max(message_counts)}")
    print(f"  Avg messages per conversation: {sum(message_counts)/len(message_counts):.1f}")
    
    print(f"\n⚠️  Compliance distribution is intentionally hidden for blind testing.")

if __name__ == "__main__":
    print("🚀 Generating synthetic compliance test conversations...")
    print("=" * 70)
    
    # Generate conversations
    conversations = generate_conversations(total=200)
    
    # Save to file
    save_conversations(conversations)
    
    # Print statistics
    print_statistics(conversations)
    
    print("\n" + "=" * 70)
    print("✨ Generation complete!")
    print("\n💡 Next steps:")
    print("   1. Review data/generated_conversations.json")
    print("   2. Build your compliance checker with confidence scoring")
    print("   3. Test against this dataset")
    print("   4. I'll help you validate results afterward")