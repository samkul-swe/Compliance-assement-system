"""
Training Pair Generator for Compliance SBERT Fine Tuning

Uses OpenAI to generate phrases per severity level per rule.
Builds sentence pairs with labels 0.0 - 1.0 for fine tuning.

Severity → Label mapping:
    compliant  → 0.0
    low        → 0.25
    medium     → 0.50
    high       → 0.75
    critical   → 1.0

Pair labels = average of the two phrase labels:
    critical vs critical   → (1.0 + 1.0) / 2 = 1.0
    critical vs compliant  → (1.0 + 0.0) / 2 = 0.5  ← wait, this is wrong
    
Actually pair label = |label_a - label_b| inverted:
    Same severity  → similarity 1.0  (they should be close)
    Far severity   → similarity 0.0  (they should be far)

So pair similarity = 1.0 - |label_a - label_b|
    critical(1.0) vs critical(1.0)  → 1.0 - 0.0 = 1.0  ✓ very similar
    critical(1.0) vs compliant(0.0) → 1.0 - 1.0 = 0.0  ✓ very different
    critical(1.0) vs high(0.75)     → 1.0 - 0.25 = 0.75 ✓ quite similar
    medium(0.5)   vs low(0.25)      → 1.0 - 0.25 = 0.75 ✓ neighbouring
    critical(1.0) vs low(0.25)      → 1.0 - 0.75 = 0.25 ✓ quite different
"""

import json
import os
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from sentence_transformers import InputExample

load_dotenv()

SEVERITY_LABELS = {
    'compliant': 0.0,
    'low':       0.25,
    'medium':    0.50,
    'high':      0.75,
    'critical':  1.0
}

SEVERITY_DESCRIPTIONS = {
    'compliant': """
        Professional, empathetic, patient agent speech. No pressure whatsoever.
        Examples: showing empathy, offering payment plans, acknowledging hardship,
        being patient, offering to escalate, giving the customer time and options.
        The agent sounds genuinely helpful and respectful at all times.
    """,
    'low': """
        Borderline language. Mild pressure or slight impatience.
        The agent might not even realise this is wrong.
        Examples: subtle urgency ("you really should sort this soon"),
        mild pushiness ("I do need something from you today"),
        slight dismissiveness without being rude.
        Not clearly a violation but worth a warning.
    """,
    'medium': """
        Clear violations but not severe. Definite pressure tactics
        or ignoring customer hardship when raised.
        Examples: demanding immediate payment without offering options,
        brushing off a customer who mentions financial difficulty,
        dismissing a complaint about the product or service.
        Agent should know better. Formal warning territory.
    """,
    'high': """
        Serious violations. Clearly inappropriate language.
        Examples: abusive or demeaning language toward the customer,
        serious pressure tactics, dismissing customer entirely,
        misrepresenting consequences of non-payment significantly.
        Would warrant formal disciplinary action.
    """,
    'critical': """
        Severe violations. Explicit threats or illegal implications.
        Examples: threatening legal action without authority,
        threatening to contact employer or family about the debt,
        claiming the customer will be arrested or imprisoned,
        threatening wage garnishment, making false legal claims.
        Immediate escalation required.
    """
}


def load_compliance_rules(path: str = "data/compliance_rules.json") -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def init_openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")
    import openai
    return openai.OpenAI(api_key=key)


def generate_phrases_for_rule_and_severity(
    client,
    rule: Dict,
    severity: str,
    n: int = 30
) -> List[str]:
    """
    Generate n phrases for a specific rule at a specific severity level.
    """
    prompt = f"""You are generating training data for a compliance detection model used in debt collection call centres.

COMPLIANCE RULE:
ID: {rule['id']}
Description: {rule['description']}
Category: {rule['category']}

SEVERITY LEVEL TO GENERATE: {severity.upper()}
SEVERITY DESCRIPTION: {SEVERITY_DESCRIPTIONS[severity]}

Generate exactly {n} different agent phrases at this severity level for this specific rule.

Requirements:
- Each phrase must be realistic call centre agent speech
- Each phrase must be relevant to this specific rule
- Vary phrasing significantly across phrases - different words, structures, formality
- Keep each phrase to 1-3 sentences
- Do NOT number the phrases
- Do NOT include meta information or rule IDs in the phrases
- Make them sound like real agents, not textbook examples

Respond with valid JSON only:
{{
  "phrases": [
    "phrase one",
    "phrase two",
    ...
  ]
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You generate realistic call centre agent training data. Respond with valid JSON only."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        max_tokens=3000,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content.strip())
    phrases = result.get('phrases', [])

    if len(phrases) < n * 0.8:
        print(f"    ⚠️  Requested {n}, got {len(phrases)} phrases")

    return phrases


def generate_all_phrases(
    rules_data: Dict,
    n_per_severity: int = 30
) -> Dict:
    """
    Generate phrases for every rule at every severity level.

    Structure:
    {
        'R001': {
            'compliant': ['phrase1', 'phrase2', ...],
            'low':       ['phrase1', ...],
            'medium':    ['phrase1', ...],
            'high':      ['phrase1', ...],
            'critical':  ['phrase1', ...]
        },
        'R002': { ... }
    }
    """
    client = init_openai()
    rules = rules_data.get('rules', [])
    severities = list(SEVERITY_LABELS.keys())

    total_calls = len(rules) * len(severities)
    input_tokens_est = total_calls * 600
    output_tokens_est = total_calls * n_per_severity * 15
    cost_est = (input_tokens_est * 0.150 / 1_000_000) + (output_tokens_est * 0.600 / 1_000_000)

    print(f"\nGeneration plan:")
    print(f"  Rules:              {len(rules)}")
    print(f"  Severity levels:    {len(severities)}")
    print(f"  Phrases per combo:  {n_per_severity}")
    print(f"  Total API calls:    {total_calls}")
    print(f"  Estimated cost:     ${cost_est:.4f}")
    print()

    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return {}

    all_phrases = {}

    for rule in rules:
        rule_id = rule['id']
        print(f"\n[{rule_id}] {rule['description'][:60]}...")
        all_phrases[rule_id] = {}

        for severity in severities:
            print(f"  Generating {n_per_severity} {severity} phrases...")
            try:
                phrases = generate_phrases_for_rule_and_severity(
                    client, rule, severity, n_per_severity
                )
                all_phrases[rule_id][severity] = phrases
                print(f"  ✓ Got {len(phrases)} phrases")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                all_phrases[rule_id][severity] = []

    return all_phrases


def build_training_pairs(all_phrases: Dict) -> List[InputExample]:
    """
    Build sentence pairs with similarity labels for fine tuning.

    Pair similarity = 1.0 - |label_a - label_b|

    This means:
    - Same severity phrases → similarity 1.0 (should be close in vector space)
    - Adjacent severity phrases → similarity 0.75 (somewhat close)
    - Far severity phrases → similarity lower
    - Critical vs compliant → similarity 0.0 (should be far apart)
    """
    pairs = []
    rule_ids = list(all_phrases.keys())

    for rule_id, severity_phrases in all_phrases.items():

        # ── Within same rule, across all severity combinations ──
        for sev_a, sev_b in combinations(SEVERITY_LABELS.keys(), 2):
            phrases_a = severity_phrases.get(sev_a, [])
            phrases_b = severity_phrases.get(sev_b, [])

            if not phrases_a or not phrases_b:
                continue

            label_a = SEVERITY_LABELS[sev_a]
            label_b = SEVERITY_LABELS[sev_b]

            # Pair similarity: how close are these severity levels?
            pair_similarity = round(1.0 - abs(label_a - label_b), 2)

            # Sample pairs — not all combinations (would be too many)
            # Take up to 20 pairs per severity combination
            sampled_a = random.sample(phrases_a, min(10, len(phrases_a)))
            sampled_b = random.sample(phrases_b, min(10, len(phrases_b)))

            for p_a in sampled_a:
                for p_b in sampled_b[:3]:   # cap inner loop
                    pairs.append(InputExample(
                        texts=[p_a, p_b],
                        label=pair_similarity
                    ))

        # ── Within same rule, same severity ──
        # These should have similarity 1.0 — same violation, different phrasing
        for severity, phrases in severity_phrases.items():
            for p_a, p_b in combinations(phrases, 2):
                pairs.append(InputExample(
                    texts=[p_a, p_b],
                    label=1.0
                ))

    # ── Across different rules, same severity ──
    # Critical from R001 vs critical from R002 → still similarity 1.0
    # Both are critical violations even if different rules
    for rule_a, rule_b in combinations(rule_ids, 2):
        for severity in SEVERITY_LABELS.keys():
            phrases_a = all_phrases[rule_a].get(severity, [])
            phrases_b = all_phrases[rule_b].get(severity, [])

            if not phrases_a or not phrases_b:
                continue

            # Sample a few cross-rule same-severity pairs
            sampled_a = random.sample(phrases_a, min(5, len(phrases_a)))
            sampled_b = random.sample(phrases_b, min(5, len(phrases_b)))

            for p_a in sampled_a:
                for p_b in sampled_b[:2]:
                    pairs.append(InputExample(
                        texts=[p_a, p_b],
                        label=1.0
                    ))

    # Shuffle — model must not see all same-severity pairs together
    random.shuffle(pairs)

    # Summary
    label_counts = {}
    for p in pairs:
        l = str(round(p.label, 2))
        label_counts[l] = label_counts.get(l, 0) + 1

    print("\nTraining pairs generated:")
    for label, count in sorted(label_counts.items()):
        print(f"  label={label}: {count} pairs")
    print(f"  Total: {len(pairs)} pairs")

    return pairs


def save_phrases(phrases: Dict, path: str = "data/training_phrases.json"):
    Path("data").mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(phrases, f, indent=2)
    print(f"\n✅ Phrases saved to {path}")


def save_pairs(pairs: List[InputExample], path: str = "data/training_pairs.json"):
    """Save pairs as JSON for inspection and reuse."""
    data = [{'text_a': p.texts[0], 'text_b': p.texts[1], 'label': p.label} for p in pairs]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Pairs saved to {path}")


def load_pairs(path: str = "data/training_pairs.json") -> List[InputExample]:
    """Reload pairs from JSON without regenerating."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [InputExample(texts=[d['text_a'], d['text_b']], label=d['label']) for d in data]


if __name__ == "__main__":
    print("="*70)
    print("TRAINING PAIR GENERATOR")
    print("="*70)

    rules_data = load_compliance_rules()

    # n_per_severity = 30 means 30 phrases per rule per severity level
    # 6 rules × 5 severity levels × 30 phrases = 900 raw phrases
    # Pair generation produces thousands of training pairs from these
    phrases = generate_all_phrases(rules_data, n_per_severity=30)

    if not phrases:
        print("No phrases generated.")
        exit()

    save_phrases(phrases)

    pairs = build_training_pairs(phrases)
    save_pairs(pairs)

    print("\n✅ Done. Run fine_tune.py next.")