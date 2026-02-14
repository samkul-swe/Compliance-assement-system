#!/usr/bin/env python3
"""
Reference implementation: load conversations and compliance rules from data/,
run a simple compliance check and situation classifier, and print results.

Run from repo root: python src/run_assessment.py

No API keys required; uses only local data and rule-based logic.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def load_conversations():
    path = DATA_DIR / "conversations.json"
    if not path.exists():
        raise FileNotFoundError(f"Expected data at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_compliance_rules():
    path = DATA_DIR / "compliance_rules.json"
    if not path.exists():
        raise FileNotFoundError(f"Expected data at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_compliance(conversation, rules_data):
    """Rule-based check: scan agent messages for rule keywords; report violations."""
    results = []
    agent_text = " ".join(
        m["text"] for m in conversation["messages"] if m["role"] == "agent"
    ).lower()
    for rule in rules_data.get("rules", []):
        for kw in rule.get("keywords") or []:
            if kw.lower() in agent_text:
                results.append(
                    {
                        "rule_id": rule["id"],
                        "category": rule["category"],
                        "severity": rule["severity"],
                        "matched_keyword": kw,
                    }
                )
                break
    return results


def classify_situation(conversation):
    """Simple heuristic: keyword-based classification for product loss vs substandard service vs other."""
    full_text = " ".join(m["text"] for m in conversation["messages"]).lower()
    has_product_loss = any(
        phrase in full_text
        for phrase in [
            "never arrived",
            "never got",
            "never received",
            "charged after i cancelled",
            "refund",
        ]
    )
    has_substandard = any(
        phrase in full_text
        for phrase in [
            "never worked",
            "didn't work",
            "was really slow",
            "bad service",
            "substandard",
            "features never worked",
        ]
    )
    situation_other = not (has_product_loss or has_substandard)
    return {
        "conversation_id": conversation["conversation_id"],
        "has_product_loss": has_product_loss,
        "has_substandard_service": has_substandard,
        "situation_other": situation_other,
        "notes": "heuristic keyword match (reference implementation)",
    }


def main():
    print("Loading data from", DATA_DIR)
    try:
        conversations = load_conversations()
        rules_data = load_compliance_rules()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print("\n--- Compliance check ---\n")
    for conv in conversations:
        violations = check_compliance(conv, rules_data)
        status = "VIOLATIONS" if violations else "OK"
        print(f"{conv['conversation_id']}: {status}")
        for v in violations:
            print(f"  - [{v['severity']}] {v['rule_id']} ({v['category']}): matched '{v['matched_keyword']}'")

    print("\n--- Customer situation classification ---\n")
    for conv in conversations:
        sit = classify_situation(conv)
        flags = []
        if sit["has_product_loss"]:
            flags.append("product_loss")
        if sit["has_substandard_service"]:
            flags.append("substandard_service")
        if sit["situation_other"]:
            flags.append("other")
        print(f"{sit['conversation_id']}: {', '.join(flags)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
