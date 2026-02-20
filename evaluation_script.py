"""
Evaluation script - compares system performance against ground truth.

Ground truth now has severity labels: compliant/low/medium/high/critical
System still outputs binary: compliant/non_compliant + detected_severity

Two evaluation modes:
    1. Binary accuracy   — does the system correctly identify compliant vs violation?
    2. Severity accuracy — does the detected severity match the ground truth severity?
"""

import json
from pathlib import Path


def load_ground_truth():
    gt_path = Path("data/ground_truth.json")
    if not gt_path.exists():
        print("❌ ground_truth.json not found")
        print("   Run generate_conversations.py first")
        return None
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    return {item['conversation_id']: item['label'] for item in gt_data}


def load_system_results():
    system_decisions = {}

    # Layer 1 auto decided
    l1_path = Path("data/layer1_output/auto_decided.json")
    if l1_path.exists():
        with open(l1_path, 'r') as f:
            for result in json.load(f):
                conv_id = result['conversation_id']
                system_decisions[conv_id] = {
                    'decision': "compliant" if result['compliant'] else "non_compliant",
                    'severity': result.get('detected_severity') or 'compliant',
                    'layer': 'layer1'
                }

    # Layer 2 confirmed severities
    l2_path = Path("data/layer2_output/confirmed_severities.json")
    if l2_path.exists():
        with open(l2_path, 'r') as f:
            for result in json.load(f):
                conv_id = result['conversation_id']
                final_severity = result.get('final_severity', 'needs_human_review')
                system_decisions[conv_id] = {
                    'decision': 'needs_human_review' if final_severity == 'needs_human_review' else 'non_compliant',
                    'severity': final_severity,
                    'layer': 'layer2'
                }

    return system_decisions


def severity_to_binary(label: str) -> str:
    """Convert severity label to binary compliant/non_compliant."""
    return 'compliant' if label == 'compliant' else 'non_compliant'


def evaluate():
    ground_truth = load_ground_truth()
    if not ground_truth:
        return

    system = load_system_results()

    print("=" * 70)
    print(f"EVALUATION RESULTS ({len(ground_truth)} Conversations)")
    print("=" * 70)

    # Binary counters
    binary_correct = binary_incorrect = not_decided = 0
    l1_correct = l1_incorrect = l1_total = 0
    l2_correct = l2_incorrect = l2_total = 0

    # Severity counters
    severity_correct = severity_incorrect = 0

    # Per severity tracking
    severity_levels = ['compliant', 'low', 'medium', 'high', 'critical']
    severity_binary = {s: {'correct': 0, 'total': 0} for s in severity_levels}
    severity_exact  = {s: {'correct': 0, 'total': 0} for s in severity_levels}

    results = []

    for conv_id, true_label in ground_truth.items():
        true_binary = severity_to_binary(true_label)

        if conv_id not in system:
            not_decided += 1
            continue

        result = system[conv_id]
        decision = result['decision']
        detected_severity = result['severity']
        layer = result['layer']

        # ── Binary evaluation ──
        if decision == 'needs_human_review':
            binary_status = "⚠️  NO DECISION"
            not_decided += 1
        elif decision == true_binary:
            binary_status = "✅ CORRECT"
            binary_correct += 1
            if layer == 'layer1':
                l1_correct += 1
                l1_total += 1
            else:
                l2_correct += 1
                l2_total += 1
        else:
            binary_status = "❌ INCORRECT"
            binary_incorrect += 1
            if layer == 'layer1':
                l1_incorrect += 1
                l1_total += 1
            else:
                l2_incorrect += 1
                l2_total += 1

        # ── Severity evaluation (only for decided cases) ──
        severity_status = "-"
        if decision not in ('needs_human_review',) and true_label in severity_levels:
            severity_exact[true_label]['total'] += 1
            if detected_severity == true_label:
                severity_correct += 1
                severity_status = "✅ SEVERITY MATCH"
                severity_exact[true_label]['correct'] += 1
            else:
                severity_incorrect += 1
                severity_status = f"❌ GOT {detected_severity}"

        # Per severity binary tracking
        if true_label in severity_levels and decision != 'needs_human_review':
            severity_binary[true_label]['total'] += 1
            if decision == true_binary:
                severity_binary[true_label]['correct'] += 1

        results.append({
            'conv_id': conv_id,
            'true_label': true_label,
            'true_binary': true_binary,
            'decision': decision,
            'detected_severity': detected_severity,
            'layer': layer,
            'binary_status': binary_status,
            'severity_status': severity_status
        })

        # Print errors only
        if "INCORRECT" in binary_status or "NO DECISION" in binary_status:
            print(f"\n{conv_id}:")
            print(f"  Ground truth:      {true_label.upper()}")
            print(f"  System decision:   {decision.upper()} ({layer})")
            print(f"  Detected severity: {detected_severity}")
            print(f"  Binary:   {binary_status}")
            print(f"  Severity: {severity_status}")

    decided = binary_correct + binary_incorrect

    # ── Overall summary ──
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"\n  Total conversations: {len(ground_truth)}")
    print(f"  ✅ Correct:          {binary_correct}")
    print(f"  ❌ Incorrect:        {binary_incorrect}")
    print(f"  ⚠️  Sent to human:   {not_decided}")

    if decided > 0:
        print(f"\n  📊 Binary Accuracy:   {binary_correct}/{decided} = {binary_correct/decided*100:.1f}%")

    severity_decided = severity_correct + severity_incorrect
    if severity_decided > 0:
        print(f"  🎯 Severity Accuracy: {severity_correct}/{severity_decided} = {severity_correct/severity_decided*100:.1f}%")

    # ── Layer breakdown ──
    print("\n" + "=" * 70)
    print("LAYER-SPECIFIC PERFORMANCE")
    print("=" * 70)

    if l1_total > 0:
        print(f"\n  🔷 Layer 1 (Centroid SBERT):")
        print(f"     Decisions: {l1_total}")
        print(f"     Correct:   {l1_correct}")
        print(f"     Incorrect: {l1_incorrect}")
        print(f"     Accuracy:  {l1_correct/l1_total*100:.1f}%")

    if l2_total > 0:
        print(f"\n  🔶 Layer 2 (Severity Adjudication):")
        print(f"     Decisions: {l2_total}")
        print(f"     Correct:   {l2_correct}")
        print(f"     Incorrect: {l2_incorrect}")
        print(f"     Accuracy:  {l2_correct/l2_total*100:.1f}%")

    # ── Binary accuracy per severity level ──
    print("\n" + "=" * 70)
    print("BINARY ACCURACY BY SEVERITY LEVEL")
    print("=" * 70)
    print(f"\n  {'Severity':10}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
    print(f"  {'-'*40}")
    for sev in severity_levels:
        d = severity_binary[sev]
        if d['total'] > 0:
            acc = d['correct'] / d['total'] * 100
            print(f"  {sev:10}  {d['correct']:>7}  {d['total']:>5}  {acc:>7.1f}%")

    # ── Severity match accuracy ──
    print("\n" + "=" * 70)
    print("SEVERITY DETECTION ACCURACY")
    print("=" * 70)
    print(f"\n  {'Severity':10}  {'Matched':>7}  {'Total':>5}  {'Accuracy':>8}")
    print(f"  {'-'*40}")
    for sev in severity_levels:
        d = severity_exact[sev]
        if d['total'] > 0:
            acc = d['correct'] / d['total'] * 100
            print(f"  {sev:10}  {d['correct']:>7}  {d['total']:>5}  {acc:>7.1f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\nEvaluating system against ground truth labels...")
    print("(Ground truth generated during conversation creation)\n")
    evaluate()