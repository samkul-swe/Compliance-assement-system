"""
Evaluation script - compares system performance against ground truth.
"""

import json
from pathlib import Path

SEVERITY_LEVELS = ['compliant', 'low', 'medium', 'high', 'critical']


def load_ground_truth():
    gt_path = Path("data/ground_truth.json")
    if not gt_path.exists():
        print("❌ ground_truth.json not found")
        return None
    with open(gt_path, 'r') as f:
        data = json.load(f)
    # Store full record not just label — we need violated_rule_id for debugging
    return {
        item['conversation_id']: {
            'label':                    item['label'],
            'violated_rule_id':         item.get('violated_rule_id'),
            'violated_rule_description': item.get('violated_rule_description')
        }
        for item in data
    }


def load_system_results():
    system_decisions = {}

    l1_path = Path("data/layer1_output/auto_decided.json")
    if l1_path.exists():
        with open(l1_path, 'r') as f:
            for result in json.load(f):
                conv_id = result['conversation_id']
                system_decisions[conv_id] = {
                    'decision': "compliant" if result['compliant'] else "non_compliant",
                    'layer': 'layer1',
                    'severity': result.get('detected_severity') or 'compliant'
                }

    l2_path = Path("data/layer2_output/confirmed_severities.json")
    if l2_path.exists():
        with open(l2_path, 'r') as f:
            for result in json.load(f):
                conv_id = result['conversation_id']
                final_severity = result.get('final_severity', 'needs_human_review')
                system_decisions[conv_id] = {
                    'decision': 'needs_human_review' if final_severity == 'needs_human_review' else 'non_compliant',
                    'layer': 'layer2',
                    'severity': final_severity
                }

    return system_decisions


def severity_to_binary(label):
    """
    Map ground truth label to expected system outcome.
    compliant + low → no_action (system should NOT flag)
    medium + high + critical → violation (system SHOULD flag)
    """
    return 'compliant' if label in ('compliant', 'low') else 'non_compliant'


def empty_severity_dict():
    return {s: {'correct': 0, 'total': 0} for s in SEVERITY_LEVELS}


def evaluate():
    ground_truth = load_ground_truth()
    if not ground_truth:
        return

    system = load_system_results()

    print("=" * 70)
    print(f"EVALUATION RESULTS ({len(ground_truth)} Conversations)")
    print("=" * 70)

    # ── Counters — medium+ only for accuracy ──
    binary_correct = binary_incorrect = not_decided = 0
    l1_correct = l1_incorrect = l1_total = 0
    l2_correct = l2_incorrect = l2_total = 0
    severity_correct = severity_incorrect = 0

    # Separate tracking for compliant/low (informational only)
    no_action_flagged   = 0   # compliant/low wrongly flagged
    no_action_passed    = 0   # compliant/low correctly not flagged
    no_action_undecided = 0   # compliant/low sent to human

    severity_binary = empty_severity_dict()
    severity_exact  = empty_severity_dict()
    layer_severity  = {
        'layer1': empty_severity_dict(),
        'layer2': empty_severity_dict()
    }

    results = []

    for conv_id, gt in ground_truth.items():
        true_label    = gt['label']
        true_binary   = severity_to_binary(true_label)
        violated_rule = gt.get('violated_rule_id')
        violated_desc = gt.get('violated_rule_description')

        is_no_action = true_label in ('compliant', 'low')

        if conv_id not in system:
            not_decided += 1
            if is_no_action:
                no_action_undecided += 1
            continue

        result    = system[conv_id]
        decision  = result['decision']
        layer     = result['layer']
        severity  = result.get('severity', '-')

        # ── Handle no_action cases separately ──
        if is_no_action:
            if decision == 'needs_human_review':
                no_action_undecided += 1
            elif decision == 'compliant':
                no_action_passed += 1
            else:
                no_action_flagged += 1

            results.append({
                'conv_id':         conv_id,
                'true_label':      true_label,
                'true_binary':     true_binary,
                'decision':        decision,
                'severity':        severity,
                'layer':           layer,
                'binary_status':   'ℹ️  NO_ACTION_TERRITORY',
                'severity_status': '-'
            })
            continue

        # ── Medium+ accuracy evaluation ──
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

        # ── Severity accuracy (medium+ only) ──
        severity_status = "-"
        if decision != 'needs_human_review' and true_label in SEVERITY_LEVELS:
            severity_exact[true_label]['total'] += 1
            if severity == true_label:
                severity_correct += 1
                severity_status = "✅ SEVERITY MATCH"
                severity_exact[true_label]['correct'] += 1
            else:
                severity_incorrect += 1
                severity_status = f"❌ GOT {severity}"

        if decision != 'needs_human_review' and true_label in SEVERITY_LEVELS:
            severity_binary[true_label]['total'] += 1
            layer_severity[layer][true_label]['total'] += 1
            if decision == true_binary:
                severity_binary[true_label]['correct'] += 1
                layer_severity[layer][true_label]['correct'] += 1

        results.append({
            'conv_id':         conv_id,
            'true_label':      true_label,
            'true_binary':     true_binary,
            'decision':        decision,
            'severity':        severity,
            'layer':           layer,
            'binary_status':   binary_status,
            'severity_status': severity_status
        })

        if "INCORRECT" in binary_status or "NO DECISION" in binary_status:
            print(f"\n{conv_id}:")
            print(f"  Ground truth:      {true_label.upper()}")
            if violated_rule:
                print(f"  Violated rule:     {violated_rule} — {violated_desc}")
            print(f"  System decision:   {decision.upper()} ({layer})")
            print(f"  Detected severity: {severity}")
            print(f"  Binary:   {binary_status}")
            print(f"  Severity: {severity_status}")

    decided = binary_correct + binary_incorrect
    medium_plus_total = sum(
        1 for _, gt in ground_truth.items()
        if gt['label'] not in ('compliant', 'low')
    )

    # ── Overall summary ──
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"\n  Total conversations:   {len(ground_truth)}")
    print(f"\n  📊 Medium+ Accuracy (what matters):")
    print(f"     Total medium+:     {medium_plus_total}")
    print(f"     Decided:           {decided}")
    print(f"     ✅ Correct:        {binary_correct}")
    print(f"     ❌ Incorrect:      {binary_incorrect}")
    print(f"     ⚠️  Undecided:     {not_decided}")
    if decided > 0:
        print(f"     Accuracy:          {binary_correct}/{decided} = {binary_correct/decided*100:.1f}%")

    sev_decided = severity_correct + severity_incorrect
    if sev_decided > 0:
        print(f"     Severity Accuracy: {severity_correct}/{sev_decided} = {severity_correct/sev_decided*100:.1f}%")

    print(f"\n  ℹ️  No-action territory (compliant + low — informational only):")
    no_action_total = no_action_passed + no_action_flagged + no_action_undecided
    print(f"     Total:             {no_action_total}")
    print(f"     Correctly ignored: {no_action_passed}")
    print(f"     Wrongly flagged:   {no_action_flagged}  ← false positives")
    print(f"     Sent to human:     {no_action_undecided}")
    if no_action_total > 0:
        print(f"     False positive rate: {no_action_flagged/no_action_total*100:.1f}%")

    # ── Layer breakdown (medium+ only) ──
    print("\n" + "=" * 70)
    print("LAYER-SPECIFIC PERFORMANCE (Medium+ only)")
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

    # ── Accuracy per severity (medium+ only) ──
    print("\n" + "=" * 70)
    print("BINARY ACCURACY BY SEVERITY (Medium+ only)")
    print("=" * 70)
    print(f"\n  {'Severity':10}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
    print(f"  {'-'*40}")
    for sev in ['medium', 'high', 'critical']:
        d = severity_binary[sev]
        if d['total'] > 0:
            acc = d['correct'] / d['total'] * 100
            print(f"  {sev:10}  {d['correct']:>7}  {d['total']:>5}  {acc:>7.1f}%")

    # ── Severity detection (medium+ only) ──
    print("\n" + "=" * 70)
    print("SEVERITY DETECTION ACCURACY (Medium+ only)")
    print("=" * 70)
    print(f"\n  {'Severity':10}  {'Matched':>7}  {'Total':>5}  {'Accuracy':>8}")
    print(f"  {'-'*40}")
    for sev in ['medium', 'high', 'critical']:
        d = severity_exact[sev]
        if d['total'] > 0:
            acc = d['correct'] / d['total'] * 100
            print(f"  {sev:10}  {d['correct']:>7}  {d['total']:>5}  {acc:>7.1f}%")

    # ── Per layer per severity (medium+ only) ──
    print("\n" + "=" * 70)
    print("BINARY ACCURACY BY LAYER AND SEVERITY (Medium+ only)")
    print("=" * 70)
    for layer_name, label in [('layer1', '🔷 Layer 1'),
                               ('layer2', '🔶 Layer 2')]:
        print(f"\n  {label}:")
        print(f"  {'Severity':10}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
        print(f"  {'-'*40}")
        has_data = False
        for sev in ['medium', 'high', 'critical']:
            d = layer_severity[layer_name][sev]
            if d['total'] > 0:
                has_data = True
                acc = d['correct'] / d['total'] * 100
                flag = "  ⚠️" if acc < 50 else ""
                print(f"  {sev:10}  {d['correct']:>7}  {d['total']:>5}  {acc:>7.1f}%{flag}")
        if not has_data:
            print(f"  No decisions made by this layer")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\nEvaluating system against ground truth labels...\n")
    evaluate()