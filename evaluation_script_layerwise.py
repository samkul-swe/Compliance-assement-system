"""
Evaluation script - compares system performance against ground truth.

Uses ground_truth.json generated during conversation creation.
"""

import json
from pathlib import Path

def load_ground_truth():
    """Load ground truth labels."""
    gt_path = Path("data/ground_truth.json")
    if not gt_path.exists():
        print("❌ ground_truth.json not found")
        print("   Run generate_conversations.py first")
        return None
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    # Convert to dict for easy lookup
    return {item['conversation_id']: item['label'] for item in gt_data}

def load_system_results():
    """Load your system's decisions."""
    # Layer 1 auto-decided
    layer1_path = Path("data/layer1_output/auto_decided.json")
    if layer1_path.exists():
        with open(layer1_path, 'r') as f:
            layer1_results = json.load(f)
    else:
        layer1_results = []
    
    # Layer 2 validated
    layer2_path = Path("data/layer2_output/validated_decisions.json")
    if layer2_path.exists():
        with open(layer2_path, 'r') as f:
            layer2_results = json.load(f)
    else:
        layer2_results = []
    
    # Combine
    system_decisions = {}
    
    for result in layer1_results:
        conv_id = result['conversation_id']
        decision = "compliant" if result['compliant'] else "non_compliant"
        system_decisions[conv_id] = {
            'decision': decision,
            'confidence': result['confidence'],
            'layer': 'layer1'
        }
    
    for result in layer2_results:
        conv_id = result['conversation_id']
        system_decisions[conv_id] = {
            'decision': result['final_decision'],
            'confidence': result['avg_confidence'],
            'layer': 'layer2'
        }
    
    return system_decisions


def evaluate():
    """Compare system decisions to ground truth."""
    ground_truth = load_ground_truth()
    if not ground_truth:
        return
    
    system_decisions = load_system_results()
    
    print("="*70)
    print(f"EVALUATION RESULTS ({len(ground_truth)} Conversations)")
    print("="*70)
    
    # Overall counters
    correct = 0
    incorrect = 0
    not_decided = 0
    
    # Layer-specific counters
    layer1_correct = 0
    layer1_incorrect = 0
    layer1_total = 0
    
    layer2_correct = 0
    layer2_incorrect = 0
    layer2_total = 0
    
    results = []
    
    for conv_id, true_label in ground_truth.items():
        if conv_id not in system_decisions:
            print(f"\n⚠️  {conv_id}: System didn't make a decision")
            not_decided += 1
            continue
        
        system = system_decisions[conv_id]
        system_decision = system['decision']
        confidence = system['confidence']
        layer = system['layer']
        
        # Compare
        is_correct = system_decision == true_label
        
        if system_decision == true_label:
            status = "✅ CORRECT"
            correct += 1
            
            # Track by layer
            if layer == 'layer1':
                layer1_correct += 1
                layer1_total += 1
            else:
                layer2_correct += 1
                layer2_total += 1
                
        elif system_decision == "needs_human_review":
            status = "⚠️  NO DECISION (sent to human)"
            not_decided += 1
        else:
            status = "❌ INCORRECT"
            incorrect += 1
            
            # Track by layer
            if layer == 'layer1':
                layer1_incorrect += 1
                layer1_total += 1
            else:
                layer2_incorrect += 1
                layer2_total += 1
        
        results.append({
            'conv_id': conv_id,
            'ground_truth': true_label,
            'system_decision': system_decision,
            'confidence': confidence,
            'layer': layer,
            'status': status
        })
        
        # Only print incorrect or uncertain cases
        if "INCORRECT" in status or "NO DECISION" in status:
            print(f"\n{conv_id}:")
            print(f"  Ground Truth: {true_label.upper()}")
            print(f"  System: {system_decision.upper()} (conf: {confidence:.2f}, {layer})")
            print(f"  {status}")
    
    # Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    total_evaluated = len(ground_truth)
    decided = correct + incorrect
    
    print(f"\nTotal conversations: {total_evaluated}")
    print(f"✅ Correct: {correct}")
    print(f"❌ Incorrect: {incorrect}")
    print(f"⚠️  Sent to human: {not_decided}")
    
    if decided > 0:
        print(f"\n📊 Overall Accuracy: {correct}/{decided} = {correct/decided*100:.1f}%")
    
    # Layer-specific accuracy
    print("\n" + "="*70)
    print("LAYER-SPECIFIC PERFORMANCE")
    print("="*70)
    
    if layer1_total > 0:
        layer1_accuracy = layer1_correct / layer1_total * 100
        print(f"\n🔷 Layer 1 (Semantic SBERT):")
        print(f"   Decisions made: {layer1_total}")
        print(f"   Correct: {layer1_correct}")
        print(f"   Incorrect: {layer1_incorrect}")
        print(f"   Accuracy: {layer1_accuracy:.1f}%")
    
    if layer2_total > 0:
        layer2_accuracy = layer2_correct / layer2_total * 100
        print(f"\n🔶 Layer 2 (Dual LLM):")
        print(f"   Decisions made: {layer2_total}")
        print(f"   Correct: {layer2_correct}")
        print(f"   Incorrect: {layer2_incorrect}")
        print(f"   Accuracy: {layer2_accuracy:.1f}%")
    
    # Breakdown by ground truth label
    print("\n" + "="*70)
    print("BREAKDOWN BY TYPE")
    print("="*70)
    
    compliant_correct = sum(1 for r in results if r['ground_truth'] == 'compliant' and 'CORRECT' in r['status'])
    compliant_total = sum(1 for r in results if r['ground_truth'] == 'compliant' and r['system_decision'] != 'needs_human_review')
    
    non_compliant_correct = sum(1 for r in results if r['ground_truth'] == 'non_compliant' and 'CORRECT' in r['status'])
    non_compliant_total = sum(1 for r in results if r['ground_truth'] == 'non_compliant' and r['system_decision'] != 'needs_human_review')
    
    if compliant_total > 0:
        print(f"\n✅ Compliant cases: {compliant_correct}/{compliant_total} ({compliant_correct/compliant_total*100:.1f}%)")
    if non_compliant_total > 0:
        print(f"❌ Non-compliant cases: {non_compliant_correct}/{non_compliant_total} ({non_compliant_correct/non_compliant_total*100:.1f}%)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("\nEvaluating system against ground truth labels...")
    print("(Ground truth generated during conversation creation)\n")
    evaluate()