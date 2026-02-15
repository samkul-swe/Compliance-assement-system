"""
Layer 1: Complete Pipeline with LLM Queue Generation

This script:
1. Runs semantic compliance checking on all conversations
2. Separates high-confidence decisions from low-confidence cases
3. Generates structured JSON files for Layer 2 (LLM review)
4. Creates summary reports and statistics
"""

import json
import os
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from pathlib import Path

# Import the semantic checker from previous artifact
# In practice, this would be: from layer1_confidence_scorer import SemanticComplianceChecker
# For now, we'll assume it's available


class Layer1Pipeline:
    """
    Complete Layer 1 pipeline that processes all conversations
    and prepares data for Layer 2 LLM review.
    """
    
    def __init__(self, 
                 conversations_file: str = "data/generated_conversations.json",
                 rules_file: str = "data/compliance_rules.json",
                 output_dir: str = "data/layer1_output"):
        """Initialize pipeline."""
        self.conversations_file = conversations_file
        self.rules_file = rules_file
        self.output_dir = output_dir
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("Loading conversations...")
        with open(conversations_file, 'r') as f:
            self.conversations = json.load(f)
        
        print("Loading compliance rules...")
        with open(rules_file, 'r') as f:
            self.rules_data = json.load(f)
        
        # Will be populated by checker
        self.checker = None
        self.results = []
        self.stats = defaultdict(int)
    
    def run(self):
        """Execute the full Layer 1 pipeline."""
        print("\n" + "="*70)
        print("LAYER 1: SEMANTIC COMPLIANCE ANALYSIS")
        print("="*70 + "\n")
        
        # Import and initialize checker
        from layer1_confidence_scorer import SemanticComplianceChecker
        print("Initializing semantic checker...")
        self.checker = SemanticComplianceChecker(self.rules_file)
        
        # Process all conversations
        print(f"\nProcessing {len(self.conversations)} conversations...")
        self.results = []
        
        for i, conv in enumerate(self.conversations, 1):
            if i % 20 == 0:
                print(f"  Processed {i}/{len(self.conversations)}...")
            
            result = self.checker.check_conversation(conv)
            self.results.append(result)
            
            # Update stats
            self._update_stats(result)
        
        print(f"✅ Completed analysis of {len(self.conversations)} conversations\n")
        
        # Generate outputs
        self._generate_outputs()
        
        # Print summary
        self._print_summary()
    
    def _update_stats(self, result):
        """Update statistics."""
        threshold = self.checker.get_threshold(result)
        
        if result.confidence >= threshold:
            if result.compliant:
                self.stats['high_conf_compliant'] += 1
            else:
                self.stats['high_conf_violation'] += 1
                # Track by severity
                for v in result.violations:
                    self.stats[f"violation_{v['severity']}"] += 1
        else:
            self.stats['low_conf_needs_llm'] += 1
            if result.compliant:
                self.stats['llm_queue_compliant_lean'] += 1
            else:
                self.stats['llm_queue_violation_lean'] += 1
    
    def _generate_outputs(self):
        """Generate all output files."""
        print("Generating output files...")
        
        # 1. Complete results (all conversations)
        self._save_complete_results()
        
        # 2. Auto-decided cases (high confidence)
        self._save_auto_decided()
        
        # 3. LLM review queue (low confidence) - MOST IMPORTANT
        self._save_llm_queue()
        
        # 4. Summary statistics
        self._save_statistics()
        
        # 5. Violations report (for high-confidence violations)
        self._save_violations_report()
        
        print("✅ All output files generated\n")
    
    def _save_complete_results(self):
        """Save all results with full details."""
        output = []
        
        for result in self.results:
            output.append({
                "conversation_id": result.conversation_id,
                "decision": "compliant" if result.compliant else "non_compliant",
                "confidence": float(result.confidence),
                "threshold": float(self.checker.get_threshold(result)),
                "needs_llm_review": bool(self.checker.needs_llm_review(result)),
                "violations": self._convert_violations(result.violations),
                "evidence": result.evidence,
                "similarity_scores": self._convert_to_float(result.similarity_scores),
                "notes": result.notes
            })
        
        filepath = os.path.join(self.output_dir, "complete_results.json")
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"  ✓ Saved: {filepath}")
    
    def _save_auto_decided(self):
        """Save high-confidence auto-decided cases."""
        auto_decided = []
        
        for i, result in enumerate(self.results):
            if not self.checker.needs_llm_review(result):
                conv = self.conversations[i]
                auto_decided.append({
                    "conversation_id": result.conversation_id,
                    "decision": "compliant" if result.compliant else "non_compliant",
                    "confidence": float(result.confidence),
                    "decided_by": "layer1_semantic",
                    "violations": self._convert_violations(result.violations) if not result.compliant else [],
                    "timestamp": datetime.now().isoformat()
                })
        
        filepath = os.path.join(self.output_dir, "auto_decided.json")
        with open(filepath, 'w') as f:
            json.dump(auto_decided, f, indent=2)
        
        print(f"  ✓ Saved: {filepath} ({len(auto_decided)} conversations)")
    
    def _save_llm_queue(self):
        """
        Save LLM review queue with ALL information needed for Layer 2.
        
        This is the key handoff file - contains everything the LLM needs:
        - Full conversation (all messages)
        - Layer 1 analysis (confidence, violations, evidence)
        - Compliance rules for context
        - Severity information
        - Similarity scores
        """
        llm_queue = []
        
        for i, result in enumerate(self.results):
            if self.checker.needs_llm_review(result):
                conv = self.conversations[i]
                
                # Build complete context for LLM
                llm_item = {
                    # Identification
                    "conversation_id": result.conversation_id,
                    "queue_position": len(llm_queue) + 1,
                    
                    # Full conversation for LLM to analyze
                    "conversation": {
                        "messages": conv['messages'],
                        "channel": conv.get('channel'),
                        "customer_segment": conv.get('customer_segment'),
                        "metadata": conv.get('metadata', {})
                    },
                    
                    # Layer 1 analysis results
                    "layer1_analysis": {
                        "decision": "compliant" if result.compliant else "non_compliant",
                        "confidence": float(result.confidence),
                        "threshold": float(self.checker.get_threshold(result)),
                        "confidence_gap": float(self.checker.get_threshold(result) - result.confidence),
                        "violations_detected": self._convert_violations(result.violations),
                        "evidence": result.evidence,
                        "similarity_scores": self._convert_to_float(result.similarity_scores),
                        "notes": result.notes
                    },
                    
                    # Severity and priority for LLM routing
                    "priority": self._calculate_priority(result),
                    "max_severity": self._get_max_severity(result),
                    
                    # Compliance rules for context
                    "relevant_rules": self._get_relevant_rules(result),
                    
                    # Flags for LLM attention
                    "flags": {
                        "has_critical_violation": self._has_severity(result, "critical"),
                        "has_high_violation": self._has_severity(result, "high"),
                        "multiple_violations": len(result.violations) > 1,
                        "borderline_confidence": 0.65 <= result.confidence <= 0.75,
                        "very_low_confidence": result.confidence < 0.60
                    },
                    
                    # Timestamp
                    "queued_at": datetime.now().isoformat()
                }
                
                llm_queue.append(llm_item)
        
        # Sort by priority (high priority first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        llm_queue.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        # Update queue positions after sorting
        for i, item in enumerate(llm_queue, 1):
            item["queue_position"] = i
        
        filepath = os.path.join(self.output_dir, "llm_review_queue.json")
        with open(filepath, 'w') as f:
            json.dump(llm_queue, f, indent=2)
        
        print(f"  ✓ Saved: {filepath} ({len(llm_queue)} conversations need LLM review)")
        
        # Also save a simplified version for quick review
        self._save_llm_queue_summary(llm_queue)
    
    def _save_llm_queue_summary(self, llm_queue: List[Dict]):
        """Save a quick-reference summary of LLM queue."""
        summary = []
        
        for item in llm_queue:
            summary.append({
                "position": item["queue_position"],
                "conversation_id": item["conversation_id"],
                "priority": item["priority"],
                "layer1_decision": item["layer1_analysis"]["decision"],
                "confidence": float(item["layer1_analysis"]["confidence"]),
                "threshold": float(item["layer1_analysis"]["threshold"]),
                "gap": float(round(item["layer1_analysis"]["confidence_gap"], 2)),
                "violations": [v["rule_id"] for v in item["layer1_analysis"]["violations_detected"]],
                "max_severity": item["max_severity"]
            })
        
        filepath = os.path.join(self.output_dir, "llm_queue_summary.json")
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Saved: {filepath} (quick reference)")
    
    def _save_statistics(self):
        """Save detailed statistics."""
        stats = {
            "run_timestamp": datetime.now().isoformat(),
            "total_conversations": len(self.conversations),
            "layer1_results": {
                "auto_decided": {
                    "compliant": self.stats['high_conf_compliant'],
                    "violations": self.stats['high_conf_violation'],
                    "total": self.stats['high_conf_compliant'] + self.stats['high_conf_violation']
                },
                "llm_review_needed": {
                    "total": self.stats['low_conf_needs_llm'],
                    "compliant_lean": self.stats['llm_queue_compliant_lean'],
                    "violation_lean": self.stats['llm_queue_violation_lean']
                }
            },
            "automation_rate": round(
                (self.stats['high_conf_compliant'] + self.stats['high_conf_violation']) / 
                len(self.conversations) * 100, 1
            ),
            "llm_review_rate": round(
                self.stats['low_conf_needs_llm'] / len(self.conversations) * 100, 1
            ),
            "violations_by_severity": {
                "critical": self.stats.get('violation_critical', 0),
                "high": self.stats.get('violation_high', 0),
                "medium": self.stats.get('violation_medium', 0),
                "low": self.stats.get('violation_low', 0)
            },
            "estimated_costs": {
                "layer1_cost": 0,  # Semantic model runs locally
                "layer2_llm_calls_needed": self.stats['low_conf_needs_llm'] * 2,  # Dual LLM
                "estimated_layer2_cost": round(self.stats['low_conf_needs_llm'] * 2 * 0.0003, 4)
            }
        }
        
        filepath = os.path.join(self.output_dir, "statistics.json")
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ Saved: {filepath}")
    
    def _save_violations_report(self):
        """Save report of high-confidence violations for immediate action."""
        violations = []
        
        for i, result in enumerate(self.results):
            if not result.compliant and not self.checker.needs_llm_review(result):
                conv = self.conversations[i]
                violations.append({
                    "conversation_id": result.conversation_id,
                    "confidence": float(result.confidence),
                    "violations": self._convert_violations(result.violations),
                    "evidence": result.evidence,
                    "agent_messages": [
                        msg['text'] for msg in conv['messages'] 
                        if msg['role'] == 'agent'
                    ],
                    "customer_segment": conv.get('customer_segment'),
                    "channel": conv.get('channel'),
                    "needs_situation_report": True  # Flag for Layer 2 situation analysis
                })
        
        filepath = os.path.join(self.output_dir, "confirmed_violations.json")
        with open(filepath, 'w') as f:
            json.dump(violations, f, indent=2)
        
        print(f"  ✓ Saved: {filepath} ({len(violations)} confirmed violations)")
    
    def _calculate_priority(self, result) -> str:
        """Calculate priority for LLM review queue."""
        if result.violations:
            max_severity = max(v['severity'] for v in result.violations)
            if max_severity == 'critical':
                return "high"
            elif max_severity == 'high':
                return "medium"
        
        if result.confidence < 0.60:
            return "high"  # Very uncertain
        elif result.confidence < 0.70:
            return "medium"
        else:
            return "low"
    
    def _get_max_severity(self, result) -> str:
        """Get maximum severity from violations, or 'none'."""
        if not result.violations:
            return "none"
        return max(v['severity'] for v in result.violations)
    
    def _has_severity(self, result, severity: str) -> bool:
        """Check if result has violation of given severity."""
        return any(v['severity'] == severity for v in result.violations)
    
    def _get_relevant_rules(self, result) -> List[Dict]:
        """Get full rule definitions for violations detected."""
        if not result.violations:
            # Return all rules for context
            return self.rules_data['rules']
        
        # Return rules that were flagged
        relevant_rule_ids = [v['rule_id'] for v in result.violations]
        return [
            rule for rule in self.rules_data['rules']
            if rule['id'] in relevant_rule_ids
        ]
    
    def _convert_to_float(self, obj):
        """Convert numpy types to Python float for JSON serialization."""
        if isinstance(obj, dict):
            return {k: float(v) if hasattr(v, 'item') else v for k, v in obj.items()}
        elif hasattr(obj, 'item'):
            return float(obj)
        return obj
    
    def _convert_violations(self, violations):
        """Convert violation list with numpy types to JSON-serializable format."""
        result = []
        for v in violations:
            result.append({
                'rule_id': v['rule_id'],
                'severity': v['severity'],
                'category': v['category'],
                'description': v['description'],
                'similarity': float(v.get('similarity', 0)) if 'similarity' in v else None,
                'keyword_match': bool(v.get('keyword_match', False)) if 'keyword_match' in v else None
            })
        return result
    
    def _print_summary(self):
        """Print summary to console."""
        print("="*70)
        print("LAYER 1 COMPLETE - SUMMARY")
        print("="*70)
        
        total = len(self.conversations)
        auto = self.stats['high_conf_compliant'] + self.stats['high_conf_violation']
        llm_needed = self.stats['low_conf_needs_llm']
        
        print(f"\n📊 PROCESSING RESULTS:")
        print(f"  Total conversations analyzed: {total}")
        print(f"  Auto-decided (high confidence): {auto} ({auto/total*100:.1f}%)")
        print(f"    • Compliant: {self.stats['high_conf_compliant']}")
        print(f"    • Violations: {self.stats['high_conf_violation']}")
        print(f"  Needs LLM review (low confidence): {llm_needed} ({llm_needed/total*100:.1f}%)")
        
        print(f"\n⚠️  VIOLATIONS BY SEVERITY:")
        for severity in ['critical', 'high', 'medium', 'low']:
            count = self.stats.get(f'violation_{severity}', 0)
            if count > 0:
                print(f"    {severity.upper()}: {count}")
        
        print(f"\n💰 COST ANALYSIS:")
        print(f"  Layer 1 (semantic): $0 (runs locally)")
        llm_calls = llm_needed * 2
        cost = llm_calls * 0.0003
        print(f"  Layer 2 (LLM): ~{llm_calls} calls, ~${cost:.4f}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  {self.output_dir}/")
        print(f"    ├─ complete_results.json (all {total} conversations)")
        print(f"    ├─ auto_decided.json ({auto} high-confidence)")
        print(f"    ├─ llm_review_queue.json ({llm_needed} for Layer 2) ⭐")
        print(f"    ├─ llm_queue_summary.json (quick reference)")
        print(f"    ├─ confirmed_violations.json ({self.stats['high_conf_violation']} violations)")
        print(f"    └─ statistics.json (detailed stats)")
        
        print(f"\n✅ Layer 1 complete! Ready for Layer 2.")
        print(f"💡 Next: Run Layer 2 on llm_review_queue.json")
        print("="*70 + "\n")


def main():
    """Run the complete Layer 1 pipeline."""
    import sys
    
    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Error: sentence-transformers not installed")
        print("Please run: pip install sentence-transformers scikit-learn")
        sys.exit(1)
    
    # Check data files
    if not os.path.exists("data/generated_conversations.json"):
        print("❌ Error: data/generated_conversations.json not found")
        print("Please run: python generate_conversations.py")
        sys.exit(1)
    
    if not os.path.exists("data/compliance_rules.json"):
        print("❌ Error: data/compliance_rules.json not found")
        sys.exit(1)
    
    # Run pipeline
    pipeline = Layer1Pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()