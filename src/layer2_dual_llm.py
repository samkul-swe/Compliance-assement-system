"""
Layer 2: Dual LLM Validation System (Enhanced)

Features:
- MPF prompt format (Markdown Prompt Format)
- Loads API keys from .env file
- Threaded parallel LLM calls for 2x speed
- Configurable thresholds from config/layer2_config.json
- Comprehensive final_report.json with clear human review items

Usage:
    python src/layer2_dual_llm.py
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


@dataclass
class LLMJudgment:
    """Result from a single LLM judge."""
    compliant: bool
    llm_confidence: float
    violations: List[str]
    evidence: str
    reasoning: str
    situation_analysis: Dict


class PromptLoader:
    """Loads and formats MPF prompt."""
    
    def __init__(self, prompt_file: str = "prompts/compliance_validation.md"):
        """Load MPF prompt template."""
        with open(prompt_file, 'r') as f:
            self.template = f.read()
        print(f"✅ Loaded prompt template: {prompt_file}")
    
    def format_prompt(self, 
                     conversation: Dict,
                     rules: List[Dict],
                     layer1_analysis: Dict) -> str:
        """Format prompt with conversation data."""
        
        # Format messages
        messages_text = ""
        for msg in conversation['messages']:
            role = msg['role'].upper()
            text = msg['text']
            messages_text += f"{role}: {text}\n"
        
        # Format rules
        rules_text = "\n".join([
            f"- **{r['id']}** ({r['severity'].upper()}): {r['description']}"
            for r in rules
        ])
        
        # Format Layer 1 hints
        layer1_text = f"""
- Initial Decision: {layer1_analysis['decision']}
- Confidence: {layer1_analysis['confidence']:.2f}
- Suspected Violations: {', '.join(v['rule_id'] for v in layer1_analysis['violations_detected']) or 'None'}
- Top Similarity Scores: {', '.join(f"{k}:{v:.2f}" for k, v in sorted(layer1_analysis['similarity_scores'].items(), key=lambda x: x[1], reverse=True)[:3])}
"""
        
        # Replace placeholders
        prompt = self.template.replace("{{MESSAGES}}", messages_text)
        prompt = prompt.replace("{{RULES}}", rules_text)
        prompt = prompt.replace("{{LAYER1_ANALYSIS}}", layer1_text)
        
        return prompt


class DualLLMValidator:
    """
    Dual LLM validation with threading and configurable settings.
    """
    
    def __init__(self, config_file: str = "config/layer2_config.json"):
        """Initialize with config."""
        # Load config
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.llm_threshold = self.config['llm_confidence_threshold']
        self.use_threading = self.config['threading']['enabled']
        self.thread_timeout = self.config['threading']['timeout_seconds']
        
        # Load prompt
        self.prompt_loader = PromptLoader()
        
        # Initialize LLM clients
        self.judge1_config = self.config['judge1']
        self.judge2_config = self.config['judge2']
        
        self.judge1_client = self._init_llm_client(self.judge1_config)
        self.judge2_client = self._init_llm_client(self.judge2_config)
        
        if not self.judge1_client or not self.judge2_client:
            raise ValueError("Failed to initialize LLM clients. Check API keys in .env")
        
        print(f"✅ Judge 1: {self.judge1_config['model']}")
        print(f"✅ Judge 2: {self.judge2_config['model']}")
        print(f"✅ Threading: {'Enabled' if self.use_threading else 'Disabled'}")
        print(f"✅ Confidence threshold: {self.llm_threshold}\n")
    
    def _init_llm_client(self, config: Dict):
        """Initialize LLM client based on provider."""
        provider = config['provider']
        
        if provider == 'openai':
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print(f"⚠️  OPENAI_API_KEY not found in .env")
                    return None
                return openai.OpenAI(api_key=api_key)
            except ImportError:
                print("⚠️  openai package not installed (pip install openai)")
                return None
                
        elif provider == 'anthropic':
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    print(f"⚠️  ANTHROPIC_API_KEY not found in .env")
                    return None
                return anthropic.Anthropic(api_key=api_key)
            except ImportError:
                print("⚠️  anthropic package not installed (pip install anthropic)")
                return None
        
        elif provider == 'google':
            try:
                # Try new package first
                try:
                    import google.genai as genai
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        print(f"⚠️  GOOGLE_API_KEY not found in .env")
                        return None
                    client = genai.Client(api_key=api_key)
                    return client
                except ImportError:
                    # Fallback to old package
                    import google.generativeai as genai
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        print(f"⚠️  GOOGLE_API_KEY not found in .env")
                        return None
                    genai.configure(api_key=api_key)
                    return genai
            except ImportError:
                print("⚠️  google-generativeai or google-genai package not installed")
                print("   Run: pip install google-generativeai")
                return None
        
        elif provider == 'groq':
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    print(f"⚠️  GROQ_API_KEY not found in .env")
                    return None
                return Groq(api_key=api_key)
            except ImportError:
                print("⚠️  groq package not installed (pip install groq)")
                return None
        
        return None
    
    def validate_conversation(self, queue_item: Dict) -> Dict:
        """
        Validate conversation with dual LLMs (threaded or sequential).
        """
        # Build prompt (same for both judges)
        prompt = self.prompt_loader.format_prompt(
            conversation=queue_item['conversation'],
            rules=queue_item['relevant_rules'],
            layer1_analysis=queue_item['layer1_analysis']
        )
        
        if self.use_threading:
            # Parallel execution
            judge1_result, judge2_result = self._call_judges_parallel(prompt)
        else:
            # Sequential execution
            judge1_result = self._call_judge(self.judge1_client, self.judge1_config, prompt, "Judge 1")
            judge2_result = self._call_judge(self.judge2_client, self.judge2_config, prompt, "Judge 2")
        
        # Analyze and return
        return self._analyze_judgments(queue_item, judge1_result, judge2_result)
    
    def _call_judges_parallel(self, prompt: str) -> Tuple[Optional[LLMJudgment], Optional[LLMJudgment]]:
        """Call both judges in parallel using threads."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(
                self._call_judge,
                self.judge1_client,
                self.judge1_config,
                prompt,
                "Judge 1"
            )
            future2 = executor.submit(
                self._call_judge,
                self.judge2_client,
                self.judge2_config,
                prompt,
                "Judge 2"
            )
            
            # Wait for both with timeout
            try:
                judge1_result = future1.result(timeout=self.thread_timeout)
                judge2_result = future2.result(timeout=self.thread_timeout)
                return judge1_result, judge2_result
            except Exception as e:
                print(f"  ⚠️  Threading error: {e}")
                return None, None
    
    def _call_judge(self, client, config: Dict, prompt: str, judge_name: str) -> Optional[LLMJudgment]:
        """Call a single judge (OpenAI, Anthropic, Google, or Groq)."""
        if not client:
            return None
        
        try:
            if config['provider'] == 'openai':
                return self._call_openai(client, config, prompt)
            elif config['provider'] == 'anthropic':
                return self._call_anthropic(client, config, prompt)
            elif config['provider'] == 'google':
                return self._call_google(client, config, prompt)
            elif config['provider'] == 'groq':
                return self._call_groq(client, config, prompt)
        except Exception as e:
            print(f"  ⚠️  {judge_name} error: {e}")
            return None
    
    def _call_openai(self, client, config: Dict, prompt: str) -> Optional[LLMJudgment]:
        """Call OpenAI API with JSON mode."""
        try:
            response = client.chat.completions.create(
                model=config['model'],
                messages=[
                    {"role": "system", "content": "You are a compliance analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                response_format={"type": "json_object"}  # Force JSON mode
            )
            
            result_text = response.choices[0].message.content
            return self._parse_llm_response(result_text)
            
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"error: {error_msg}")
            return None
    
    def _call_anthropic(self, client, config: Dict, prompt: str) -> Optional[LLMJudgment]:
        """Call Anthropic API."""
        response = client.messages.create(
            model=config['model'],
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result_text = response.content[0].text
        return self._parse_llm_response(result_text)
    
    def _call_groq(self, client, config: Dict, prompt: str) -> Optional[LLMJudgment]:
        """Call Groq API (uses OpenAI-compatible interface)."""
        response = client.chat.completions.create(
            model=config['model'],
            messages=[
                {"role": "system", "content": "You are a compliance analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )
        
        result_text = response.choices[0].message.content
        return self._parse_llm_response(result_text)
    
    def _call_google(self, client, config: Dict, prompt: str) -> Optional[LLMJudgment]:
        """Call Google Gemini API."""
        try:
            model = client.GenerativeModel(
                model_name=config['model'],
                generation_config={
                    "temperature": config['temperature'],
                    "max_output_tokens": config['max_tokens'],
                },
                # Disable safety filters for compliance analysis
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }
            )
            
            # Add system instruction to prompt
            full_prompt = "You are a compliance analyst. Respond only with valid JSON.\n\n" + prompt
            
            response = model.generate_content(full_prompt)
            
            # Check if response was blocked
            if not response.parts:
                # Try to get reason from candidates
                if response.candidates and len(response.candidates) > 0:
                    finish_reason = response.candidates[0].finish_reason
                    safety_ratings = response.candidates[0].safety_ratings
                    print(f"  ⚠️  Gemini blocked response. Finish reason: {finish_reason}")
                    if safety_ratings:
                        print(f"  ⚠️  Safety ratings: {[(r.category, r.probability) for r in safety_ratings]}")
                return None
            
            result_text = response.text
            return self._parse_llm_response(result_text)
            
        except Exception as e:
            print(f"  ⚠️  Gemini call error: {str(e)[:200]}")
            return None
    
    def _parse_llm_response(self, response_text: str) -> Optional[LLMJudgment]:
        """Parse JSON response from LLM."""
        try:
            # Clean up response
            cleaned = response_text.strip()
            
            # Remove markdown if present
            cleaned = cleaned.replace('```json\n', '').replace('```json', '').replace('```\n', '').replace('```', '').strip()
            
            # Extract JSON
            if '{' in cleaned and '}' in cleaned:
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                cleaned = cleaned[start:end]
            
            # Parse
            data = json.loads(cleaned)
            
            # Validate
            if 'compliant' not in data or 'llm_confidence' not in data:
                return None
            
            return LLMJudgment(
                compliant=bool(data['compliant']),
                llm_confidence=float(data['llm_confidence']),
                violations=data.get('violations', []),
                evidence=data.get('evidence', ''),
                reasoning=data.get('reasoning', ''),
                situation_analysis=data.get('situation_analysis', {})
            )
            
        except Exception:
            # Silent fail - error already printed by caller
            return None
    
    def _try_fix_json(self, text: str) -> Optional[str]:
        """Try to fix common JSON issues, especially from Gemini."""
        # Remove markdown code blocks more aggressively
        text = text.replace('```json\n', '').replace('```json', '').replace('```\n', '').replace('```', '').strip()
        
        # Extract JSON from text
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            text = text[start:end]
        
        # Fix common Gemini issues
        # 1. Missing commas between fields
        import re
        # Add comma after closing quote if followed by opening quote (missing comma)
        text = re.sub(r'"\s*\n\s*"', '",\n  "', text)
        
        # 2. Fix escaped quotes that shouldn't be escaped
        # Gemini sometimes does \\" when it should be \"
        text = text.replace('\\\\"', '\\"')
        
        # 3. Try to fix truncated JSON by adding closing braces
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        
        # 4. Try to fix truncated strings
        open_quotes = text.count('"')
        if open_quotes % 2 != 0:
            # Odd number of quotes - add closing quote before last }
            last_brace = text.rfind('}')
            if last_brace > 0:
                text = text[:last_brace] + '"' + text[last_brace:]
        
        # 5. Fix arrays without closing bracket
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        if open_brackets > close_brackets:
            # Find last occurrence of opening bracket and add closing
            last_open = text.rfind('[')
            next_newline = text.find('\n', last_open)
            if next_newline > 0:
                text = text[:next_newline] + ']' + text[next_newline:]
        
        return text
    
    def _analyze_judgments(self, queue_item: Dict,
                          judge1: Optional[LLMJudgment],
                          judge2: Optional[LLMJudgment]) -> Dict:
        """Analyze judgments and make final decision."""
        
        conv_id = queue_item['conversation_id']
        layer1 = queue_item['layer1_analysis']
        
        if not judge1 or not judge2:
            return {
                "conversation_id": conv_id,
                "status": "error",
                "final_decision": None,
                "confidence": 0.0,
                "judges_agreed": False,
                "human_review_needed": True,
                "reason": "llm_call_failed",
                "layer1_analysis": layer1,
                "judge1": None,
                "judge2": None
            }
        
        # Check agreement
        judges_agree = (judge1.compliant == judge2.compliant)
        avg_confidence = (judge1.llm_confidence + judge2.llm_confidence) / 2
        
        # Decision logic
        if judges_agree and avg_confidence >= self.llm_threshold:
            final_decision = "compliant" if judge1.compliant else "non_compliant"
            human_review = False
            status = "validated"
            reason = "both_judges_agree_confident"
            
        elif judges_agree and avg_confidence < self.llm_threshold:
            final_decision = "uncertain"
            human_review = True
            status = "needs_review"
            reason = "both_judges_agree_but_uncertain"
            
        else:
            final_decision = "uncertain"
            human_review = True
            status = "needs_review"
            reason = "judges_disagree"
        
        return {
            "conversation_id": conv_id,
            "status": status,
            "final_decision": final_decision,
            "confidence": float(avg_confidence),
            "judges_agreed": judges_agree,
            "human_review_needed": human_review,
            "reason": reason,
            
            "judge1": self._judgment_to_dict(judge1),
            "judge2": self._judgment_to_dict(judge2),
            
            "layer1_analysis": layer1,
            "priority": queue_item['priority'],
            
            "validated_at": datetime.now().isoformat()
        }
    
    def _judgment_to_dict(self, judgment: Optional[LLMJudgment]) -> Optional[Dict]:
        """Convert judgment to dict."""
        if not judgment:
            return None
        
        return {
            "compliant": judgment.compliant,
            "confidence": float(judgment.llm_confidence),
            "violations": judgment.violations,
            "evidence": judgment.evidence,
            "reasoning": judgment.reasoning,
            "situation_analysis": judgment.situation_analysis
        }


class Layer2Pipeline:
    """Complete Layer 2 pipeline with enhanced reporting."""
    
    def __init__(self,
                 queue_file: str = "data/layer1_output/llm_review_queue.json",
                 output_dir: str = "data/layer2_output"):
        """Initialize pipeline."""
        self.queue_file = queue_file
        self.output_dir = output_dir
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("Loading LLM review queue from Layer 1...")
        with open(queue_file, 'r') as f:
            self.queue = json.load(f)
        
        print(f"✅ Loaded {len(self.queue)} conversations for validation\n")
        
        self.validator = DualLLMValidator()
        
        self.results = []
        self.stats = {
            'total': len(self.queue),
            'validated_compliant': 0,
            'validated_violation': 0,
            'human_review': 0,
            'judges_agreed': 0,
            'judges_disagreed': 0
        }
    
    def run(self, limit: Optional[int] = None):
        """Run validation pipeline."""
        print("="*70)
        print("LAYER 2: DUAL LLM VALIDATION")
        print("="*70 + "\n")
        
        queue_to_process = self.queue[:limit] if limit else self.queue
        
        print(f"Processing {len(queue_to_process)} conversations with threading...\n")
        
        start_time = time.time()
        
        for i, item in enumerate(queue_to_process, 1):
            conv_id = item['conversation_id']
            priority = item['priority']
            
            print(f"\n[{i}/{len(queue_to_process)}] {conv_id} (priority: {priority})")
            
            result = self.validator.validate_conversation(item)
            self.results.append(result)
            
            self._update_stats(result)
            
            # Small delay between conversations
            if i < len(queue_to_process):
                time.sleep(1.0)
            
            # Rate limiting
            time.sleep(0.2)
        
        elapsed = time.time() - start_time
        print(f"⏱️  Processing time: {elapsed:.1f}s ({elapsed/len(queue_to_process):.1f}s per conversation)\n")
        
        self._generate_outputs()
        self._print_summary()
    
    def _update_stats(self, result: Dict):
        """Update statistics."""
        if result['status'] == 'error':
            self.stats['human_review'] += 1
            return
        
        if result['status'] == 'partial_validation':
            # One judge succeeded
            if not result['human_review_needed']:
                if result['final_decision'] == 'compliant':
                    self.stats['validated_compliant'] += 1
                else:
                    self.stats['validated_violation'] += 1
            else:
                self.stats['human_review'] += 1
            return
        
        # Normal case: both judges responded
        if result['human_review_needed']:
            self.stats['human_review'] += 1
        else:
            if result['final_decision'] == 'compliant':
                self.stats['validated_compliant'] += 1
            else:
                self.stats['validated_violation'] += 1
        
        if result['judges_agreed']:
            self.stats['judges_agreed'] += 1
        else:
            self.stats['judges_disagreed'] += 1
    
    def _print_final_decision(self, result: Dict):
        """Print clean final decision."""
        if result['status'] == 'error':
            print(f"  ⚠️  FINAL: Both judges failed → HUMAN REVIEW\n")
            return
        
        if result['status'] == 'partial_validation':
            which = "J1" if result.get('judge1') else "J2"
            conf = result['confidence']
            if result['human_review_needed']:
                print(f"  ⚠️  FINAL: Only {which} succeeded (conf: {conf:.2f}) → HUMAN REVIEW\n")
            else:
                decision = "COMPLIANT" if result['final_decision'] == "compliant" else "VIOLATION"
                print(f"  ✓ FINAL: {decision} (conf: {conf:.2f}, {which} only)\n")
            return
        
        # Both succeeded
        if result['judges_agreed']:
            decision = "COMPLIANT" if result['final_decision'] == "compliant" else "VIOLATION"
            conf = result['confidence']
            if result['human_review_needed']:
                print(f"  ⚠️  FINAL: Both agree on {decision} but LOW confidence ({conf:.2f}) → HUMAN REVIEW\n")
            else:
                print(f"  ✅ FINAL: {decision} (confidence: {conf:.2f}, both judges agree)\n")
        else:
            print(f"  ⚠️  FINAL: JUDGES DISAGREE → HUMAN REVIEW\n")
    
    def _generate_outputs(self):
        """Generate all output files."""
        print("Generating output files...")
        
        # 1. Complete results
        filepath = os.path.join(self.output_dir, "validation_results.json")
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  ✓ {filepath}")
        
        # 2. Validated decisions
        validated = [r for r in self.results if not r['human_review_needed']]
        filepath = os.path.join(self.output_dir, "validated_decisions.json")
        with open(filepath, 'w') as f:
            json.dump(validated, f, indent=2)
        print(f"  ✓ {filepath} ({len(validated)} validated)")
        
        # 3. Human review queue
        human_review = [r for r in self.results if r['human_review_needed']]
        filepath = os.path.join(self.output_dir, "human_review_queue.json")
        with open(filepath, 'w') as f:
            json.dump(human_review, f, indent=2)
        print(f"  ✓ {filepath} ({len(human_review)} need review)")
        
        # 4. FINAL REPORT (comprehensive)
        self._generate_final_report()
        
        # 5. Statistics
        self._save_statistics()
    
    def _generate_final_report(self):
        """Generate comprehensive final report."""
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "summary": {
                "total_conversations_analyzed": len(self.results),
                "automated_decisions": self.stats['validated_compliant'] + self.stats['validated_violation'],
                "human_review_required": self.stats['human_review'],
                "automation_rate": round((1 - self.stats['human_review']/len(self.results)) * 100, 1) if self.results else 0
            },
            
            "compliant_conversations": {
                "count": self.stats['validated_compliant'],
                "description": "Conversations where LLM judges agreed the agent was compliant",
                "conversations": [
                    {
                        "conversation_id": r['conversation_id'],
                        "confidence": r['confidence'],
                        "decided_by": "both_judges" if r.get('judge1') and r.get('judge2') else "single_judge",
                        "judge1_reasoning": r['judge1']['reasoning'][:200] + "..." if r.get('judge1') else None,
                        "judge2_reasoning": r['judge2']['reasoning'][:200] + "..." if r.get('judge2') else None
                    }
                    for r in self.results
                    if r['final_decision'] == 'compliant' and not r['human_review_needed'] and r.get('status') != 'error'
                ]
            },
            
            "non_compliant_conversations": {
                "count": self.stats['validated_violation'],
                "description": "Conversations where LLM judges agreed there were violations",
                "conversations": [
                    {
                        "conversation_id": r['conversation_id'],
                        "confidence": r['confidence'],
                        "decided_by": "both_judges" if r.get('judge1') and r.get('judge2') else "single_judge",
                        "violations": r['judge1']['violations'] if r.get('judge1') else r['judge2']['violations'] if r.get('judge2') else [],
                        "evidence": r['judge1']['evidence'] if r.get('judge1') else r['judge2']['evidence'] if r.get('judge2') else "",
                        "situation": r['judge1']['situation_analysis'] if r.get('judge1') else r['judge2']['situation_analysis'] if r.get('judge2') else {},
                        "judge1_reasoning": r['judge1']['reasoning'][:200] + "..." if r.get('judge1') else None,
                        "judge2_reasoning": r['judge2']['reasoning'][:200] + "..." if r.get('judge2') else None
                    }
                    for r in self.results
                    if r['final_decision'] == 'non_compliant' and not r['human_review_needed'] and r.get('status') != 'error'
                ]
            },
            
            "requires_human_review": {
                "count": self.stats['human_review'],
                "description": "Cases where LLM judges disagreed or were uncertain - human decision needed",
                "conversations": [
                    {
                        "conversation_id": r['conversation_id'],
                        "priority": r.get('priority', 'unknown'),
                        "reason": r['reason'],
                        "judges_agreed": r['judges_agreed'],
                        "average_confidence": r['confidence'],
                        
                        "judge1_says": {
                            "decision": "compliant" if r['judge1']['compliant'] else "non_compliant",
                            "confidence": r['judge1']['confidence'],
                            "violations": r['judge1']['violations'],
                            "evidence": r['judge1']['evidence'],
                            "reasoning": r['judge1']['reasoning']
                        } if r.get('judge1') else None,
                        
                        "judge2_says": {
                            "decision": "compliant" if r['judge2']['compliant'] else "non_compliant",
                            "confidence": r['judge2']['confidence'],
                            "violations": r['judge2']['violations'],
                            "evidence": r['judge2']['evidence'],
                            "reasoning": r['judge2']['reasoning']
                        } if r.get('judge2') else None,
                        
                        "layer1_hint": {
                            "decision": r['layer1_analysis']['decision'],
                            "confidence": r['layer1_analysis']['confidence'],
                            "violations": [v['rule_id'] for v in r['layer1_analysis']['violations_detected']]
                        } if r.get('layer1_analysis') else None,
                        
                        "recommendation": self._generate_human_review_recommendation(r)
                    }
                    for r in self.results
                    if r['human_review_needed']
                ]
            }
        }
        
        filepath = os.path.join(self.output_dir, "final_report.json")
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  ✓ {filepath} ⭐ (COMPREHENSIVE REPORT)")
    
    def _generate_human_review_recommendation(self, result: Dict) -> str:
        """Generate recommendation for human reviewer."""
        # Handle error cases
        if result.get('status') == 'error':
            return "ERROR: LLM call failed. Manual review required."
        
        # Handle missing judge data
        if not result.get('judge1') or not result.get('judge2'):
            return "ERROR: One or both LLM judges failed to respond. Manual review required."
        
        if not result['judges_agreed']:
            j1_violations = result['judge1'].get('violations', [])
            j2_violations = result['judge2'].get('violations', [])
            return f"PRIORITY: Judges fundamentally disagree. Judge 1 says {j1_violations or 'compliant'}, Judge 2 says {j2_violations or 'compliant'}. Review conversation carefully."
        else:
            return f"Both judges agree on {result['final_decision']} but confidence is low ({result['confidence']:.2f}). This is a borderline case requiring human judgment."
    
    def _save_statistics(self):
        """Save statistics."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_processed": len(self.results),
            
            "breakdown": {
                "validated_compliant": self.stats['validated_compliant'],
                "validated_violations": self.stats['validated_violation'],
                "human_review_needed": self.stats['human_review']
            },
            
            "judge_agreement": {
                "agreed": self.stats['judges_agreed'],
                "disagreed": self.stats['judges_disagreed'],
                "agreement_rate": round(self.stats['judges_agreed'] / len(self.results) * 100, 1) if self.results else 0
            },
            
            "automation_rate": round((self.stats['validated_compliant'] + self.stats['validated_violation']) / len(self.results) * 100, 1) if self.results else 0
        }
        
        filepath = os.path.join(self.output_dir, "layer2_statistics.json")
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ {filepath}")
    
    def _print_summary(self):
        """Print summary."""
        print("\n" + "="*70)
        print("LAYER 2 COMPLETE - SUMMARY")
        print("="*70)
        
        total = len(self.results)
        
        print(f"\n📊 RESULTS:")
        print(f"  Total processed: {total}")
        print(f"  ✅ Validated compliant: {self.stats['validated_compliant']}")
        print(f"  ❌ Validated violations: {self.stats['validated_violation']}")
        print(f"  👤 Human review needed: {self.stats['human_review']}")
        
        print(f"\n🤝 JUDGE AGREEMENT:")
        print(f"  Agreed: {self.stats['judges_agreed']} ({self.stats['judges_agreed']/total*100:.1f}%)")
        print(f"  Disagreed: {self.stats['judges_disagreed']} ({self.stats['judges_disagreed']/total*100:.1f}%)")
        
        automation = (self.stats['validated_compliant'] + self.stats['validated_violation']) / total * 100
        print(f"\n⚡ AUTOMATION:")
        print(f"  {automation:.1f}% automated by dual LLM validation")
        print(f"  {100-automation:.1f}% require human review")
        
        print(f"\n📁 KEY OUTPUTS:")
        print(f"  Machine-readable (JSON):")
        print(f"    • validation_results.json - All results")
        print(f"    • validated_decisions.json - Auto-decided cases")
        print(f"    • layer2_statistics.json - Metrics")
        print(f"  Human-readable (Excel):")
        print(f"    • final_report.xlsx - Summary + all decisions ⭐")
        print(f"    • human_review.xlsx - Cases needing review")
        
        print("="*70 + "\n")


def main():
    """Run Layer 2."""
    import sys
    
    # Check dependencies
    required_files = [
        "data/layer1_output/llm_review_queue.json",
        "prompts/compliance_validation.md",
        "config/layer2_config.json",
        ".env"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing: {file}")
            if file == ".env":
                print("   Create .env file with: OPENAI_API_KEY=sk-... and/or ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)
    
    # Run pipeline
    pipeline = Layer2Pipeline()
    
    # For testing: limit to 5 conversations
    # For full run: remove limit parameter
    pipeline.run(limit=5)


if __name__ == "__main__":
    main()