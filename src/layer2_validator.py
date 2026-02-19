"""
Layer 2: Dual LLM Validation System

Takes low-confidence cases from Layer 1 and validates with two LLMs.
Tracks token usage and costs.

Features:
- Sequential LLM calls (no threading)
- Configurable LLMs from JSON
- Prompt from Markdown file
- Cost tracking
- Excel output for human review cases
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class LLMJudgment:
    """Result from a single LLM judge."""
    compliant: bool
    confidence: float
    violations: List[str]
    evidence: str
    reasoning: str
    has_product_loss: bool
    has_substandard_service: bool
    situation_other: bool
    tokens_used: int


@dataclass
class ValidationResult:
    """Final validation result for one conversation."""
    conversation_id: str
    judge1_decision: str  # "compliant" or "non_compliant"
    judge1_confidence: float
    judge1_violations: List[str]
    judge1_reasoning: str
    judge1_tokens: int
    judge2_decision: str
    judge2_confidence: float
    judge2_violations: List[str]
    judge2_reasoning: str
    judge2_tokens: int
    judges_agree: bool
    final_decision: str  # "compliant", "non_compliant", or "needs_human_review"
    avg_confidence: float
    situation_product_loss: bool
    situation_substandard_service: bool
    situation_other: bool
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'conversation_id': self.conversation_id,
            'judge1': {
                'decision': self.judge1_decision,
                'confidence': self.judge1_confidence,
                'violations': self.judge1_violations,
                'reasoning': self.judge1_reasoning,
                'tokens': self.judge1_tokens
            },
            'judge2': {
                'decision': self.judge2_decision,
                'confidence': self.judge2_confidence,
                'violations': self.judge2_violations,
                'reasoning': self.judge2_reasoning,
                'tokens': self.judge2_tokens
            },
            'judges_agree': self.judges_agree,
            'final_decision': self.final_decision,
            'avg_confidence': self.avg_confidence,
            'situation': {
                'has_product_loss': self.situation_product_loss,
                'has_substandard_service': self.situation_substandard_service,
                'situation_other': self.situation_other
            },
            'timestamp': self.timestamp
        }


class ConfigLoader:
    """Load Layer 2 configuration."""
    
    @staticmethod
    def load_llm_config(config_file: str = "config/llm_config.json") -> Dict:
        """Load LLM configuration."""
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"LLM config not found: {config_file}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_prompt(prompt_file: str = "prompts/compliance_validation.md") -> str:
        """Load prompt template from Markdown."""
        path = Path(prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_compliance_rules(rules_file: str = "data/compliance_rules.json") -> Dict:
        """Load compliance rules."""
        with open(rules_file, 'r', encoding='utf-8') as f:
            return json.load(f)


class LLMClient:
    """Wrapper for different LLM providers."""
    
    def __init__(self, provider: str, model: str, temperature: float, max_tokens: int, system_prompt: str = None):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or "You are a compliance analyst. Respond only with valid JSON."
        self.client = self._init_client()
    
    def _init_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in .env")
                return openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("pip install openai")
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in .env")
                return anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("pip install anthropic")
        
        elif self.provider == "google":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found in .env")
                genai.configure(api_key=api_key)
                return genai
            except ImportError:
                raise ImportError("pip install google-generativeai")
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def call(self, prompt: str) -> Tuple[str, int]:
        """
        Call the LLM with prompt.
        Returns: (response_text, tokens_used)
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            tokens = response.usage.total_tokens
            return response.choices[0].message.content, tokens
        
        elif self.provider == "anthropic":
            # Anthropic doesn't have system role, prepend to prompt
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text, tokens
        
        elif self.provider == "google":
            # Google uses system instruction differently
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            model = self.client.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            response = model.generate_content(full_prompt)
            # Google doesn't provide token counts easily, estimate
            tokens = len(prompt.split()) * 1.3 + len(response.text.split()) * 1.3
            return response.text, int(tokens)


class PromptFormatter:
    """Format prompts with conversation data."""
    
    def __init__(self, template: str, rules_data: Dict):
        self.template = template
        self.rules_data = rules_data
    
    def format(self, conversation: Dict, layer1_data: Dict) -> str:
        """Format prompt with conversation and Layer 1 data."""
        # Format messages
        messages_text = ""
        for msg in conversation['messages']:
            role = msg['role'].upper()
            text = msg['text']
            messages_text += f"{role}: {text}\n"
        
        # Format rules
        rules_text = ""
        for rule in self.rules_data['rules']:
            rules_text += f"- **{rule['id']}** ({rule['severity'].upper()}): {rule['description']}\n"
        
        # Format Layer 1 hints
        layer1_text = f"""
- Layer 1 Decision: {layer1_data.get('compliant', 'unknown')}
- Layer 1 Confidence: {layer1_data.get('confidence', 0):.2f}
- Suspected Violations: {', '.join([v['rule_id'] for v in layer1_data.get('violations', [])])}
"""
        
        # Replace placeholders
        prompt = self.template.replace("{{MESSAGES}}", messages_text)
        prompt = prompt.replace("{{RULES}}", rules_text)
        prompt = prompt.replace("{{LAYER1_HINTS}}", layer1_text)
        
        return prompt


class CostTracker:
    """Track token usage and costs."""
    
    # Cost per 1M tokens (as of 2024)
    COSTS = {
        'gpt-4o-mini': {'input': 0.150, 'output': 0.600},
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
        'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
        'gemini-1.5-flash': {'input': 0.075, 'output': 0.30},
        'gemini-1.5-pro': {'input': 1.25, 'output': 5.00}
    }
    
    def __init__(self):
        self.judge1_tokens = 0
        self.judge2_tokens = 0
        self.judge1_model = None
        self.judge2_model = None
        self.per_conversation_costs = []  # Track individual conversation costs
    
    def add_judge1(self, tokens: int, model: str):
        self.judge1_tokens += tokens
        self.judge1_model = model
    
    def add_judge2(self, tokens: int, model: str):
        self.judge2_tokens += tokens
        self.judge2_model = model
    
    def add_conversation_cost(self, conv_id: str, judge1_tokens: int, judge2_tokens: int):
        """Track cost for individual conversation."""
        judge1_cost = self.get_cost(self.judge1_model, judge1_tokens)
        judge2_cost = self.get_cost(self.judge2_model, judge2_tokens)
        
        self.per_conversation_costs.append({
            'conversation_id': conv_id,
            'judge1_tokens': judge1_tokens,
            'judge1_cost': judge1_cost,
            'judge2_tokens': judge2_tokens,
            'judge2_cost': judge2_cost,
            'total_cost': judge1_cost + judge2_cost
        })
    
    def get_cost(self, model: str, tokens: int) -> float:
        """Calculate cost for given model and tokens."""
        if model not in self.COSTS:
            # Unknown model, estimate
            return tokens * (1.0 / 1_000_000)  # $1 per 1M tokens estimate
        
        # Rough split: 60% input, 40% output
        costs = self.COSTS[model]
        input_tokens = tokens * 0.6
        output_tokens = tokens * 0.4
        
        cost = (input_tokens * costs['input'] / 1_000_000) + \
               (output_tokens * costs['output'] / 1_000_000)
        
        return cost
    
    def get_summary(self) -> Dict:
        """Get cost summary."""
        judge1_cost = self.get_cost(self.judge1_model, self.judge1_tokens)
        judge2_cost = self.get_cost(self.judge2_model, self.judge2_tokens)
        
        return {
            'judge1': {
                'model': self.judge1_model,
                'tokens': self.judge1_tokens,
                'cost_usd': round(judge1_cost, 4)
            },
            'judge2': {
                'model': self.judge2_model,
                'tokens': self.judge2_tokens,
                'cost_usd': round(judge2_cost, 4)
            },
            'total': {
                'tokens': self.judge1_tokens + self.judge2_tokens,
                'cost_usd': round(judge1_cost + judge2_cost, 4)
            },
            'per_conversation_breakdown': self.per_conversation_costs
        }


class DualLLMValidator:
    """Validate conversations with two LLMs sequentially."""
    
    def __init__(self, config: Dict, prompt_template: str, rules_data: Dict):
        self.config = config
        self.rules_data = rules_data
        self.prompt_formatter = PromptFormatter(prompt_template, rules_data)
        self.cost_tracker = CostTracker()
        
        # Initialize LLM clients
        judge1_cfg = config['judge1']
        judge2_cfg = config['judge2']
        
        self.judge1 = LLMClient(
            provider=judge1_cfg['provider'],
            model=judge1_cfg['model'],
            temperature=judge1_cfg['temperature'],
            max_tokens=judge1_cfg['max_tokens'],
            system_prompt=judge1_cfg.get('system_prompt')
        )
        
        self.judge2 = LLMClient(
            provider=judge2_cfg['provider'],
            model=judge2_cfg['model'],
            temperature=judge2_cfg['temperature'],
            max_tokens=judge2_cfg['max_tokens'],
            system_prompt=judge2_cfg.get('system_prompt')
        )
        
        logger.info(f"Judge 1: {judge1_cfg['model']} (Strict Compliance)")
        logger.info(f"Judge 2: {judge2_cfg['model']} (Empathetic Compliance)")
    
    def validate_conversation(
        self, 
        conversation: Dict, 
        layer1_data: Dict
    ) -> ValidationResult:
        """Validate one conversation with both judges."""
        conv_id = conversation['conversation_id']
        
        # Format prompt
        prompt = self.prompt_formatter.format(conversation, layer1_data)
        
        # Call Judge 1
        logger.info(f"  Judge 1 evaluating {conv_id}...")
        judge1_result = self._call_judge(self.judge1, prompt, conv_id)
        if judge1_result:
            judge1_cost = self.cost_tracker.get_cost(self.judge1.model, judge1_result.tokens_used)
            self.cost_tracker.add_judge1(judge1_result.tokens_used, self.judge1.model)
            decision1 = "COMPLIANT" if judge1_result.compliant else "NON-COMPLIANT"
            print(f"    → Decision: {decision1}, Confidence: {judge1_result.confidence:.2f}, Cost: ${judge1_cost:.4f}")
        else:
            print(f"    → FAILED")
        
        # Small delay between calls
        time.sleep(0.5)
        
        # Call Judge 2
        logger.info(f"  Judge 2 evaluating {conv_id}...")
        judge2_result = self._call_judge(self.judge2, prompt, conv_id)
        if judge2_result:
            judge2_cost = self.cost_tracker.get_cost(self.judge2.model, judge2_result.tokens_used)
            self.cost_tracker.add_judge2(judge2_result.tokens_used, self.judge2.model)
            decision2 = "COMPLIANT" if judge2_result.compliant else "NON-COMPLIANT"
            print(f"    → Decision: {decision2}, Confidence: {judge2_result.confidence:.2f}, Cost: ${judge2_cost:.4f}")
        else:
            print(f"    → FAILED")
        
        # Analyze results
        return self._make_final_decision(conv_id, judge1_result, judge2_result)
    
    def _call_judge(self, judge: LLMClient, prompt: str, conv_id: str) -> Optional[LLMJudgment]:
        """Call a single judge."""
        try:
            response_text, tokens = judge.call(prompt)
            
            # Parse JSON response
            data = self._parse_json(response_text)
            if not data:
                logger.error(f"    Failed to parse JSON for {conv_id}")
                return None
            
            return LLMJudgment(
                compliant=data.get('compliant', False),
                confidence=float(data.get('llm_confidence', 0.0)),
                violations=data.get('violations', []),
                evidence=data.get('evidence', ''),
                reasoning=data.get('reasoning', ''),
                has_product_loss=data.get('situation_analysis', {}).get('has_product_loss', False),
                has_substandard_service=data.get('situation_analysis', {}).get('has_substandard_service', False),
                situation_other=data.get('situation_analysis', {}).get('situation_other', False),
                tokens_used=tokens
            )
        
        except Exception as e:
            logger.error(f"    Error calling judge: {str(e)[:100]}")
            return None
    
    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        # Remove markdown fences
        text = text.replace('```json', '').replace('```', '').strip()
        
        # Extract JSON if embedded in text
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            text = text[start:end]
        
        try:
            return json.loads(text)
        except:
            return None
    
    def _make_final_decision(
        self, 
        conv_id: str,
        judge1: Optional[LLMJudgment],
        judge2: Optional[LLMJudgment]
    ) -> ValidationResult:
        """Make final decision based on both judges."""
        # Handle missing judges
        if not judge1 or not judge2:
            return self._create_error_result(conv_id, judge1, judge2)
        
        # Check agreement
        judges_agree = (judge1.compliant == judge2.compliant)
        avg_confidence = (judge1.confidence + judge2.confidence) / 2
        
        # Decision logic
        if judges_agree and avg_confidence >= self.config['agreement_threshold']:
            final_decision = "compliant" if judge1.compliant else "non_compliant"
        else:
            final_decision = "needs_human_review"
        
        # Situation (use judge1 or combine)
        situation_product_loss = judge1.has_product_loss or judge2.has_product_loss
        situation_substandard = judge1.has_substandard_service or judge2.has_substandard_service
        situation_other = judge1.situation_other and judge2.situation_other
        
        return ValidationResult(
            conversation_id=conv_id,
            judge1_decision="compliant" if judge1.compliant else "non_compliant",
            judge1_confidence=judge1.confidence,
            judge1_violations=judge1.violations,
            judge1_reasoning=judge1.reasoning,
            judge1_tokens=judge1.tokens_used,
            judge2_decision="compliant" if judge2.compliant else "non_compliant",
            judge2_confidence=judge2.confidence,
            judge2_violations=judge2.violations,
            judge2_reasoning=judge2.reasoning,
            judge2_tokens=judge2.tokens_used,
            judges_agree=judges_agree,
            final_decision=final_decision,
            avg_confidence=avg_confidence,
            situation_product_loss=situation_product_loss,
            situation_substandard_service=situation_substandard,
            situation_other=situation_other,
            timestamp=datetime.now().isoformat()
        )
    
    def _create_error_result(self, conv_id: str, judge1, judge2) -> ValidationResult:
        """Create result when one or both judges failed."""
        return ValidationResult(
            conversation_id=conv_id,
            judge1_decision="error" if not judge1 else ("compliant" if judge1.compliant else "non_compliant"),
            judge1_confidence=judge1.confidence if judge1 else 0.0,
            judge1_violations=judge1.violations if judge1 else [],
            judge1_reasoning=judge1.reasoning if judge1 else "Failed to get response",
            judge1_tokens=judge1.tokens_used if judge1 else 0,
            judge2_decision="error" if not judge2 else ("compliant" if judge2.compliant else "non_compliant"),
            judge2_confidence=judge2.confidence if judge2 else 0.0,
            judge2_violations=judge2.violations if judge2 else [],
            judge2_reasoning=judge2.reasoning if judge2 else "Failed to get response",
            judge2_tokens=judge2.tokens_used if judge2 else 0,
            judges_agree=False,
            final_decision="needs_human_review",
            avg_confidence=0.0,
            situation_product_loss=False,
            situation_substandard_service=False,
            situation_other=True,
            timestamp=datetime.now().isoformat()
        )


class OutputGenerator:
    """Generate Layer 2 outputs."""
    
    def __init__(self, output_dir: str = "data/layer2_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_all(self, results: List[ValidationResult], cost_summary: Dict):
        """Save all outputs."""
        logger.info("\nGenerating outputs...")
        
        # 1. Validated decisions (JSON for final report)
        validated = [r for r in results if r.final_decision != "needs_human_review"]
        self._save_json(validated, "validated_decisions.json")
        
        # 2. Human review needed (Excel only - for humans)
        human_review = [r for r in results if r.final_decision == "needs_human_review"]
        if human_review:
            self._save_excel(human_review, "human_review_needed.xlsx")
        
        # 3. All results (Excel for human review of everything)
        self._save_excel_all(results, "layer2_all_results.xlsx")
        
        # 4. Cost report (JSON)
        self._save_cost_report(cost_summary, len(results))
        
        logger.info("✅ All outputs generated")
    
    def _save_json(self, results: List[ValidationResult], filename: str):
        """Save to JSON."""
        filepath = self.output_dir / filename
        data = [r.to_dict() for r in results]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"  ✓ {filepath} ({len(results)} records)")
    
    def _save_excel_all(self, results: List[ValidationResult], filename: str):
        """Save all results to Excel for complete human review."""
        filepath = self.output_dir / filename
        
        rows = []
        for r in results:
            rows.append({
                'Conversation ID': r.conversation_id,
                'Final Decision': r.final_decision.replace('_', ' ').upper(),
                'Judges Agree': 'YES' if r.judges_agree else 'NO',
                'Avg Confidence': f"{r.avg_confidence:.0%}",
                'Judge 1 Says': r.judge1_decision.upper(),
                'Judge 1 Conf': f"{r.judge1_confidence:.0%}",
                'Judge 1 Violations': ', '.join(r.judge1_violations) if r.judge1_violations else '-',
                'Judge 2 Says': r.judge2_decision.upper(),
                'Judge 2 Conf': f"{r.judge2_confidence:.0%}",
                'Judge 2 Violations': ', '.join(r.judge2_violations) if r.judge2_violations else '-',
                'Product Loss': 'YES' if r.situation_product_loss else 'NO',
                'Substandard Service': 'YES' if r.situation_substandard_service else 'NO',
                'Other Situation': 'YES' if r.situation_other else 'NO'
            })
        
        df = pd.DataFrame(rows)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All Results', index=False)
            
            worksheet = writer.sheets['All Results']
            for column in worksheet.columns:
                max_length = max(len(str(cell.value)) for cell in column)
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        logger.info(f"  ✓ {filepath} (Excel - all results)")

    
    def _save_excel(self, results: List[ValidationResult], filename: str):
        """Save subset of results to Excel (used for human review queue)."""
        filepath = self.output_dir / filename

        rows = []
        for r in results:
            rows.append({
                'Conversation ID': r.conversation_id,
                'Final Decision': r.final_decision.replace('_', ' ').upper(),
                'Judges Agree': 'YES' if r.judges_agree else 'NO',
                'Avg Confidence': f"{r.avg_confidence:.0%}",
                'Judge 1 Says': r.judge1_decision.upper(),
                'Judge 1 Conf': f"{r.judge1_confidence:.0%}",
                'Judge 1 Violations': ', '.join(r.judge1_violations) if r.judge1_violations else '-',
                'Judge 1 Reasoning': r.judge1_reasoning[:200] if r.judge1_reasoning else '-',
                'Judge 2 Says': r.judge2_decision.upper(),
                'Judge 2 Conf': f"{r.judge2_confidence:.0%}",
                'Judge 2 Violations': ', '.join(r.judge2_violations) if r.judge2_violations else '-',
                'Judge 2 Reasoning': r.judge2_reasoning[:200] if r.judge2_reasoning else '-',
                'Product Loss': 'YES' if r.situation_product_loss else 'NO',
                'Substandard Service': 'YES' if r.situation_substandard_service else 'NO',
                'Other Situation': 'YES' if r.situation_other else 'NO'
            })

        df = pd.DataFrame(rows)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Human Review', index=False)

            worksheet = writer.sheets['Human Review']
            for column in worksheet.columns:
                max_length = max(len(str(cell.value or '')) for cell in column)
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

        logger.info(f"  ✓ {filepath} ({len(results)} records for human review)")
    
    def _save_cost_report(self, cost_summary: Dict, total_convs: int):
        """Save cost report."""
        filepath = self.output_dir / "cost_report.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'conversations_processed': total_convs,
            'judge1': cost_summary['judge1'],
            'judge2': cost_summary['judge2'],
            'total': cost_summary['total'],
            'cost_per_conversation': round(cost_summary['total']['cost_usd'] / total_convs, 4) if total_convs > 0 else 0
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"  ✓ {filepath}")


def main():
    """Run Layer 2 validation."""
    print("="*70)
    print("LAYER 2: DUAL LLM VALIDATION")
    print("="*70 + "\n")
    
    # Load configuration
    logger.info("Loading configuration...")
    config = ConfigLoader.load_llm_config()
    prompt_template = ConfigLoader.load_prompt()
    rules_data = ConfigLoader.load_compliance_rules()
    
    # Load Layer 1 queue
    queue_file = "data/layer1_output/llm_review_queue.json"
    if not Path(queue_file).exists():
        print(f"❌ {queue_file} not found")
        print("   Run Layer 1 first to generate the queue")
        return
    
    with open(queue_file, 'r') as f:
        queue = json.load(f)
    
    logger.info(f"Loaded {len(queue)} conversations from Layer 1 queue\n")
    
    # Load conversations
    with open("data/conversations.json", 'r') as f:
        all_conversations = {c['conversation_id']: c for c in json.load(f)}
    
    # Initialize validator
    validator = DualLLMValidator(config, prompt_template, rules_data)
    
    # Process queue
    print("="*70)
    print("VALIDATING CONVERSATIONS")
    print("="*70 + "\n")
    
    results = []
    for i, layer1_item in enumerate(queue, 1):
        conv_id = layer1_item['conversation_id']
        print(f"\n[{i}/{len(queue)}] Processing {conv_id}...")
        
        conversation = all_conversations[conv_id]
        result = validator.validate_conversation(conversation, layer1_item)
        results.append(result)
        
        # Track per-conversation cost
        validator.cost_tracker.add_conversation_cost(
            conv_id,
            result.judge1_tokens,
            result.judge2_tokens
        )
        
        print(f"  Final: {result.final_decision.upper()}")
        if not result.judges_agree:
            print(f"  ⚠️  Judges disagreed - needs human review")
        
        # Rate limiting
        time.sleep(1.0)
    
    # Generate outputs
    cost_summary = validator.cost_tracker.get_summary()
    output_gen = OutputGenerator()
    output_gen.save_all(results, cost_summary)
    
    # Print summary
    print("\n" + "="*70)
    print("LAYER 2 SUMMARY")
    print("="*70)
    
    total = len(results)
    validated = sum(1 for r in results if r.final_decision != "needs_human_review")
    human_review = sum(1 for r in results if r.final_decision == "needs_human_review")
    
    # Count compliant vs non-compliant in validated
    validated_compliant = sum(1 for r in results if r.final_decision == "compliant")
    validated_non_compliant = sum(1 for r in results if r.final_decision == "non_compliant")
    
    print(f"\n📊 High-Level Summary:")
    print(f"  Total conversations processed: {total}")
    print(f"  ✅ Compliant (validated): {validated_compliant}")
    print(f"  ❌ Non-Compliant (validated): {validated_non_compliant}")
    print(f"  ⚠️  Needs Human Review: {human_review}")
    
    print(f"\n🤝 Judge Agreement:")
    agreed = sum(1 for r in results if r.judges_agree)
    print(f"  Agreed: {agreed} ({agreed/total*100:.1f}%)")
    print(f"  Disagreed: {total-agreed} ({(total-agreed)/total*100:.1f}%)")
    
    print(f"\n💰 Cost Summary:")
    print(f"  Judge 1 ({cost_summary['judge1']['model']}): ${cost_summary['judge1']['cost_usd']:.4f}")
    print(f"  Judge 2 ({cost_summary['judge2']['model']}): ${cost_summary['judge2']['cost_usd']:.4f}")
    print(f"  Total: ${cost_summary['total']['cost_usd']:.4f}")
    print(f"  Per conversation: ${cost_summary['total']['cost_usd']/total:.4f}")
    
    print(f"\n📁 Outputs:")
    print(f"  data/layer2_output/")
    print(f"    ├─ validated_decisions.json ({validated} conversations) → JSON")
    if human_review > 0:
        print(f"    ├─ human_review_needed.xlsx ({human_review} conversations) → Excel ⭐")
    print(f"    ├─ layer2_all_results.xlsx (all {total} conversations) → Excel ⭐")
    print(f"    └─ cost_report.json (detailed costs) → JSON")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()