"""
Agent-based question quality evaluator for the file benchmark.

This script runs an AI agent that evaluates generated questions based on
a quality rubric, accepting or rejecting them with detailed reasoning.
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import yaml

from anthropic import Anthropic
from jinja2 import Template
from leaderboard.letta_file_bench.tools.evaluation_decision_tool import EvaluationDecisionTool


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


class QuestionEvaluatorAgent:
    """Agent that evaluates question quality using Claude."""
    
    def __init__(
        self,
        output_dir: Path,
        model: str = None,
        config: Dict[str, Any] = None
    ):
        self.output_dir = output_dir
        
        # Load config if not provided
        if config is None:
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                config = full_config.get('agent_question_evaluator', {})
        
        self.config = config
        self.model = model or config.get('default_model', 'claude-sonnet-4-20250514')
        
        # Initialize evaluation tool
        self.evaluation_tool = EvaluationDecisionTool(output_dir)
        
        # Initialize Claude client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
        
        # Load system prompt
        prompt_path = Path(__file__).parent / "prompts" / "evaluator_system_prompt.j2"
        with open(prompt_path, 'r') as f:
            prompt_template = Template(f.read())
        
        # Load quality rubric
        rubric_path = Path(__file__).parent / "prompts" / "question_quality_rubric.txt"
        with open(rubric_path, 'r') as f:
            rubric_content = f.read()
        
        self.system_prompt = prompt_template.render(rubric=rubric_content)
        
        # Track token usage
        self.total_tokens = {"input": 0, "output": 0}
    
    def _print_separator(self, char: str = "─", length: int = 80):
        """Print a separator line."""
        print(f"{Colors.DIM}{char * length}{Colors.ENDC}")
    
    def _print_question_info(self, question_num: int, question_data: Dict[str, Any]):
        """Print question information."""
        question = question_data.get('question', '')
        print(f"\n{Colors.BLUE}Evaluating Question {question_num}{Colors.ENDC}")
        print(f"{Colors.DIM}Question: {question[:150]}{'...' if len(question) > 150 else ''}{Colors.ENDC}")
    
    def _print_evaluation_result(self, result: Dict[str, Any], reasoning: str):
        """Print evaluation result."""
        if result.get('decision') == 'accept':
            print(f"{Colors.GREEN}✓ ACCEPTED{Colors.ENDC}")
        else:
            print(f"{Colors.RED}✗ REJECTED{Colors.ENDC}")
        
        # Print reasoning summary (first line)
        reasoning_lines = reasoning.strip().split('\n')
        if reasoning_lines:
            print(f"{Colors.DIM}Reasoning: {reasoning_lines[0]}{Colors.ENDC}")
    
    def _print_token_usage(self, usage):
        """Print token usage from response."""
        if usage:
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            
            self.total_tokens["input"] += input_tokens
            self.total_tokens["output"] += output_tokens
            
            print(f"{Colors.DIM}Tokens: {input_tokens} in, {output_tokens} out{Colors.ENDC}")
    
    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question using the agent."""
        # Build the evaluation prompt
        messages = [
            {
                "role": "user",
                "content": f"""Please evaluate the following question:

QUESTION: {question_data.get('question', '')}

ANSWER: {question_data.get('answer', '')}

SQL QUERIES USED:
{json.dumps(question_data.get('sql_queries', []), indent=2)}

Evaluate this question against the quality rubric and use the record_decision tool to record your verdict."""
            }
        ]
        
        # Get agent response
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                system=self.system_prompt,
                max_tokens=2048,
                tool_choice={"type": "any"},
                tools=[
                    {
                        "name": "record_decision",
                        "description": "Record the evaluation decision for a question",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "decision": {
                                    "type": "string",
                                    "enum": ["accept", "reject"],
                                    "description": "Accept or reject the question"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Detailed explanation of your decision"
                                },
                                "quality_aspects": {
                                    "type": "object",
                                    "description": "Breakdown of quality assessment",
                                    "properties": {
                                        "complexity_level": {
                                            "type": "string",
                                            "enum": ["too_simple", "balanced", "too_complex"]
                                        },
                                        "coherence": {
                                            "type": "string",
                                            "enum": ["poor", "fair", "good"]
                                        },
                                        "has_obscure_patterns": {
                                            "type": "string",
                                            "enum": ["yes", "no"]
                                        },
                                        "rejection_category": {
                                            "type": "string",
                                            "description": "If rejected, the main reason"
                                        }
                                    }
                                }
                            },
                            "required": ["decision", "reasoning"]
                        }
                    }
                ]
            )
            
            # Print token usage
            self._print_token_usage(response.usage)
            
            # Process tool calls
            evaluation_result = None
            for content_block in response.content:
                if content_block.type == "tool_use" and content_block.name == "record_decision":
                    tool_input = content_block.input
                    
                    # Record the decision
                    result = self.evaluation_tool.record_decision(
                        question_data=question_data,
                        decision=tool_input["decision"],
                        reasoning=tool_input["reasoning"],
                        quality_aspects=tool_input.get("quality_aspects", {})
                    )
                    
                    evaluation_result = {
                        "success": result["success"],
                        "decision": tool_input["decision"],
                        "reasoning": tool_input["reasoning"],
                        "quality_aspects": tool_input.get("quality_aspects", {})
                    }
                    
                    # Print result
                    self._print_evaluation_result(evaluation_result, tool_input["reasoning"])
            
            if not evaluation_result:
                print(f"{Colors.YELLOW}Warning: Agent did not call evaluation tool{Colors.ENDC}")
                evaluation_result = {"success": False, "error": "No evaluation decision made"}
            
            return evaluation_result
            
        except Exception as e:
            print(f"{Colors.RED}Error during evaluation: {e}{Colors.ENDC}")
            return {"success": False, "error": str(e)}
    
    def evaluate_questions_file(self, input_path: Path):
        """Evaluate all questions in a file."""
        print(f"\n{Colors.HEADER}Question Quality Evaluation with {self.model}{Colors.ENDC}")
        print(f"Input: {input_path}")
        print(f"Output: {self.output_dir}")
        self._print_separator("=")
        
        # Load questions
        questions = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(questions)} questions to evaluate")
        self._print_separator()
        
        # Evaluate each question
        for i, question_data in enumerate(questions, 1):
            self._print_question_info(i, question_data)
            self.evaluate_single_question(question_data)
            self._print_separator()
        
        # Print summary
        summary = self.evaluation_tool.get_summary()
        
        print(f"\n{Colors.HEADER}Evaluation Complete!{Colors.ENDC}")
        self._print_separator("=")
        print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
        print(f"   Total evaluated: {summary['total_evaluated']}")
        print(f"   {Colors.GREEN}Accepted: {summary['accepted']} ({summary['acceptance_rate']}){Colors.ENDC}")
        print(f"   {Colors.RED}Rejected: {summary['rejected']}{Colors.ENDC}")
        
        if summary['rejection_categories']:
            print(f"\n{Colors.YELLOW}Rejection Categories:{Colors.ENDC}")
            for category, count in summary['rejection_categories'].items():
                print(f"   - {category}: {count}")
        
        print(f"\n{Colors.BOLD}Output Files:{Colors.ENDC}")
        for file_type, path in summary['output_files'].items():
            print(f"   - {file_type}: {path}")
        
        print(f"\n{Colors.DIM}Total tokens used: {self.total_tokens['input']:,} input, {self.total_tokens['output']:,} output{Colors.ENDC}")
        print(f"{Colors.DIM}Estimated cost: ${(self.total_tokens['input'] * 0.003 + self.total_tokens['output'] * 0.015) / 1000:.2f}{Colors.ENDC}")
        self._print_separator("=")
    
    def save_summary_report(self):
        """Save a summary report of the evaluation."""
        report_path = self.output_dir / "evaluation_report.json"
        summary = self.evaluation_tool.get_summary()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "statistics": summary,
            "token_usage": self.total_tokens,
            "estimated_cost": (self.total_tokens['input'] * 0.003 + self.total_tokens['output'] * 0.015) / 1000
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nSaved evaluation report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate question quality using AI agent")
    parser.add_argument(
        "--input",
        type=Path,
        help="Input file with questions to evaluate. If not specified, evaluates the latest generated questions."
    )
    parser.add_argument(
        "--input-run",
        type=str,
        help="Specific run directory to evaluate (e.g., 'run_20250729_143025'). Used if --input is not specified."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for evaluation results. If not specified, saves in the same directory as input."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use for evaluation"
    )
    
    args = parser.parse_args()
    
    # Determine input file
    if args.input:
        input_path = args.input
    else:
        # Look for generated questions
        generated_dir = Path(__file__).parent / "data" / "generated_questions"
        
        if args.input_run:
            # Use specific run
            run_dir = generated_dir / args.input_run
            input_path = run_dir / "agent_generated_questions.jsonl"
        else:
            # Find latest run
            if generated_dir.exists():
                run_dirs = [d for d in generated_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
                if run_dirs:
                    latest_run = sorted(run_dirs)[-1]  # Sort by name (timestamp)
                    input_path = latest_run / "agent_generated_questions.jsonl"
                    print(f"Using latest run: {latest_run.name}")
                else:
                    print(f"Error: No runs found in {generated_dir}")
                    return
            else:
                print(f"Error: Generated questions directory not found: {generated_dir}")
                return
    
    # Verify input file exists
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return
    
    # Determine output directory
    if args.output_dir:
        # Use specified output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir / f"eval_{timestamp}"
    else:
        # Use same directory as input file
        output_dir = input_path.parent
    
    # Create evaluator and run evaluation
    try:
        evaluator = QuestionEvaluatorAgent(
            output_dir=output_dir,
            model=args.model
        )
        
        evaluator.evaluate_questions_file(input_path)
        evaluator.save_summary_report()
        
    except Exception as e:
        print(f"{Colors.RED}Fatal error: {e}{Colors.ENDC}")
        return


if __name__ == "__main__":
    main()