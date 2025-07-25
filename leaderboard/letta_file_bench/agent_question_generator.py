"""
Agent-based question generator for the file benchmark.

This script runs an AI agent that generates difficult questions by:
1. Exploring the SQLite database
2. Finding unique identifiers and relationships
3. Creating questions that require multiple file lookups
4. Verifying answers through SQL execution
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import asyncio

from anthropic import Anthropic
from leaderboard.letta_file_bench.tools.sql_execute_tool import SQLExecuteTool
from leaderboard.letta_file_bench.tools.register_question_tool import RegisterQuestionTool


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


class QuestionGeneratorAgent:
    """Agent that generates questions using Claude Sonnet 4."""
    
    def __init__(
        self, 
        db_path: Path, 
        output_path: Path,
        model: str = "claude-sonnet-4-20250514"
    ):
        self.db_path = db_path
        self.output_path = output_path
        self.model = model
        
        # Initialize tools
        self.sql_tool = SQLExecuteTool(db_path)
        self.register_tool = RegisterQuestionTool(output_path, db_path)
        
        # Initialize Claude client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
        
        # Load system prompt
        prompt_path = Path(__file__).parent / "prompts" / "agent_system_prompt.txt"
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read()
        
        # Track token usage
        self.total_tokens = {"input": 0, "output": 0}
    
    def _print_separator(self, char: str = "─", length: int = 80):
        """Print a separator line."""
        print(f"{Colors.DIM}{char * length}{Colors.ENDC}")
    
    def _print_tool_call(self, tool_name: str, tool_input: Dict[str, Any]):
        """Print tool call information."""
        print(f"\n{Colors.CYAN}Tool: {tool_name}{Colors.ENDC}")
        if tool_name == "execute_sql":
            # Format SQL query nicely
            query = tool_input["query"]
            # Simple formatting - indent and clean up
            formatted_query = "\n   ".join(line.strip() for line in query.split("\n") if line.strip())
            print(f"   {Colors.BOLD}Query:{Colors.ENDC}\n   {Colors.DIM}{formatted_query}{Colors.ENDC}")
        elif tool_name == "register_question":
            print(f"   {Colors.BOLD}Question:{Colors.ENDC} {tool_input['question']}")
            print(f"   {Colors.BOLD}SQL:{Colors.ENDC} {tool_input['sql_query'][:100]}..." if len(tool_input['sql_query']) > 100 else f"   {Colors.BOLD}SQL:{Colors.ENDC} {tool_input['sql_query']}")
    
    def _print_tool_result(self, tool_name: str, result: Dict[str, Any]):
        """Print tool result information."""
        if tool_name == "execute_sql":
            if result["success"]:
                # Format result based on type
                res = result['result']
                if isinstance(res, list) and len(res) > 5:
                    # Truncate long lists
                    print(f"   {Colors.GREEN}Result: {res[:3]} ... (showing 3 of {len(res)} rows){Colors.ENDC}")
                elif isinstance(res, str) and len(res) > 200:
                    # Truncate long strings
                    print(f"   {Colors.GREEN}Result: {res[:200]}...{Colors.ENDC}")
                else:
                    print(f"   {Colors.GREEN}Result: {res}{Colors.ENDC}")
                print(f"   {Colors.DIM}Rows: {result['row_count']} | Time: {result['execution_time_ms']:.1f}ms{Colors.ENDC}")
            else:
                print(f"   {Colors.RED}Error: {result['error']}{Colors.ENDC}")
        elif tool_name == "register_question":
            if result["success"]:
                print(f"   {Colors.GREEN}{result['message']}{Colors.ENDC}")
                print(f"   {Colors.BOLD}Answer: {result['answer']}{Colors.ENDC}")
            else:
                print(f"   {Colors.RED}{result['error']}{Colors.ENDC}")
    
    def _print_progress(self, question_num: int, total: int, iteration: int, max_iterations: int):
        """Print progress information."""
        print(f"\n{Colors.BLUE}Question {question_num}/{total} | Iteration {iteration}/{max_iterations}{Colors.ENDC}")
    
    def _print_token_usage(self, usage, session_tokens=None):
        """Print token usage from response."""
        if usage:
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            
            # Update both session and total tokens
            if session_tokens:
                session_tokens["input"] += input_tokens
                session_tokens["output"] += output_tokens
                print(f"   Tokens: {input_tokens} in, {output_tokens} out (Session: {session_tokens['input']:,} in, {session_tokens['output']:,} out)")
            
            self.total_tokens["input"] += input_tokens
            self.total_tokens["output"] += output_tokens
    
    def _format_tool_response(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format tool response for the agent."""
        if tool_name == "execute_sql":
            if result["success"]:
                return f"SQL executed successfully.\nResult: {result['result']}\nRows: {result['row_count']}"
            else:
                return f"SQL error: {result['error']}"
        elif tool_name == "register_question":
            if result["success"]:
                return f"Question registered! {result['message']}"
            else:
                return f"Registration failed: {result['error']}"
        return str(result)
    
    def get_existing_questions(self) -> List[Dict[str, str]]:
        """Get list of already generated questions from output file."""
        if not self.output_path.exists():
            return []
        
        questions = []
        with open(self.output_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return questions
    
    def _format_existing_questions(self, questions: List[Dict[str, str]]) -> str:
        """Format ALL existing questions for agent context."""
        if not questions:
            return "No questions generated yet. You have complete freedom to be creative!"
        
        summary = [f"Here are ALL {len(questions)} existing questions (you must generate something different):"]
        summary.append("=" * 80)
        
        # Include ALL questions
        for i, q in enumerate(questions, 1):
            summary.append(f"{i}. {q['question']}")
            # Also show the answer to help avoid similar patterns
            if 'answer' in q:
                summary.append(f"   Answer: {q['answer']}")
        
        summary.append("=" * 80)
        summary.append("Generate a completely different and creative question!")
        return "\n".join(summary)
    
    def _save_full_trace(self, session_id: str, all_conversations: List[Dict[str, Any]]):
        """Save the complete conversation trace."""
        logs_dir = self.output_path.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        trace_path = logs_dir / f"agent_trace_{session_id}.json"
        
        # Count successful questions
        successful_questions = sum(1 for conv in all_conversations if conv["success"])
        
        with open(trace_path, 'w') as f:
            json.dump({
                "session_id": session_id,
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "total_questions_target": len(all_conversations),
                "total_questions_generated": successful_questions,
                "conversations": all_conversations
            }, f, indent=2, default=str)
        print(f"Saved conversation trace to {trace_path}")
    
    def generate_single_question(
        self, 
        question_number: int,
        total: int,
        existing_questions: List[Dict[str, str]],
        max_iterations: int = 20
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Generate a single question with fresh conversation."""
        
        # Track tokens for this session only
        session_tokens = {"input": 0, "output": 0}
        
        # Build user message with context about existing questions
        existing_summary = self._format_existing_questions(existing_questions)
        
        messages = [
            {
                "role": "user",
                "content": f"Generate question #{question_number}.\n\n"
                          f"{existing_summary}\n\n"
                          f"Your task: Generate ONE new, unique, difficult question that is completely different from the above.\n"
                          f"Remember to:\n"
                          f"1. Explore the data to find unique patterns\n"
                          f"2. Verify your question has exactly ONE answer\n"
                          f"3. Be creative - don't follow the same patterns as existing questions\n\n"
                          f"Use execute_sql to explore thoroughly. When you have found a great question and verified it has a unique answer, "
                          f"call register_question. Note: register_question will END this session immediately, so only call it when completely ready!"
            }
        ]
        
        conversation_trace = messages.copy()
        question_registered = False
        
        for iteration in range(1, max_iterations + 1):
            self._print_progress(question_number, total, iteration, max_iterations)
            # Get agent response
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                system=self.system_prompt,
                max_tokens=4096,
                tool_choice={"type": "any"},
                tools=[
                    {
                        "name": "execute_sql",
                        "description": "Execute a SQL query against the database",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The SQL query to execute"
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "register_question",
                        "description": "Register a question with its SQL query after verifying it has a unique answer",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The human-readable question"
                                },
                                "sql_query": {
                                    "type": "string",
                                    "description": "The SQL query that produces the answer"
                                }
                            },
                            "required": ["question", "sql_query"]
                        }
                    }
                ]
            )
            
            # Print token usage
            self._print_token_usage(response.usage, session_tokens)
            
            # Add assistant message to conversation
            assistant_message = {"role": "assistant", "content": response.content}
            messages.append(assistant_message)
            conversation_trace.append(assistant_message)
            
            # Process tool calls
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_name = content_block.name
                    tool_input = content_block.input
                    
                    # Print tool call
                    self._print_tool_call(tool_name, tool_input)
                    
                    # Record tool call
                    tool_call_record = {
                        "role": "tool_call",
                        "tool": tool_name,
                        "input": tool_input
                    }
                    conversation_trace.append(tool_call_record)
                    
                    # Execute tool
                    if tool_name == "execute_sql":
                        result = self.sql_tool.execute(tool_input["query"])
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": self._format_tool_response(tool_name, result)
                        })
                    elif tool_name == "register_question":
                        result = self.register_tool.register(
                            tool_input["question"],
                            tool_input["sql_query"]
                        )
                        if result["success"]:
                            question_registered = True
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": self._format_tool_response(tool_name, result)
                        })
                    
                    # Print tool result
                    self._print_tool_result(tool_name, result)
                    
                    # Record tool result
                    tool_result_record = {
                        "role": "tool_result",
                        "tool": tool_name,
                        "result": result
                    }
                    conversation_trace.append(tool_result_record)
            
            # Add tool results to messages if any
            if tool_results:
                tool_message = {
                    "role": "user",
                    "content": tool_results
                }
                messages.append(tool_message)
                # Note: tool results are already in conversation_trace
            
            # Check if we successfully registered a question
            if question_registered:
                print(f"\n{Colors.GREEN}Session ended - Question successfully registered!{Colors.ENDC}")
                break
        
        # If we hit max iterations without registering, force registration
        if not question_registered and iteration == max_iterations:
            print(f"\n{Colors.YELLOW}Hit max iterations without registering - forcing registration...{Colors.ENDC}")
            
            # Add a final user message
            force_message = {
                "role": "user",
                "content": "You've explored enough! You must now register a question immediately. "
                          "Pick your best question idea and register it NOW using register_question."
            }
            messages.append(force_message)
            conversation_trace.append(force_message)
            
            # Force tool use with register_question
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    system=self.system_prompt,
                    max_tokens=4096,
                    tool_choice={"type": "tool", "name": "register_question"},
                    tools=[
                        {
                            "name": "register_question",
                            "description": "Register a question with its SQL query after verifying it has a unique answer",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The human-readable question"
                                    },
                                    "sql_query": {
                                        "type": "string",
                                        "description": "The SQL query that produces the answer"
                                    }
                                },
                                "required": ["question", "sql_query"]
                            }
                        }
                    ]
                )
                
                # Print token usage
                self._print_token_usage(response.usage, session_tokens)
                
                # Process the forced registration
                assistant_message = {"role": "assistant", "content": response.content}
                messages.append(assistant_message)
                conversation_trace.append(assistant_message)
                
                for content_block in response.content:
                    if content_block.type == "tool_use" and content_block.name == "register_question":
                        tool_input = content_block.input
                        
                        # Print tool call
                        self._print_tool_call("register_question", tool_input)
                        
                        # Record tool call
                        tool_call_record = {
                            "role": "tool_call",
                            "tool": "register_question",
                            "input": tool_input
                        }
                        conversation_trace.append(tool_call_record)
                        
                        # Execute registration
                        result = self.register_tool.register(
                            tool_input["question"],
                            tool_input["sql_query"]
                        )
                        
                        if result["success"]:
                            question_registered = True
                        
                        # Print tool result
                        self._print_tool_result("register_question", result)
                        
                        # Record tool result
                        tool_result_record = {
                            "role": "tool_result",
                            "tool": "register_question",
                            "result": result
                        }
                        conversation_trace.append(tool_result_record)
                        
                        print(f"\n{Colors.GREEN}Forced registration completed!{Colors.ENDC}")
                        
            except Exception as e:
                print(f"\n{Colors.RED}Error during forced registration: {e}{Colors.ENDC}")
                error_record = {
                    "role": "error",
                    "message": f"Forced registration failed: {str(e)}"
                }
                conversation_trace.append(error_record)
        
        return question_registered, conversation_trace
    
    def generate_questions(self, num_questions: int = 10):
        """Generate questions using the agent - one at a time with fresh context."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_conversations = []
        
        print(f"\n{Colors.HEADER}Starting question generation with {self.model}{Colors.ENDC}")
        print(f"Target: {num_questions} questions")
        print(f"Output: {self.output_path}")
        self._print_separator("=")
        
        for question_num in range(num_questions):
            # Get all existing questions for context
            existing_questions = self.get_existing_questions()
            
            print(f"\n{Colors.BOLD}Generating question {question_num + 1}/{num_questions}{Colors.ENDC}")
            print(f"   {Colors.DIM}Existing questions in corpus: {len(existing_questions)}{Colors.ENDC}")
            self._print_separator()
            
            # Generate one question with fresh context
            success, conversation = self.generate_single_question(
                question_num + 1,
                num_questions,
                existing_questions,
                max_iterations=20
            )
            
            # Store conversation
            all_conversations.append({
                "question_number": question_num + 1,
                "success": success,
                "conversation": conversation
            })
            
            self._print_separator()
            if success:
                print(f"\n{Colors.GREEN}Successfully generated question {question_num + 1}{Colors.ENDC}")
                # Get the newly registered question to show it
                new_questions = self.get_existing_questions()
                if new_questions and len(new_questions) > len(existing_questions):
                    latest = new_questions[-1]
                    print(f"   {Colors.BOLD}Question:{Colors.ENDC} {latest['question']}")
                    print(f"   {Colors.BOLD}Answer:{Colors.ENDC} {latest['answer']}")
            else:
                print(f"\n{Colors.RED}Failed to generate question {question_num + 1} within iteration limit{Colors.ENDC}")
            
            self._print_separator("=")
        
        # Save full session trace
        self._save_full_trace(session_id, all_conversations)
        
        # Final summary
        successful_count = sum(1 for conv in all_conversations if conv["success"])
        
        print(f"\n{Colors.HEADER}Generation complete!{Colors.ENDC}")
        self._print_separator("=")
        print(f"{Colors.BOLD}Results:{Colors.ENDC}")
        print(f"   {Colors.GREEN if successful_count == num_questions else Colors.YELLOW}Successfully generated: {successful_count}/{num_questions} questions{Colors.ENDC}")
        print(f"   Questions saved to: {self.output_path}")
        print(f"   Full trace saved to: {self.output_path.parent / 'logs' / f'agent_trace_{session_id}.json'}")
        print(f"   {Colors.DIM}Total tokens used: {self.total_tokens['input']:,} input, {self.total_tokens['output']:,} output{Colors.ENDC}")
        print(f"   {Colors.DIM}Estimated cost: ${(self.total_tokens['input'] * 0.003 + self.total_tokens['output'] * 0.015) / 1000:.2f}{Colors.ENDC}")
        self._print_separator("=")


def main():
    parser = argparse.ArgumentParser(description="Generate questions using AI agent")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="Number of questions to generate"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(__file__).parent / "data" / "letta_file_bench.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).parent / "data" / "agent_generated_questions.jsonl",
        help="Output path for generated questions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use"
    )
    
    args = parser.parse_args()
    
    # Verify database exists
    if not args.db_path.exists():
        print(f"Error: Database not found at {args.db_path}")
        print("Please run the JSONL to SQLite conversion first.")
        return
    
    # Create agent and generate questions
    agent = QuestionGeneratorAgent(
        db_path=args.db_path,
        output_path=args.output_path,
        model=args.model
    )
    
    print(f"Starting question generation with {args.model}...")
    print(f"Database: {args.db_path}")
    print(f"Output: {args.output_path}")
    print(f"Target: {args.num_questions} questions\n")
    
    agent.generate_questions(args.num_questions)


if __name__ == "__main__":
    main()