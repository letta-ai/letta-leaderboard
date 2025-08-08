"""
Evaluation decision tool for recording question quality assessments.
"""
import json
from pathlib import Path
from typing import Dict, Any, Literal, Optional
from datetime import datetime


class EvaluationDecisionTool:
    """Tool for recording question evaluation decisions."""
    
    def __init__(self, output_dir: Path):
        """Initialize with output directory for evaluation results."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize output files with unique names to avoid overwriting
        self.accepted_path = self.output_dir / "evaluated_accepted_questions.jsonl"
        self.rejected_path = self.output_dir / "evaluated_rejected_questions.jsonl"
        self.evaluation_log_path = self.output_dir / "evaluation_log.jsonl"
        
        # Statistics
        self.stats = {
            "total_evaluated": 0,
            "accepted": 0,
            "rejected": 0,
            "rejection_categories": {}
        }
    
    def record_decision(
        self,
        question_data: Dict[str, Any],
        decision: Literal["accept", "reject"],
        reasoning: str,
        quality_aspects: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Record an evaluation decision for a question.
        
        Args:
            question_data: The full question data including question, answer, sql_queries
            decision: "accept" or "reject"
            reasoning: Detailed explanation of the decision
            quality_aspects: Optional breakdown of quality aspects evaluated
            
        Returns:
            Result dictionary with status and message
        """
        try:
            self.stats["total_evaluated"] += 1
            
            # Add evaluation metadata
            evaluation_metadata = {
                "_evaluation": {
                    "decision": decision,
                    "reasoning": reasoning,
                    "quality_aspects": quality_aspects or {},
                    "evaluated_at": datetime.now().isoformat()
                }
            }
            
            # Merge metadata with original question data
            evaluated_question = {**question_data, **evaluation_metadata}
            
            # Write to appropriate file based on decision
            if decision == "accept":
                self.stats["accepted"] += 1
                with open(self.accepted_path, 'a') as f:
                    f.write(json.dumps(evaluated_question) + '\n')
                result_msg = f"Question ACCEPTED and saved to evaluated_accepted_questions.jsonl"
            else:
                self.stats["rejected"] += 1
                with open(self.rejected_path, 'a') as f:
                    f.write(json.dumps(evaluated_question) + '\n')
                result_msg = f"Question REJECTED and saved to evaluated_rejected_questions.jsonl"
                
                # Track rejection categories if provided
                if quality_aspects and "rejection_category" in quality_aspects:
                    category = quality_aspects["rejection_category"]
                    self.stats["rejection_categories"][category] = \
                        self.stats["rejection_categories"].get(category, 0) + 1
            
            # Log the evaluation decision
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question_data.get("question", ""),
                "decision": decision,
                "reasoning": reasoning,
                "quality_aspects": quality_aspects
            }
            with open(self.evaluation_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            return {
                "success": True,
                "message": result_msg,
                "decision": decision,
                "stats": self.stats
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary statistics."""
        total = self.stats["total_evaluated"]
        if total == 0:
            acceptance_rate = 0
        else:
            acceptance_rate = (self.stats["accepted"] / total) * 100
        
        return {
            "total_evaluated": total,
            "accepted": self.stats["accepted"],
            "rejected": self.stats["rejected"],
            "acceptance_rate": f"{acceptance_rate:.1f}%",
            "rejection_categories": self.stats["rejection_categories"],
            "output_files": {
                "accepted": str(self.accepted_path),
                "rejected": str(self.rejected_path),
                "log": str(self.evaluation_log_path)
            }
        }