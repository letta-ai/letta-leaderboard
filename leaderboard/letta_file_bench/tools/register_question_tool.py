"""
Question registration tool for agent question generation.

This tool allows the agent to register validated questions with answers.
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class RegisterQuestionTool:
    """Tool for registering validated questions."""
    
    def __init__(self, output_path: Path, db_path: Path):
        """Initialize with output path for questions and database path."""
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.registered_count = 0
    
    def register(
        self,
        question: str,
        sql_query: str
    ) -> Dict[str, Any]:
        """
        Register a new question by executing its SQL query to get the answer.
        
        Args:
            question: The human-readable question
            sql_query: The SQL query that produces the answer
            
        Returns:
            Dictionary with registration status and the answer
        """
        try:
            # Execute the SQL query to get the answer
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            
            # Convert result to appropriate format
            if len(rows) == 0:
                answer = "No results"
            elif len(rows) == 1 and len(rows[0]) == 1:
                # Single value result
                answer = str(list(dict(rows[0]).values())[0])
            else:
                # Multiple rows or columns
                answer = str([dict(row) for row in rows])
            
            conn.close()
            
            self.registered_count += 1
            
            # Store question with answer
            question_data = {
                "question": question,
                "answer": answer,
                "sql_query": sql_query,
                "timestamp": datetime.now().isoformat()
            }
            
            # Append to output file
            with open(self.output_path, 'a') as f:
                f.write(json.dumps(question_data) + '\n')
            
            return {
                "success": True,
                "message": f"Question registered successfully. Answer: {answer}",
                "answer": answer,
                "total_questions": self.registered_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }