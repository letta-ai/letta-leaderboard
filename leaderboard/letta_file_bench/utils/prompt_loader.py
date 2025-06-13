"""
Utilities for loading and formatting prompt templates.
"""
from pathlib import Path
from typing import Dict, Any, List


def load_prompt_template(prompt_file: Path) -> str:
    """Load the prompt template from file."""
    return prompt_file.read_text(encoding="utf-8")


def create_people_context(selected_people: List[Dict[str, Any]]) -> tuple[str, str]:
    """Create context strings about the selected people."""
    
    # Create summary info
    people_info = ""
    for i, person in enumerate(selected_people, 1):
        people_info += f"Person {i}: {person['full_name']} (ID: {person['person_id']})\n"
        people_info += f"  Basic Info: DOB {person['dob']}, Email {person['email']}, Phone {person['phone']}\n"
        
        # Count records in each category
        for category, display_name in [
            ('addresses', 'Addresses'), ('bank_accounts', 'Bank Accounts'), 
            ('employments', 'Employments'), ('credit_cards', 'Credit Cards'),
            ('vehicles', 'Vehicles'), ('pets', 'Pets'), 
            ('internet_accounts', 'Internet Accounts'), ('insurances', 'Insurance'),
            ('medical_records', 'Medical Records')
        ]:
            if person.get(category):
                people_info += f"  {display_name}: {len(person[category])} records\n"
        people_info += "\n"
    
    # Complete data as JSON
    import json
    complete_data = json.dumps(selected_people, indent=2)
    
    return people_info, complete_data