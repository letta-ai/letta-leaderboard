"""
Utilities for selecting random people from the golden answers dataset.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def select_random_people(golden_answers_path: Path, n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Randomly select N people from the golden answers file."""
    random.seed(seed)
    
    with open(golden_answers_path, 'r') as f:
        golden_data = json.load(f)
    
    all_people = list(golden_data.values())
    selected_people = random.sample(all_people, min(n, len(all_people)))
    
    return selected_people