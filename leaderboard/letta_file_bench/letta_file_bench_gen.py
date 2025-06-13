"""
File Ledger Benchmark Data Generator

Generates synthetic people and their associated records across multiple files
to simulate a multi-hop file navigation scenario. This creates a realistic
file-based database where information is spread across multiple files and
requires cross-references to answer questions.

Generated file structure:
    dataset/
      ├── people.txt              # Person info (name, DOB, email, phone)
      ├── addresses.txt            # Home/work addresses
      ├── bank_accounts.txt        # Banking information (routing, account numbers)
      ├── employments.txt          # Job history and salary info
      ├── credit_cards.txt         # Credit card details
      ├── vehicles.txt             # Vehicle registrations
      ├── pets.txt                 # Pet ownership records
      ├── internet_accounts.txt    # Online account details
      ├── insurance_policies.txt   # Insurance policy information
      ├── medical_records.txt      # Medical history (anonymized)
      └── golden_answers.json      # Ground truth for evaluation

Each entity record includes the owner's person_id to enable multi-hop queries
like "What is John Smith's bank routing number?" which requires:
1. Finding John Smith's person_id in people.txt
2. Finding his bank account in bank_accounts.txt using that person_id
3. Extracting the routing number from the bank account record

The golden_answers.json file contains the flattened ground truth for
automated evaluation of agent responses.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

from faker import Faker
from openai import AsyncOpenAI
from tqdm import tqdm

# Import from refactored modules
from leaderboard.letta_file_bench.models.question_models import QuestionAnswer
from leaderboard.letta_file_bench.models.entities import (
    Person, Address, BankAccount, Employment, CreditCard, Vehicle, Pet,
    InternetAccount, InsurancePolicy, MedicalRecord, ENTITY_MAP
)
from leaderboard.letta_file_bench.utils.id_generator import reset_id_counters, generate_unique_id
from leaderboard.letta_file_bench.utils.uniqueness import reset_uniqueness_tracking, ensure_unique_value

# Backwards compatibility functions
def _uid(pref: str) -> str:
    return generate_unique_id(pref)

def _ensure_unique_name(name_generator, name_type: str, max_attempts: int = 100):
    return ensure_unique_value(name_generator, name_type, max_attempts)

def _reset_name_tracking():
    reset_uniqueness_tracking()
    reset_id_counters()




def gen_population(args):
    # Reset name tracking for fresh generation
    _reset_name_tracking()
    
    random.seed(args.seed)
    fk = Faker(args.locales)
    Faker.seed(args.seed)

    people: List[Person] = []
    for _ in tqdm(range(args.num_people), desc="Generating people", unit="person"):
        p = Person.fake(fk, random.choice(args.locales))
        p.addresses += [Address.fake(fk) for _ in range(random.randint(1, args.max_addresses))]
        p.accounts += [BankAccount.fake(fk) for _ in range(random.randint(1, args.max_accounts))]
        p.employments += [Employment.fake(fk) for _ in range(random.randint(0, args.max_employments))]
        p.credit_cards += [CreditCard.fake(fk) for _ in range(random.randint(0, args.max_credit_cards))]
        p.vehicles += [Vehicle.fake(fk) for _ in range(random.randint(0, args.max_vehicles))]
        p.pets += [Pet.fake(fk) for _ in range(random.randint(0, args.max_pets))]
        p.net_accounts += [InternetAccount.fake(fk) for _ in range(random.randint(0, args.max_net_accounts))]
        p.insurances += [InsurancePolicy.fake(fk) for _ in range(random.randint(0, args.max_insurances))]
        p.medical_records += [MedicalRecord.fake(fk) for _ in range(random.randint(0, args.max_medical_records))]
        people.append(p)
    return people



def write_corpus(people: List[Person], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"people": (out_dir / "people.txt").open("w", encoding="utf-8")}
    for key, (fname, _, _) in ENTITY_MAP.items():
        files[key] = (out_dir / fname).open("w", encoding="utf-8")

    evidence = {}

    for person in tqdm(sorted(people, key=lambda p: int(p.person_id.split('-')[1])), desc="Writing corpus files", unit="person"):
        pf = files["people"]
        start = pf.tell()
        pf.write(f"### {person.full_name} (ID: {person.person_id})\n")
        pf.write(f"DOB: {person.dob} | Email: {person.email} | Phone: {person.phone}\n\n")
        evidence[person.person_id] = {"file": str(Path(pf.name)), "offset": start, "bytes": pf.tell()-start}

        # Stream each linked entity to its own file
        for attr, (fname, _cls, id_field) in ENTITY_MAP.items():
            entities = sorted(getattr(person, attr), key=lambda e: int(getattr(e, id_field).split('-')[1]))
            for ent in entities:
                fh = files[attr]
                fh.write(f"### {getattr(ent, id_field)} (owner: {person.person_id})\n")
                for k, v in asdict(ent).items():
                    fh.write(f"{k}: {v}\n")
                fh.write("\n")

    for fh in files.values():
        fh.close()
    return evidence


def export_golden(people: List[Person], evidence: dict, out_dir: Path):
    golden = {p.person_id: p.flatten() | {"evidence": evidence[p.person_id]} for p in people}
    path = out_dir / "golden_answers.json"
    path.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    return path




def load_prompts(prompts_dir: Path) -> Dict[str, str]:
    """Load all three prompt templates."""
    return {
        "single_hop": (prompts_dir / "single_hop_prompt.txt").read_text(encoding="utf-8"),
        "multi_hop": (prompts_dir / "multi_hop_prompt.txt").read_text(encoding="utf-8"),
        "comparison": (prompts_dir / "comparison_prompt.txt").read_text(encoding="utf-8")
    }


async def generate_single_question(
    question_type: str,
    all_people: List[Dict[str, Any]],
    prompts: Dict[str, str],
    model: str,
    client: AsyncOpenAI,
    temperature: float = 0.8
) -> Dict[str, Any]:
    """Generate a single question by randomly sampling people from the entire dataset."""
    
    if question_type == "comparison":
        # Randomly sample 2 people for comparison
        person1, person2 = random.sample(all_people, 2)
        formatted_prompt = prompts["comparison"].format(
            person1_name=person1["full_name"],
            person2_name=person2["full_name"],
            person1_data=json.dumps(person1, indent=2),
            person2_data=json.dumps(person2, indent=2)
        )
    else:
        # Randomly sample 1 person for single_hop or multi_hop
        person = random.choice(all_people)
        formatted_prompt = prompts[question_type].format(
            person_name=person["full_name"],
            person_data=json.dumps(person, indent=2)
        )
    
    try:
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at creating challenging file-reading questions. Always provide questions with specific, verifiable answers."
                },
                {
                    "role": "user", 
                    "content": formatted_prompt
                }
            ],
            response_format=QuestionAnswer,
            temperature=temperature
        )
        
        qa = response.choices[0].message.parsed
        
        return {
            "question": qa.question,
            "answer": qa.answer,
            "difficulty": qa.difficulty,
            "question_type": qa.question_type,
            "required_files": qa.required_files,
            "reasoning_steps": qa.reasoning_steps
        }
        
    except Exception as e:
        print(f"Warning: Failed to generate {question_type} question: {e}")
        return None


def generate_questions_with_llm(
    all_people: List[Dict[str, Any]], 
    num_questions: int = 10,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    temperature: float = 0.8,
    max_concurrent: int = 5,
    single_hop_pct: float = 0.4,
    multi_hop_pct: float = 0.2,
    comparison_pct: float = 0.4,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Use an LLM with structured outputs to generate challenging questions by randomly sampling from all people."""
    
    if not api_key:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    # Validate percentages
    total_pct = single_hop_pct + multi_hop_pct + comparison_pct
    if abs(total_pct - 1.0) > 1e-6:
        raise ValueError(f"Question type percentages must sum to 1.0, got {total_pct}")
    
    client = AsyncOpenAI(api_key=api_key)
    prompts_dir = Path(__file__).parent / "prompts"
    prompts = load_prompts(prompts_dir)
    
    # Set random seed for reproducible sampling
    random.seed(seed)
    
    # Run async individual question generation
    all_questions = asyncio.run(generate_all_questions_async(
        all_people,
        prompts,
        num_questions,
        model,
        client,
        temperature,
        max_concurrent,
        single_hop_pct,
        multi_hop_pct,
        comparison_pct
    ))
    
    return all_questions


async def generate_all_questions_async(
    all_people: List[Dict[str, Any]], 
    prompts: Dict[str, str],
    num_questions: int,
    model: str,
    client: AsyncOpenAI,
    temperature: float,
    max_concurrent: int,
    single_hop_pct: float,
    multi_hop_pct: float,
    comparison_pct: float
) -> List[Dict[str, Any]]:
    """Generate individual questions concurrently with controlled concurrency."""
    
    # Calculate question type distribution
    single_hop_count = int(num_questions * single_hop_pct)
    multi_hop_count = int(num_questions * multi_hop_pct)
    comparison_count = num_questions - single_hop_count - multi_hop_count  # Remainder goes to comparison
    
    # Create question type schedule
    question_types = (
        ["single_hop"] * single_hop_count + 
        ["multi_hop"] * multi_hop_count + 
        ["comparison"] * comparison_count
    )
    
    # Shuffle the question types for variety
    random.shuffle(question_types)
    
    # Create all question generation tasks
    tasks = [
        generate_single_question(
            question_type, 
            all_people, 
            prompts, 
            model, 
            client, 
            temperature
        ) for question_type in question_types
    ]
    
    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(task):
        async with semaphore:
            return await task
    
    # Execute all questions with progress bar
    with tqdm(total=num_questions, desc=f"Generating questions with {model} (async)", unit="question") as pbar:
        results = []
        for coro in asyncio.as_completed([run_with_semaphore(task) for task in tasks]):
            result = await coro
            if result:  # Skip None results from failed generations
                results.append(result)
            pbar.update(1)
    
    return results




def export_llm_questions(questions: List[Dict[str, Any]], out_dir: Path, filename: str = "llm_generated_questions.jsonl"):
    """Export the LLM-generated questions to a JSONL file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    
    with open(path, 'w') as f:
        for question in questions:
            f.write(json.dumps(question) + '\n')
    
    return path


def parse_args():
    ap = argparse.ArgumentParser("Synthetic corpus generator")
    ap.add_argument("--num-people", type=int, default=100)
    ap.add_argument("--max-addresses", type=int, default=2)
    ap.add_argument("--max-accounts", type=int, default=3)
    ap.add_argument("--max-employments", type=int, default=1)
    ap.add_argument("--max-credit-cards", type=int, default=2)
    ap.add_argument("--max-vehicles", type=int, default=1)
    ap.add_argument("--max-pets", type=int, default=1)
    ap.add_argument("--max-net-accounts", type=int, default=2)
    ap.add_argument("--max-insurances", type=int, default=1)
    ap.add_argument("--max-medical-records", type=int, default=1)
    ap.add_argument("--locales", nargs="*", default=["en_US"], help="Faker locales")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    
    # LLM question generation options
    ap.add_argument("--generate-llm-questions", action="store_true", help="Generate questions using LLM from existing golden_answers.json")
    ap.add_argument("--llm-model", type=str, default="gpt-4o", help="LLM model to use for question generation")
    ap.add_argument("--num-questions", type=int, default=10, help="Number of questions to generate")
    ap.add_argument("--llm-seed", type=int, default=42, help="Seed for random selection")
    ap.add_argument("--temperature", type=float, default=0.8, help="LLM temperature for diversity vs accuracy balance")
    ap.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent requests for parallel generation")
    ap.add_argument("--single-hop-pct", type=float, default=0.4, help="Percentage of single-hop questions (0.0-1.0)")
    ap.add_argument("--multi-hop-pct", type=float, default=0.2, help="Percentage of multi-hop questions (0.0-1.0)")
    ap.add_argument("--comparison-pct", type=float, default=0.4, help="Percentage of comparison questions (0.0-1.0)")
    
    return ap.parse_args()


def main():
    args = parse_args()
    
    if args.generate_llm_questions:
        # LLM question generation mode - use existing golden_answers.json
        golden_path = args.output_dir / "golden_answers.json"
        
        if not golden_path.exists():
            print(f"❌ Error: {golden_path} does not exist. Generate data first without --generate-llm-questions flag.")
            return
        
        print(f"🔍 Generating LLM questions from existing data at {golden_path}")
        
        # Load all people from existing data
        with open(golden_path, 'r') as f:
            all_people_data = json.load(f)
        all_people = list(all_people_data.values())
        print(f"📝 Loaded {len(all_people)} people from golden_answers.json")
        
        # Generate questions using LLM
        try:
            questions = generate_questions_with_llm(
                all_people,
                args.num_questions,
                args.llm_model,
                temperature=args.temperature,
                max_concurrent=args.max_concurrent,
                single_hop_pct=args.single_hop_pct,
                multi_hop_pct=args.multi_hop_pct,
                comparison_pct=args.comparison_pct,
                seed=args.llm_seed
            )
            print(f"🤖 Generated {len(questions)} questions using {args.llm_model}")
            
            # Export questions
            questions_path = export_llm_questions(questions, args.output_dir)
            print(f"✅ LLM questions exported to {questions_path}")
            
            # Print some sample questions
            print("\n📋 Sample generated questions:")
            for i, q in enumerate(questions[:3], 1):
                print(f"{i}. {q['question']}")
                print(f"   Answer: {q['answer']}")
                print(f"   Difficulty: {q['difficulty']}")
                print()
                
        except Exception as e:
            print(f"❌ Error generating LLM questions: {e}")
            
    else:
        # Original data generation mode
        people = gen_population(args)
        evidence = write_corpus(people, args.output_dir)
        gpath = export_golden(people, evidence, args.output_dir)
        print(f"✅ Generated {len(people)} people; corpus @ {args.output_dir} | golden → {gpath}")

if __name__ == "__main__":
    main()

