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
from leaderboard.letta_file_bench.models.question_models import QuestionSet
from leaderboard.letta_file_bench.models.entities import (
    Person, Address, BankAccount, Employment, CreditCard, Vehicle, Pet,
    InternetAccount, InsurancePolicy, MedicalRecord, ENTITY_MAP
)
from leaderboard.letta_file_bench.utils.id_generator import reset_id_counters, generate_unique_id
from leaderboard.letta_file_bench.utils.uniqueness import reset_uniqueness_tracking, ensure_unique_value
from leaderboard.letta_file_bench.utils.prompt_loader import load_prompt_template, create_people_context
from leaderboard.letta_file_bench.utils.people_selector import select_random_people

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




async def generate_questions_batch(
    people_subset: List[Dict[str, Any]], 
    prompt_template: str,
    batch_size: int,
    model: str,
    client: AsyncOpenAI,
    temperature: float = 0.8,
    batch_id: int = 0
) -> List[Dict[str, Any]]:
    """Generate a single batch of questions for a subset of people asynchronously."""
    
    people_info, complete_data = create_people_context(people_subset)
    
    formatted_prompt = prompt_template.format(
        num_questions=batch_size,
        people_info=people_info,
        complete_data=complete_data
    )
    
    try:
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at creating challenging, multi-hop reasoning questions for file-reading benchmarks. You always provide questions with specific, verifiable answers."
                },
                {
                    "role": "user", 
                    "content": formatted_prompt
                }
            ],
            response_format=QuestionSet,
            temperature=temperature
        )
        
        question_set = response.choices[0].message.parsed
        
        # Convert to the format expected by the benchmark system
        questions = []
        for qa in question_set.questions:
            questions.append({
                "question": qa.question,
                "answer": qa.answer,
                "difficulty": qa.difficulty,
                "required_files": qa.required_files,
                "reasoning_steps": qa.reasoning_steps,
                "question_type": "multi_hop_llm_generated",
                "batch_id": batch_id  # For debugging/tracking
            })
        
        return questions
        
    except Exception as e:
        print(f"Warning: Batch {batch_id} failed: {e}")
        return []


def generate_questions_with_llm(
    selected_people: List[Dict[str, Any]], 
    prompt_file: Path,
    num_questions: int = 10,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_batch_size: int = 20,
    temperature: float = 0.8,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """Use an LLM with structured outputs to generate challenging questions in batches."""
    
    if not api_key:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = AsyncOpenAI(api_key=api_key)
    prompt_template = load_prompt_template(prompt_file)
    
    # Run async batch generation
    all_questions = asyncio.run(generate_all_batches_async(
        selected_people, 
        prompt_template,
        num_questions,
        model,
        client,
        max_batch_size,
        temperature,
        max_concurrent
    ))
    
    return all_questions


async def generate_all_batches_async(
    selected_people: List[Dict[str, Any]], 
    prompt_template: str,
    num_questions: int,
    model: str,
    client: AsyncOpenAI,
    max_batch_size: int,
    temperature: float,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """Generate all question batches concurrently with controlled concurrency."""
    
    # Calculate batching strategy
    num_batches = max(1, (num_questions + max_batch_size - 1) // max_batch_size)
    questions_per_batch = num_questions // num_batches
    remaining_questions = num_questions % num_batches
    
    # Ensure we have enough people for diversity (need unique people per batch)
    people_per_batch = max(2, len(selected_people) // num_batches)
    min_people_needed = people_per_batch * num_batches
    if len(selected_people) < min_people_needed:
        raise ValueError(f"Need at least {min_people_needed} people for {num_batches} batches with {people_per_batch} people each, got {len(selected_people)}")
    
    # Shuffle people and create non-overlapping batches
    import random
    shuffled_people = selected_people.copy()
    random.shuffle(shuffled_people)
    
    # Create all batch tasks
    batch_tasks = []
    for batch_idx in range(num_batches):
        # Calculate questions for this batch
        batch_questions = questions_per_batch + (1 if batch_idx < remaining_questions else 0)
        
        # Select unique people for this batch (no overlap with other batches)
        start_idx = batch_idx * people_per_batch
        end_idx = start_idx + people_per_batch
        batch_people = shuffled_people[start_idx:end_idx]
        
        # Create async task for this batch
        task = generate_questions_batch(
            batch_people, 
            prompt_template, 
            batch_questions, 
            model, 
            client,
            temperature,
            batch_idx
        )
        batch_tasks.append(task)
    
    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(task):
        async with semaphore:
            return await task
    
    # Execute all batches with progress bar
    with tqdm(total=num_batches, desc=f"Generating question batches with {model} (async)", unit="batch") as pbar:
        results = []
        for coro in asyncio.as_completed([run_with_semaphore(task) for task in batch_tasks]):
            result = await coro
            results.append(result)
            pbar.update(1)
    
    # Flatten results
    all_questions = []
    for batch_result in results:
        if batch_result:  # Skip empty results from failed batches
            all_questions.extend(batch_result)
    
    return all_questions




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
    ap.add_argument("--num-llm-people", type=int, default=5, help="Number of people to select for LLM question generation")
    ap.add_argument("--num-questions", type=int, default=10, help="Number of questions to generate")
    ap.add_argument("--llm-seed", type=int, default=42, help="Seed for random person selection")
    ap.add_argument("--prompt-file", type=Path, default=Path(__file__).parent / "question_generation_prompt.txt", help="Path to prompt template file")
    ap.add_argument("--max-batch-size", type=int, default=20, help="Maximum questions per LLM batch")
    ap.add_argument("--temperature", type=float, default=0.8, help="LLM temperature for diversity vs accuracy balance")
    ap.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent requests for parallel generation")
    
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
        
        # Select random people from existing data
        selected_people = select_random_people(golden_path, args.num_llm_people, args.llm_seed)
        print(f"📝 Selected {len(selected_people)} people for question generation")
        
        # Generate questions using LLM
        try:
            questions = generate_questions_with_llm(
                selected_people, 
                args.prompt_file,
                args.num_questions,
                args.llm_model,
                max_batch_size=args.max_batch_size,
                temperature=args.temperature,
                max_concurrent=args.max_concurrent
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

