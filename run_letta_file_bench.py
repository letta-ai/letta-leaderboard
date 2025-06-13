#!/usr/bin/env python3
"""
Simple wrapper script to run letta_file_bench_gen as a module.

Example Usage:

# Generate small test dataset (20 people)
python3 run_letta_file_bench.py \
  --num-people 20 --max-addresses 2 --max-accounts 2 --max-employments 1 \
  --max-credit-cards 1 --max-vehicles 1 --max-pets 1 --max-net-accounts 2 \
  --max-insurances 1 --max-medical-records 1 --seed 123 --output-dir ./test_dataset

# Generate small question set (20 questions, balanced distribution)
python3 run_letta_file_bench.py \
  --generate-llm-questions --output-dir ./test_dataset \
  --num-questions 20 --llm-model gpt-4o --temperature 0.8 --max-concurrent 2 \
  --single-hop-pct 0.4 --multi-hop-pct 0.2 --comparison-pct 0.4 --llm-seed 456

# Generate large production dataset (500 people)
python3 run_letta_file_bench.py \
  --num-people 500 --max-addresses 5 --max-accounts 6 --max-employments 4 \
  --max-credit-cards 5 --max-vehicles 4 --max-pets 4 --max-net-accounts 8 \
  --max-insurances 4 --max-medical-records 3 --seed 12345 --output-dir ./large_dataset

# Generate large question set (500 questions, optimized distribution)
python3 run_letta_file_bench.py \
  --generate-llm-questions --output-dir ./large_dataset \
  --num-questions 500 --llm-model gpt-4o --temperature 0.75 --max-concurrent 10 \
  --single-hop-pct 0.3 --multi-hop-pct 0.3 --comparison-pct 0.4 --llm-seed 42

# Quick test workflow (data + questions)
python3 run_letta_file_bench.py --num-people 20 --seed 123 --output-dir ./test_dataset
python3 run_letta_file_bench.py --generate-llm-questions --output-dir ./test_dataset \
  --num-questions 20 --max-concurrent 2 --single-hop-pct 0.5 --multi-hop-pct 0.25 --comparison-pct 0.25
"""

import sys
import subprocess

if __name__ == "__main__":
    # Run the module with all provided arguments
    cmd = [sys.executable, "-m", "leaderboard.letta_file_bench.letta_file_bench_gen"] + sys.argv[1:]
    subprocess.run(cmd)