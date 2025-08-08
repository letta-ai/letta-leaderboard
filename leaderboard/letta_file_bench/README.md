## Running the Benchmark

```bash
python3 -m leaderboard.evaluate \
    --benchmark=letta_file_bench \
    --benchmark_variable=file_open_benchmark \
    --questions_file=/Users/mattzhou/letta-leaderboard/leaderboard/letta_file_bench/data/generated_questions/run_20250729_150617/agent_generated_questions.jsonl \
    --model=claude-4-sonnet
```

## Generating New Questions

```bash
python3 -m leaderboard.letta_file_bench.agent_question_generator \
    --num-questions 25 \
    --model claude-sonnet-4-20250514 \
    --db-path leaderboard/letta_file_bench/data/letta_file_bench.db \
    --output-dir leaderboard/letta_file_bench/data/generated_questions
```

### Parameters for Question Generation

- `--num-questions`: Number of questions to generate (default: 25)
- `--model`: Model to use for question generation
- `--db-path`: Path to the SQLite database file
- `--output-dir`: Directory to save generated questions

### Parameters for Running Benchmark

- `--benchmark`: Benchmark name (letta_file_bench)
- `--benchmark_variable`: Specific benchmark to run (e.g., file_open_benchmark)
- `--questions_file`: Path to the JSONL file containing questions
- `--model`: Model to evaluate