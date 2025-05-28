# Usage:
# ./run_all_models.sh "python -m leaderboard.evaluate --benchmark=letta_bench --dataset_size=200 --timeout=10 --repeat=10 --benchmark_variable=core_memory_read_benchmark"

all_models=(
    "claude-3-5-haiku"
    "gemini-2-5-pro"
    "gemini-2-5-flash"
    "gemini-2-5-flash-0520"
    "openai-gpt-4.1"
    "claude-3-7-sonnet"
    "claude-4-sonnet"
    "openai-gpt-4o"
    "openai-gpt-4.1-mini"
    "openai-gpt-4.1-nano"
    "openai-gpt-4o-mini"
    "openai-o3-mini"
    "openai-o4-mini"
    "together-llama-4-scout-17b"
    "together-llama-4-maverick-17b"
    "together-deepseek-v3"
    "together-llama-3-2-3b"
    "together-meta-llama-3-1-8b"
    "together-llama-3-3-70b"
    "together-qwen-3-235b"
    # "claude-3-7-sonnet-extended" isn't working now, investigating
)


base_command="bash ./run_all_benchmark.sh"

for model in "${all_models[@]}"; do
  echo "Running model: $model"
  eval "$base_command $model"
done
