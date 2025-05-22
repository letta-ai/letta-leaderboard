# Usage:
# Before running, set "OPENAI_API_KEY" to together's OpenAI proxy key, and "OPENAI_API_KEY_NOT_PROXY" to your actual OpenAI key.
# ./run_all_models.sh "python -m leaderboard.evaluate --benchmark=letta_bench --dataset_size=200 --timeout=10 --repeat=10 --benchmark_variable=core_memory_read_benchmark"

# Define your list of models
together_models=(
    "together-llama-4-scout-17b"
    "together-qwen-2-5-72b"
    "together-llama-3-1-405b"
    "together-llama-4-maverick-17b"
    "together-deepseek-v3"
    "together-llama-3-2-3b"
    "together-llama-3-70b"
    "together-meta-llama-3-1-8b"
    "together-llama-3-3-70b"
    "together-meta-llama-3-1-70b"
    "together-qwen-2-5-7b"
)

non_together_models=(
    "claude-3-5-haiku"
    "gemini-2-5-pro"
    "claude-3-7-sonnet-extended"
    "gemini-2-5-flash"
    "openai-gpt-4.1"
    "claude-3-7-sonnet"
    "claude-4-sonnet"
    "claude-3-5-sonnet"
    "openai-gpt-4o"
    "openai-gpt-4.1-mini"
    "openai-o3-mini"
    "openai-o4-mini"
    "openai-gpt-4.1-nano"
    "openai-gpt-4o-mini"
    "openai-gpt-3.5-turbo"
)

# Check if a base command is provided
if [ -z "$1" ]; then
  echo "Error: No base command provided."
  echo "Usage: ./run_all_models.sh \"<base_command>\""
  exit 1
fi

base_command="$1"

# Loop over each model and run the command
export TOGETHER_AI_KEY="$OPENAI_API_KEY"
export OPENAI_API_KEY="$OPENAI_API_KEY_NOT_PROXY"

for model in "${non_together_models[@]}"; do
  echo "Running model: $model"
  eval "$base_command --model=$model"
done

export OPENAI_API_KEY="$TOGETHER_AI_KEY"

for model in "${together_models[@]}"; do
  echo "Running model: $model"
  eval "$base_command --model=$model"
done