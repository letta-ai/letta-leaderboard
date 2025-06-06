#!/bin/bash
#  bash ./run_all_benchmark.sh gemini-2-5-flash gemini-2-5-pro ..

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_name1> [<model_name2> ...]"
    exit 1
fi

all_benchmarks=(
    "core_memory_read_benchmark"
    "core_memory_write_benchmark_hard"
    "archival_memory_read_benchmark"
    "core_memory_update_benchmark"
)

dataset_size=100
out_dir="results_$(date +%m%d)"

for model in "$@"; do
    echo "Evaluating model: $model"

    for benchmark in "${all_benchmarks[@]}"; do
        echo "Running benchmark: $benchmark"
        start_time=$(date +%s)

        ( set -x; python -m leaderboard.evaluate \
            --benchmark=letta_bench \
            --benchmark_variable="$benchmark" \
            --dataset_size="$dataset_size" \
            --timeout=300 \
            --repeat=3 \
            --max_concurrency=100 \
            --model="$model" \
            --out_dir="$out_dir") # max_concurrency needs to be lower if ratelimit is low on LLM api key
    done

    # Collect results for all benchmarks at once
    echo "- model: $model"
    python -m leaderboard.utils \
        --get_results_for_model "$model" \
        --result_dir "$out_dir"
done
