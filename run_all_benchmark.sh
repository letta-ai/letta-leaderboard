#!/bin/bash
#  bash ./run_all_benchmark.sh gemini-2-5-flash gemini-2-5-pro .. (more models, as you wish)

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_name1> [<model_name2> ...]"
    exit 1
fi

all_benchmarks=(
    "archival_benchmark"
    "core_memory_append_benchmark"
    "core_memory_benchmark"
    "core_memory_benchmark_hard"
)

for model in "$@"; do
    echo "Evaluating model: $model"
    for benchmark in "${all_benchmarks[@]}"; do
        echo "Running benchmark: $benchmark"
        python -m leaderboard.evaluate \
            --benchmark=letta_bench \
            --benchmark_variable="$benchmark" \
            --dataset_size=100 \
            --timeout=300 \
            --repeat=3 \
            --num_threads=16 \
            --model="$model"
    done
done
