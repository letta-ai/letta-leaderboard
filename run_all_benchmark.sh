#!/bin/bash
#  bash ./run_all_benchmark.sh gemini-2-5-flash gemini-2-5-pro ..

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_name1> [<model_name2> ...]"
    exit 1
fi

all_benchmarks=(
    "core_memory_write_benchmark_hard"
    "core_memory_read_benchmark"
    "archival_memory_read_benchmark"
    "core_memory_update_benchmark"
)

dataset_size=100
out_dir="results_$(date +%m%d)"

for model in "$@"; do
    echo "Evaluating model: $model"

    declare -A scores
    total_input_tokens=0
    total_output_tokens=0

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

        result_dir="${out_dir}/letta_bench_${benchmark}_${dataset_size}"
        stats=$(python -m leaderboard.utils \
            --get_results_for_model "$model" \
            --result_dir "$result_dir")

        # Parse the output: model,mean_score,input_tokens,output_tokens
        IFS=',' read -r parsed_model mean input_toks output_toks <<< "$stats"
        scores["$benchmark"]=$mean
        total_input_tokens=$((total_input_tokens + input_toks))
        total_output_tokens=$((total_output_tokens + output_toks))
    done

    # Compute average score
    total_score=0
    for benchmark in "${all_benchmarks[@]}"; do
        total_score=$(echo "$total_score + ${scores[$benchmark]}" | bc)
    done
    average=$(echo "scale=2; $total_score / ${#all_benchmarks[@]}" | bc)

    # Print YAML result
    echo "- model: $model"
    echo "  average: $average"
    echo "  total_input_tokens: $total_input_tokens"
    echo "  total_output_tokens: $total_output_tokens"
    for benchmark in "${all_benchmarks[@]}"; do
        echo "  $benchmark: ${scores[$benchmark]}"
    done
done
