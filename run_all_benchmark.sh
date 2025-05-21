#!/bin/bash
#  bash ./run_all_benchmark.sh gemini-2-5-flash gemini-2-5-pro ..

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_name1> [<model_name2> ...]"
    exit 1
fi

all_benchmarks=(
    "archival_memory_read_benchmark"
    "core_memory_write_benchmark"
    "core_memory_read_benchmark"
)

dataset_size=1

for model in "$@"; do
    echo "Evaluating model: $model"

    declare -A scores
    total_input_tokens=0
    total_output_tokens=0

    for benchmark in "${all_benchmarks[@]}"; do
        echo "Running benchmark: $benchmark"
        python -m leaderboard.evaluate \
            --benchmark=letta_bench \
            --benchmark_variable="$benchmark" \
            --dataset_size="$dataset_size" \
            --timeout=300 \
            --repeat=1 \
            --num_threads=16 \
            --model="$model"

        result_dir="results/letta_bench_${benchmark}_${dataset_size}"
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
