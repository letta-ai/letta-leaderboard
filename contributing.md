# Contributing Guide

## Contributing new results
We welcome submission of new results of customized language  models for the official `letta-leaderboard`.

### Bring your own model to letta-leaderboard
To start experimenting with your language model, first, prepare a `leaderboard/llm_model_configs/{your_model}.json`. Then, start the latest letta local server on port 8283 ([tutorials](https://github.com/letta-ai/letta?tab=readme-ov-file#-run-the-letta-server)).

To run all the official `letta-leaderboard` benchmarks, run the following command:

```bash
bash run_all_benchmark.sh {your_model}
```

### Upload the result
TODO


## Contributing new tasks

### Register your own benchmark

To register your own benchmark and reuse `letta-leaderboard`'s existing infrastructure, read and inherit the `Benchmark` class in `leaderboard/{your_benchmark}/{your_benchmark}_benchmark.py`. Then, at the top level of this file, define instances of this Benchmark as `{benchmark_variable_name}`, and you can have multiple benchmark instances of different configurations.

To run the experiment, use the following command:
```python
python -m  leaderboard.evaluate --benchmark={your_bench} --benchmark_variable={benchmark_variable_name} --model=openai-gpt-4.1 ...
```

To see a list of options, try `python -m  leaderboard.evaluate --help`.

### Contriburing to letta-leaderboard

To use `letta-leaderboard` for your own use case, simply fork this repository! To submit a task for the leaderboard, submit a PR with the prefix "[leaderboard]".
