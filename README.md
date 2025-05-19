# letta-leaderboard

Letta Leaderboard helps users select which language models work well in the Letta framework by reporting the performance of popular models on a series of tasks. The tasks are designed to test the core memory management functionality in Letta.  Models that are strong at function calling and aware of their limitations typically work well in Letta.


# Setup

First set the environment variables for API keys. If using OpenAI proxy, then set `OPENAI_API_KEY` to the corresponding provider used to set up the proxy, eg. Together AI. `OPENAI_API_KEY_NOT_PROXY` is used for the LLM-as-a-judge via OpenAI chat completions, set this to your OpenAI key.
```
export OPENAI_API_KEY=
export ANTHROPIC_API_KEY=
export OPENAI_API_KEY_NOT_PROXY=
```

### Benchmark methods
Benchmark class is used to encapsulate raw datasets and provide interfaces to letta apis:
```python


# here is an overview for the benchmark class
class Benchmark(metaclass=ABCMeta):
    dataset: list[Dotdict]  # TODO(shangyin): change to a more specific type
    benchmark_type: Literal["general", "feature"]  # need a way to specify feature

    @abstractmethod
    def setup_agent(self, datum: Dotdict, client: Letta, agent_id: str) -> None:
        # this prepares the agent for the current evaluation, e.g. evaluating archival memory requires inserting the context into the agent's memory
        pass

    @abstractmethod
    def metric(
        self, predicted_answer: str, true_answer: str, datum: Dotdict, agent_id=None
    ) -> float:
        pass

    # This function is used to collect the usage statistics of the agents
    # in the final result file, expect a dict with:
    # { "usage_statistics": usage_statistics}
    def get_usage_statistics(
        self, client: Letta, agent_ids: list[str], evaluation_result: EvaluationResult
    ) -> dict:
        return {}

    # this allows the benchmark to request response from the evaluating agent
    # override to get customized response
    def get_response(
        self, client: Letta, agent_id: str, datum: Dotdict
    ) -> LettaResponse:
        return super().get_response(client, agent_id, datum)
    
    # IMPORTANT: this creates custom agents to start with the evaluation
    # if not defined, the evaluator will create a default agent
    def create_agent_fun(self, client: Letta, datum, llm_config, embedding_config):
        pass
```

### To run a benchmark
```python -m  leaderboard.evaluate --benchmark=letta_bench  --dataset_size=100 --timeout=100 --repeat=3 --benchmark_variable=core_memory_benchmark --model=openai-gpt-4.1-mini```

make sure `model=...` is in `leaderboard/llm_model_configs`.
