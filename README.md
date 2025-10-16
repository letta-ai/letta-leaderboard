# Letta Leaderboard

## Repo Archived
The [Letta Leaderboard](https://docs.letta.com/leaderboard) has now integrated with [Letta-Evals](https://github.com/letta-ai/letta-evals)! Thhis repository is no longer maintained.

The [Letta Leaderboard](https://docs.letta.com/leaderboard) helps users select which language models work well in the Letta framework by reporting the performance of popular models on a series of tasks. The tasks are designed to test the core memory management functionality in Letta.  Models that are strong at function calling and aware of their limitations typically work well in Letta.

To view a table of the most recent results, visit [our docs page](https://docs.letta.com/leaderboard).

# Setup

## Start a letta server
```sh
export LETTA_PG_POOL_SIZE=25
export LETTA_PG_MAX_OVERFLOW=10
export LETTA_UVICORN_WORKERS=10
```
Then set the environment variables for API keys. If using OpenAI proxy, then set `OPENAI_API_KEY_PROXY` to the corresponding provider used to set up the proxy, eg. Together AI. 

```sh
export OPENAI_API_KEY=
export ANTHROPIC_API_KEY=
export OPENAI_API_KEY_PROXY=
export GEMINI_API_KEY=
```

When launching the experiments, please also include `OPENAI_API_KEY` for using the LLM judge.

### Benchmark methods
Benchmark class is used to encapsulate raw datasets and provide interfaces to letta apis:
```python


# here is an overview for the benchmark class
class Benchmark(metaclass=ABCMeta):
    dataset: list[Dotdict]
    benchmark_type: Literal["general", "feature"]  # need a way to specify feature

    @abstractmethod
    async def setup_agent(self, datum: Dotdict, client: Letta, agent_id: str) -> None:
        # this prepares the agent for the current evaluation, e.g. evaluating archival memory requires inserting the context into the agent's memory
        pass

    @abstractmethod
    async def metric(
        self, predicted_answer: str, true_answer: str, datum: Dotdict, agent_id=None
    ) -> float:
        pass

    # This function is used to collect the usage statistics of the agents
    # in the final result file, return a UsageStatistics(run_stat: dict[any, any], agent_stat[agent_id, any])
    async def get_usage_statistics(
        self, client: Letta, agent_ids: list[str], evaluation_result: EvaluationResult
    ) -> dict:
        return UsageStatistics({}, {agent_id: {} for agent_id in agent_ids})

    # this allows the benchmark to request response from the evaluating agent
    # override to get customized response
    # default behavior is sending a single "datum.message" 
    async def get_response(
        self, client: Letta, agent_id: str, datum: Dotdict
    ) -> LettaResponse:
        return client.agents.messages.create( 
            agent_id=agent_id,
            messages=[MessageCreate(role="user", content=datum.message)],
        )
    
    # IMPORTANT: creates custom agents to start with the evaluation
    # if not defined, the evaluator will create a default agent
    async def create_agent_fun(self, client: Letta, datum, agent_config):
        pass
```

### To run a benchmark
```sh
python -m  leaderboard.evaluate --benchmark=letta_bench  --dataset_size=100 --timeout=100 --repeat=3 --benchmark_variable=core_memory_read_benchmark --model=openai-gpt-4.1-mini
```

make sure `model=...` is in `leaderboard/llm_model_configs`.
