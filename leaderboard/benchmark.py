from abc import abstractmethod, ABCMeta
from typing import Literal, Optional, Callable
from letta_client import AsyncLetta, LettaResponse, MessageCreate

from leaderboard.utils import Dotdict, EvaluationResult, UsageStatistics


class Benchmark(metaclass=ABCMeta):
    dataset: list[Dotdict]
    benchmark_type: Literal["general", "feature"]
    create_agent_fun: Optional[Callable] = None
    tool_functions: list[Callable] = []
    tool_names: list[str] = []

    @abstractmethod
    async def setup_agent(
        self, datum: Dotdict, client: AsyncLetta, agent_id: str
    ) -> None:
        pass

    async def setup_tools(self, client: AsyncLetta) -> None:
        for tool_function in self.tool_functions:
            client.tools.delete
            tool = await client.tools.upsert_from_function(
                func=tool_function
            )
            self.tool_names.append(tool.name)


    @abstractmethod
    async def metric(
        self, predicted: str, true: str, datum: Dotdict, agent_id: str
    ) -> float:
        pass

    def truncate_dataset(self, num_datapoints: int) -> None:
        self.dataset = self.dataset[:num_datapoints]

    def print_stats(self):
        print(f"Number of data points: {len(self.dataset)}")
        print(f"Benchmark type: {self.benchmark_type}")
        print(f"Example data point: {self.dataset[0]}")

    async def get_response(
        self,
        client: AsyncLetta,
        agent_id: str,
        datum: Dotdict,
    ) -> LettaResponse:
        return await client.agents.messages.create(
            agent_id=agent_id,
            messages=[
                MessageCreate(
                    role="user",
                    content=datum.message,
                )
            ],
        )

    async def get_response_from_message_list(
        self,
        client: AsyncLetta,
        agent_id: str,
        datum: Dotdict,
    ) -> list[LettaResponse]:
        responses = []
        for message in datum.message_list:
            response = await client.agents.messages.create(
                agent_id=agent_id,
                messages=[
                    MessageCreate(
                        role="user",
                        content=message,
                    )
                ],
            )
            responses.append(response)
        return responses

    async def get_usage_statistics(
        self, client: AsyncLetta, agent_ids: list[str], evaluation_result: EvaluationResult
    ) -> UsageStatistics:
        return UsageStatistics({}, {agent_id: {} for agent_id in agent_ids})

    # IMPORTANT: creates custom agents to start with the evaluation
    # if not defined, the evaluator will create a default agent
    # agent_config should contain llm_config, embedding_config and any other agent creation parameters
    # Implementations should include assertions to validate required keys are present
    # async def create_agent_fun(self, client: AsyncLetta, datum: Dotdict, agent_config: dict) -> str:
    #     pass
