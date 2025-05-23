from abc import abstractmethod, ABCMeta
from typing import Literal, Optional, Callable
from letta_client import AsyncLetta, LettaResponse, MessageCreate

from leaderboard.utils import Dotdict, EvaluationResult, UsageStatistics


class Benchmark(metaclass=ABCMeta):
    dataset: list[Dotdict]
    benchmark_type: Literal["general", "feature"]
    create_agent_fun: Optional[Callable] = None

    @abstractmethod
    async def setup_agent(
        self, datum: Dotdict, client: AsyncLetta, agent_id: str
    ) -> None:
        pass

    @abstractmethod
    async def metric(
        self, predicted_answer: str, true_answer: str, datum: Dotdict, agent_id=None
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
        self, client: AsyncLetta, agent_id: str, evaluation_result: EvaluationResult
    ) -> UsageStatistics:
        return UsageStatistics({}, {})
