from abc import abstractmethod, ABCMeta
from typing import Literal
from letta_client import Letta, LettaResponse, MessageCreate


from leaderboard.utils import Dotdict, EvaluationResult


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

    def truncate_dataset(self, num_datapoints: int) -> None:
        self.dataset = self.dataset[:num_datapoints]

    def print_stats(self):
        print(f"Number of data points: {len(self.dataset)}")
        print(f"Benchmark type: {self.benchmark_type}")
        print(f"Example data point: {self.dataset[0]}")

    def get_response(
        self,
        client: Letta,
        agent_id: str,
        datum: Dotdict,
    ) -> LettaResponse:
        return client.agents.messages.create(
            agent_id=agent_id,
            messages=[
                MessageCreate(
                    role="user",
                    content=datum.message,
                )
            ],
        )

    def get_response_from_message_list(
        self,
        client: Letta,
        agent_id: str,
        datum: Dotdict,
    ) -> list[LettaResponse]:
        responses = []
        for message in datum.message_list:
            responses.append(
                client.agents.messages.create(
                    agent_id=agent_id,
                    messages=[
                        MessageCreate(
                            role="user",
                            content=message,
                        )
                    ],
                )
            )

        return responses

    def get_usage_statistics(
        self, client: Letta, agent_id: str, evaluation_result: EvaluationResult
    ) -> dict:
        # default implementation
        return {}
