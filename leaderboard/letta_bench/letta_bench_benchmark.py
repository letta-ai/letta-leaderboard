from typing import List, Dict

from letta_client import (
    CreateBlock,
    AsyncLetta,
    LettaMessageUnion,
    LettaResponse,
    MessageCreate,
)
from leaderboard.benchmark import Benchmark
from datasets import load_dataset
from leaderboard.evaluate import EvaluationResult
from leaderboard.utils import (
    Dotdict,
    total_archival_usage,
    grade_sample,
    UsageStatistics,
)

# Shared facts used by archival benchmarks
obvious_facts = [
    "The sun rises in the east.",
    "Water is wet.",
    "Fire is hot.",
    "The Earth is round.",
    "Humans need oxygen to breathe.",
    "2 + 2 equals 4.",
    "Gravity pulls objects toward the ground.",
    "Ice melts when heated.",
    "The sky appears blue during the day.",
    "Time moves forward.",
]


class LettaBenchmark(Benchmark):
    def __init__(self):
        raw = load_dataset(
            "json",
            data_files="leaderboard/letta_bench/letta_bench_gen_200.jsonl",
        )["train"]
        self.raw_datasets = raw
        self.agent_core_memory_messages: Dict[str, List[LettaMessageUnion]] = {}
        self.agent_datum_mapping: Dict[str, Dotdict] = {}
        self.core_memory_agent_state: Dict[str, str] = {}
        self.dataset = self._build_dataset()
        self.benchmark_type = "feature"

    def _build_dataset(self) -> List[Dotdict]:
        data: List[Dotdict] = []

        for raw_datum in self.raw_datasets:
            facts = raw_datum["facts"]
            name = raw_datum["name"]

            for (
                question,
                answer,
                support_indices,
                contradiction,
                contradiction_answer,
            ) in zip(
                raw_datum["question"],
                raw_datum["answer"],
                raw_datum["supporting_fact_indices"],
                raw_datum["contradicting_facts"],
                raw_datum["contradicting_answers"],
            ):
                data.append(
                    Dotdict(
                        {
                            "message": question,
                            "message_list": [question],
                            "answer": answer,
                            "supporting_fact": facts,
                            "name": name,
                            "supporting_fact_indices": [
                                int(idx) for idx in support_indices
                            ],
                            "contradicting_fact": contradiction,
                            "contradicting_answer": contradiction_answer,
                        }
                    )
                )

        return data

    async def setup_agent(self, datum: Dotdict, client: AsyncLetta, agent_id: str):
        pass

    async def create_agent_fun(
        self,
        client: AsyncLetta,
        datum: Dotdict,
        agent_config: dict,
    ) -> str:
        # Ensure agent_config contains required keys
        assert "llm_config" in agent_config, "agent_config must contain 'llm_config'"
        assert "embedding_config" in agent_config, "agent_config must contain 'embedding_config'"
        
        return (
            await client.agents.create(**agent_config)
        ).id

    async def get_response(
        self, client: AsyncLetta, agent_id: str, datum: Dotdict
    ) -> LettaResponse:
        return await super().get_response(client, agent_id, datum)

    async def metric(
        self, predicted: str, true: str, datum: Dotdict, agent_id: str
    ) -> float:
        result = await grade_sample(datum.message, true, predicted)
        return 1.0 if result == "A" else 0.0


class ArchivalMemoryReadBenchmark(LettaBenchmark):

    # async def create_agent_fun(
    #     self,
    #     client: AsyncLetta,
    #     datum: Dotdict,
    #     agent_config: dict,
    # ) -> str:
    #     memory_blocks = [CreateBlock(label="Persona", value="You are trying to answer a question about a person. Search the archival memory for the answer if you don't know the answer.")]
    #     agent = await client.agents.create(
    #         memory_blocks=memory_blocks,
    #         **agent_config,
    #     )
    #     return agent.id

    async def setup_agent(self, datum: Dotdict, client: AsyncLetta, agent_id: str):
        for fact in datum.supporting_fact:
            await client.agents.passages.create(agent_id=agent_id, text=fact)
        # for fact in obvious_facts:
        #     await client.agents.passages.create(agent_id=agent_id, text=fact)

    async def get_usage_statistics(
        self,
        client: AsyncLetta,
        agent_ids: List[str],
        evaluation_result: EvaluationResult,
    ) -> UsageStatistics:
        return UsageStatistics(
            await total_archival_usage(
                client, agent_ids, evaluation_result.individual_scores
            ),
            {},
        )


class CoreMemoryReadBenchmark(LettaBenchmark):
    def __init__(self, hard: bool = False):
        self.hard = hard
        super().__init__()

    def _build_dataset(self) -> List[Dotdict]:
        data: List[Dotdict] = []

        for dataset_index, raw_datum in enumerate(self.raw_datasets):
            facts = raw_datum["facts"]
            entry_name = raw_datum["name"]

            for question, answer in zip(
                raw_datum["question"],
                raw_datum["answer"],
            ):
                message_list: List[str] = []

                if self.hard:
                    next_index = dataset_index
                    while True:
                        next_index = (next_index + 1) % len(self.raw_datasets)
                        candidates = self.raw_datasets[next_index]["question"][:3]
                        if candidates:
                            message_list = candidates
                            break

                message_list.append(question)
                data.append(
                    Dotdict(
                        {
                            "message": question,
                            "message_list": message_list,
                            "answer": answer,
                            "supporting_fact": facts,
                            "name": entry_name,
                        }
                    )
                )

        return data

    async def create_agent_fun(
        self,
        client: AsyncLetta,
        datum: Dotdict,
        agent_config: dict,
    ) -> str:
        # Ensure agent_config contains required keys
        assert "llm_config" in agent_config, "agent_config must contain 'llm_config'"
        assert "embedding_config" in agent_config, "agent_config must contain 'embedding_config'"
        
        block = CreateBlock(
            label="Supporting Facts",
            value="\n".join(f"{i}. {f}" for i, f in enumerate(datum.supporting_fact)),
        )
        agent = await client.agents.create(
            memory_blocks=[block],
            **agent_config,
        )
        return agent.id

    async def get_response(
        self, client: AsyncLetta, agent_id: str, datum: Dotdict
    ) -> LettaResponse:
        if self.hard:
            return (
                await super().get_response_from_message_list(client, agent_id, datum)
            )[-1]
        return await super().get_response(client, agent_id, datum)


class CoreMemoryWriteBenchmark(LettaBenchmark):
    def __init__(self, hard: bool = False):
        self.hard = hard
        super().__init__()

    async def create_agent_fun(
        self,
        client: AsyncLetta,
        datum: Dotdict,
        agent_config: dict,
    ) -> str:
        # Ensure agent_config contains required keys
        assert "llm_config" in agent_config, "agent_config must contain 'llm_config'"
        assert "embedding_config" in agent_config, "agent_config must contain 'embedding_config'"
        
        names = [n for n in datum.name if n.lower() in datum.message.lower()]
        persona = []
        if not self.hard:
            persona = [
                CreateBlock(
                    label="Persona",
                    value=(
                        f"You need to take notes about facts on specific person: {', '.join(names)}."
                    ),
                )
            ]
        person_blocks = [CreateBlock(label=n, value="") for n in names]
        state = await client.agents.create(
            memory_blocks=persona + person_blocks,
            **agent_config,
        )
        agent_id = state.id
        self.agent_datum_mapping[agent_id] = datum
        for idx in datum.supporting_fact_indices:
            await client.agents.messages.create(
                agent_id=agent_id,
                messages=[
                    MessageCreate(role="user", content=datum.supporting_fact[idx])
                ],
            )
        self.agent_core_memory_messages[agent_id] = await client.agents.messages.list(
            agent_id=agent_id, limit=1000
        )
        await client.agents.messages.reset(agent_id=agent_id)
        return agent_id

    async def get_usage_statistics(
        self,
        client: AsyncLetta,
        agent_ids: List[str],
        evaluation_result: EvaluationResult,
    ) -> UsageStatistics:
        return UsageStatistics(
            {},
            {
                agent_id: {
                    "memory_messages": [m.model_dump(mode="json") for m in messages]
                }
                for agent_id, messages in self.agent_core_memory_messages.items()
                if agent_id in agent_ids
            },
        )


class CoreMemoryUpdateBenchmark(LettaBenchmark):
    def __init__(self):
        super().__init__()
        self.agent_core_memory_update_messages = {}

    def _build_dataset(self):
        dataset = super()._build_dataset()
        for datum in dataset:
            datum.answer = datum.contradicting_answer
        return dataset

    async def create_agent_fun(
        self,
        client: AsyncLetta,
        datum: Dotdict,
        agent_config: dict,
    ) -> str:
        # Ensure agent_config contains required keys
        assert "llm_config" in agent_config, "agent_config must contain 'llm_config'"
        assert "embedding_config" in agent_config, "agent_config must contain 'embedding_config'"
        
        block = CreateBlock(
            label="Supporting Facts",
            value="\n".join(f"{i}. {f}" for i, f in enumerate(datum.supporting_fact)),
        )
        agent = await client.agents.create(
            memory_blocks=[block],
            **agent_config,
        )
        return agent.id

    async def setup_agent(self, datum, client, agent_id):
        await client.agents.messages.create(
            agent_id=agent_id,
            messages=[MessageCreate(role="user", content=datum.contradicting_fact)],
        )
        self.agent_core_memory_update_messages[
            agent_id
        ] = await client.agents.messages.list(agent_id=agent_id, limit=1000)
        await client.agents.messages.reset(agent_id=agent_id)

    async def get_usage_statistics(
        self,
        client: AsyncLetta,
        agent_ids: List[str],
        evaluation_result: EvaluationResult,
    ) -> UsageStatistics:
        return UsageStatistics(
            {},
            {
                agent_id: {
                    "memory_messages": [m.model_dump(mode="json") for m in messages]
                }
                for agent_id, messages in self.agent_core_memory_update_messages.items()
                if agent_id in agent_ids
            },
        )


# Final benchmark instances (names preserved)
archival_memory_read_benchmark = ArchivalMemoryReadBenchmark()
core_memory_read_benchmark = CoreMemoryReadBenchmark()
core_memory_read_benchmark_hard = CoreMemoryReadBenchmark(hard=True)
core_memory_write_benchmark = CoreMemoryWriteBenchmark()
core_memory_write_benchmark_hard = CoreMemoryWriteBenchmark(hard=True)
core_memory_update_benchmark = CoreMemoryUpdateBenchmark()
