from typing import List, Dict

from letta_client import (
    CreateBlock,
    Letta,
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
    """
    Base class: loads dataset and provides default pipeline hooks.
    Subclasses should override setup_agent, create_agent_fun, get_response, and get_usage_statistics.
    """

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
        for i, x in enumerate(self.raw_datasets):
            for q, a, idxs in zip(
                x["question"], x["answer"], x["supporting_fact_indices"]
            ):
                data.append(
                    Dotdict(
                        {
                            "message": q,
                            "message_list": [q],
                            "answer": a,
                            "supporting_fact": x["facts"],
                            "name": x["name"],
                            "supporting_fact_indices": [int(i) for i in idxs],
                            "contridicting_facts": x.get("contridicting_facts", []),
                        }
                    )
                )
        return data

    def setup_agent(self, datum: Dotdict, client: Letta, agent_id: str):
        # default: no special setup
        pass

    def create_agent_fun(
        self,
        client: Letta,
        datum: Dotdict,
        llm_config,
        embedding_config,
    ) -> str:
        # default: create agent without initial memory
        return client.agents.create(
            llm_config=llm_config, embedding_config=embedding_config
        ).id

    def get_response(
        self, client: Letta, agent_id: str, datum: Dotdict
    ) -> LettaResponse:
        return super().get_response(client, agent_id, datum)

    def metric(self, predicted: str, true: str, datum: Dotdict, agent_id: str) -> float:
        # default scoring via grade_sample
        result = grade_sample(datum.message, true, predicted)
        return 1.0 if result == "A" else 0.0


class ArchivalMemoryReadBenchmark(LettaBenchmark):
    """
    Agents preload both provided supporting facts and a set of obvious facts into archival memory.
    """

    def setup_agent(self, datum: Dotdict, client: Letta, agent_id: str):
        for fact in datum.supporting_fact:
            client.agents.passages.create(agent_id=agent_id, text=fact)
        for fact in obvious_facts:
            client.agents.passages.create(agent_id=agent_id, text=fact)

    def get_usage_statistics(
        self,
        client: Letta,
        agent_ids: List[str],
        evaluation_result: EvaluationResult,
    ) -> UsageStatistics:
        return UsageStatistics(
            total_archival_usage(
                client, agent_ids, evaluation_result.individual_scores
            ),
            {},
        )


class CoreMemoryReadBenchmark(LettaBenchmark):
    """
    Agents created with a core memory block of supporting facts. Supports a "hard" mode where a few trick questions
    precede the real query in message_list to test memory recall under noise.
    """

    def __init__(self, hard: bool = False):
        self.hard = hard
        super().__init__()

    def _build_dataset(self) -> List[Dotdict]:
        data: List[Dotdict] = []
        for i, x in enumerate(self.raw_datasets):
            for q, a, idxs in zip(
                x["question"], x["answer"], x["supporting_fact_indices"]
            ):
                msgs: List[str] = []
                if self.hard:
                    j = i
                    while True:
                        j += 1
                        cand = self.raw_datasets[j % len(self.raw_datasets)][
                            "question"
                        ][:3]
                        if cand:
                            msgs = cand
                            break
                msgs.append(q)
                data.append(
                    Dotdict(
                        {
                            "message": q,
                            "message_list": msgs,
                            "answer": a,
                            "supporting_fact": x["facts"],
                            "name": x["name"],
                            "supporting_fact_indices": [int(i) for i in idxs],
                            "contridicting_facts": x.get("contridicting_facts", []),
                        }
                    )
                )
        return data

    def create_agent_fun(
        self,
        client: Letta,
        datum: Dotdict,
        llm_config,
        embedding_config,
    ) -> str:
        block = CreateBlock(
            label="Supporting Facts",
            value="\n".join(f"{i}. {f}" for i, f in enumerate(datum.supporting_fact)),
        )
        agent = client.agents.create(
            llm_config=llm_config,
            embedding_config=embedding_config,
            memory_blocks=[block],
        )
        return agent.id

    def get_response(
        self, client: Letta, agent_id: str, datum: Dotdict
    ) -> LettaResponse:
        if self.hard:
            return super().get_response_from_message_list(client, agent_id, datum)[-1]
        return super().get_response(client, agent_id, datum)


class CoreMemoryWriteBenchmark(LettaBenchmark):
    """
    Agents start with persona instructions and empty person-specific blocks, then write core memory entries
    based on observed facts. Supports a "hard" mode where persona instructions are omitted.
    """

    def __init__(self, hard: bool = False):
        self.hard = hard
        super().__init__()

    def create_agent_fun(
        self,
        client: Letta,
        datum: Dotdict,
        llm_config,
        embedding_config,
    ) -> str:
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
        state = client.agents.create(
            llm_config=llm_config,
            embedding_config=embedding_config,
            memory_blocks=persona + person_blocks,
        )
        agent_id = state.id
        self.agent_datum_mapping[agent_id] = datum
        for idx in datum.supporting_fact_indices:
            client.agents.messages.create(
                agent_id=agent_id,
                messages=[
                    MessageCreate(role="user", content=datum.supporting_fact[idx])
                ],
            )
        self.agent_core_memory_messages[agent_id] = client.agents.messages.list(
            agent_id=agent_id, limit=1000
        )
        client.agents.messages.reset(agent_id=agent_id)
        return agent_id

    def get_usage_statistics(
        self,
        client: Letta,
        agent_ids: List[str],
        evaluation_result: EvaluationResult,
    ) -> dict:
        return UsageStatistics(
            {},
            {
                agent_id: {"memory_messages":[m.model_dump(mode="json") for m in messages]}
                for agent_id, messages in self.agent_core_memory_messages.items()
            },
        )


# Final benchmark instances (names preserved)
archival_memory_read_benchmark = ArchivalMemoryReadBenchmark()
core_memory_read_benchmark = CoreMemoryReadBenchmark()
core_memory_read_benchmark_hard = CoreMemoryReadBenchmark(hard=True)
core_memory_write_benchmark = CoreMemoryWriteBenchmark()
core_memory_write_benchmark_hard = CoreMemoryWriteBenchmark(hard=True)
core_memory_update_benchmark = LettaBenchmark()
