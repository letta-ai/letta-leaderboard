from typing import List
from letta_client import (
    CreateBlock,
    Letta,
    LettaMessageUnion,
    LettaResponse,
    MessageCreate,
)
from leaderboard.benchmark import Benchmark
from datasets import load_dataset
from rich import print
from leaderboard.evaluate import EvaluationResult
from leaderboard.utils import Dotdict, total_archival_usage, grade_sample

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


class NeuralDatabaseBenchmark(Benchmark):
    def __init__(
        self,
        test_archival_memory=False,
        test_confidence=False,
        test_core_memory_in_memory=False,
        test_core_memory_append=False,
        test_core_memory_hard=False,
    ):
        raw_datasets = load_dataset(
            "json",
            data_files="leaderboard/letta_bench/letta_bench_gen_200.jsonl",
        )["train"]

        self.test_confidence = test_confidence
        self.test_archival_memory = test_archival_memory
        self.test_core_memory_in_memory = test_core_memory_in_memory
        self.test_core_memory_hard = test_core_memory_hard

        self.test_core_memory_append = test_core_memory_append
        self.agent_core_memory_messages: dict[str, List[LettaMessageUnion]] = {}
        self.agent_datum_mapping: dict[str, Dotdict] = {}
        # map from supporting fact to agent id
        self.core_memory_agent_state: dict[str, str] = {}

        self.dataset = []
        for i, x in enumerate(raw_datasets):
            for question, answer, supporting_fact_indices in zip(
                x["question"], x["answer"], x["supporting_fact_indices"]
            ):
                if test_confidence:
                    message = question + (
                        "DO NOT ANSWER. Give a confidence score between 0 and 1 only. "
                        "Just answer with the confidence score."
                    )
                else:
                    message = question

                message_list = []
                if test_core_memory_hard:
                    trick_question_idx = i
                    while True:
                        trick_question_idx += 1
                        trick_questions = raw_datasets[
                            trick_question_idx % len(raw_datasets)
                        ]["question"][:3]
                        if len(trick_questions):
                            break
                    message_list = trick_questions
                message_list.append(message)

                self.dataset.append(
                    Dotdict(
                        {
                            "message": message,
                            "message_list": message_list,
                            "answer": answer,
                            "supporting_fact": x["facts"],
                            "name": x["name"],
                            "supporting_fact_indices": [
                                int(i) for i in supporting_fact_indices
                            ],
                        }
                    )
                )

        self.benchmark_type = "feature"

    def setup_agent(self, datum, client, agent_id):
        if self.test_core_memory_in_memory:
            # because we already setup the corememory on creation, we don't need to setup again
            return
            self.setup_agent_core_memory(datum, client, agent_id)
        elif self.test_archival_memory:
            # defaulting to test archival memory
            self.setup_agent_archival_memory(datum, client, agent_id)
        elif self.test_core_memory_append:
            return

    def setup_agent_archival_memory(self, datum, client: Letta, agent_id):
        for fact in datum.supporting_fact:
            client.agents.passages.create(agent_id=agent_id, text=fact)

        for fact in obvious_facts:
            client.agents.passages.create(agent_id=agent_id, text=fact)

    def setup_agent_core_memory(self, datum, client: Letta, agent_id):
        # formatting the supporting facts
        memory_block_str = "\n".join(
            map(
                lambda i, f: f"{i}. {f}",
                range(len(datum.supporting_fact)),
                datum.supporting_fact,
            )
        )
        block = client.blocks.create(label="Supporting Facts", value=memory_block_str)
        client.agents.blocks.attach(agent_id=agent_id, block_id=block.id)

    def metric(self, predicted_answer, true_answer, datum, agent_id):
        if self.test_confidence:
            # parse the first float from the predicted_answer (might be a whole sentence like The confidence is 0.8)
            confidence = 0.0
            for word in predicted_answer.split():
                try:
                    confidence = float(word)
                    if confidence > 1.0:
                        confidence = 1.0
                except ValueError:
                    continue
            return confidence

        result = grade_sample(datum.message, true_answer, predicted_answer)
        # TODO(shangyin) need to log incorrect versus not attempt
        print(
            "\n[red]Predicted Answer: "
            + str(predicted_answer)
            + "[/red]"
            + "\n"
            + "[green]True Answer: "
            + str(true_answer)
            + "[/green]"
            + "\n"
            + "[red]Correct?"
            + str(result == "A")
            + "[/red]\n"
            + "[cyan]agent_id:"
            + str(agent_id)
            + "[/cyan]\n"
        )
        return 1.0 if result == "A" else 0.0

    def simple_metric(self, predicted_answer, true_answer, datum):
        print(
            "\n[red]Predicted Answer: "
            + str(predicted_answer)
            + "[/red]"
            + "\n"
            + "[green]True Answer: "
            + str(true_answer)
            + "[/green]"
            + "\n"
            + "[red]Correct?"
            + str(true_answer in predicted_answer)
            + "[/red]\n"
        )

        return 1.0 if true_answer in predicted_answer else 0.0

    def create_agent_fun(self, client: Letta, datum, llm_config, embedding_config):
        if self.test_core_memory_in_memory or self.test_core_memory_hard:
            supporting_fact_block = CreateBlock(
                label="Supporting Facts",
                value="\n".join(
                    map(
                        lambda i, f: f"{i}. {f}",
                        range(len(datum.supporting_fact)),
                        datum.supporting_fact,
                    )
                ),
            )
            agent_id = client.agents.create(
                llm_config=llm_config,
                embedding_config=embedding_config,
                memory_blocks=[supporting_fact_block],
            ).id
        elif self.test_core_memory_append:
            # TODO(shangyin): this is not working, because we are using multi-thread
            # and the agent with the same fact will be created multiple times
            # first_supporting_fact = datum.supporting_fact[0]
            # if first_supporting_fact in self.core_memory_agent_state:
            #     return self.core_memory_agent_state[first_supporting_fact]

            # create a persona block, with instruction on taking notes about specific person
            # then create empty persons block in core memory
            person_names = [n for n in datum.name if n.lower() in datum.message.lower()]
            persona_block = CreateBlock(
                label="Persona",
                value=f"You need to take notes about facts on specific person, namely: {', '.join(person_names)}. Do not rely on message history as they may get wiped. Only take notes if the facts contain any one of {', '.join(person_names)}.",
            )

            # exclude the relevant supporting facts
            unused_facts = [
                fact
                for idx, fact in enumerate(datum.supporting_fact)
                if idx not in datum.supporting_fact_indices
            ]

            personal_name_facts = {name: [] for name in person_names}
            for i, fact in enumerate(unused_facts):
                for name in person_names:
                    if name.lower() in fact.lower():
                        personal_name_facts[name].append(fact)

            persons_block = [
                CreateBlock(label=name, value="\n".join(facts))
                for name, facts in personal_name_facts.items()
            ]

            with open("leaderboard/letta_bench/core_memory_agent.txt", "r") as f:
                core_memory_agent_system = f.read()
            agent_state = client.agents.create(
                system=core_memory_agent_system,
                llm_config=llm_config,
                embedding_config=embedding_config,
                memory_blocks=[persona_block] + persons_block,
            )
            agent_id = agent_state.id

            archival_tool_ids = [
                tool.id for tool in agent_state.tools if "archival" in tool.name
            ]

            for archival_tool_id in archival_tool_ids:
                client.agents.tools.detach(agent_id=agent_id, tool_id=archival_tool_id)

            self.agent_datum_mapping[agent_id] = datum

            for i in datum.supporting_fact_indices:
                client.agents.messages.create(
                    agent_id=agent_id,
                    messages=[
                        MessageCreate(
                            role="user",
                            content=datum.supporting_fact[i],
                        )
                    ],
                )

            self.agent_core_memory_messages[agent_id] = client.agents.messages.list(
                agent_id=agent_id,
                limit=1000,
            )

            client.agents.messages.reset(agent_id=agent_id)
        else:
            agent_id = client.agents.create(
                llm_config=llm_config, embedding_config=embedding_config
            ).id
        return agent_id

    def get_response(
        self, client: Letta, agent_id: str, datum: Dotdict
    ) -> LettaResponse:
        if self.test_core_memory_hard:
            return super().get_response_from_message_list(client, agent_id, datum)[-1]
        return super().get_response(client, agent_id, datum)

    def get_usage_statistics(
        self, client: Letta, agent_ids: list[str], evaluation_result: EvaluationResult
    ) -> dict:
        # TODO(shangyin): implement this
        if self.test_archival_memory:
            return total_archival_usage(
                client, agent_ids, evaluation_result.individual_scores
            )
        # control group for archival memory, lower is better
        elif self.test_core_memory_in_memory:
            return total_archival_usage(
                client, agent_ids, evaluation_result.individual_scores
            )
        # control group for core memory
        elif self.test_core_memory_append:
            return self._get_core_memory_usage(agent_ids)

    def _get_core_memory_usage(self, agent_ids: list[str]) -> dict:

        successful_core_memory_appends_all = []
        golden_core_memory_appends_all = []
        total_core_memory_appends_all = []

        for agent_id in agent_ids:
            datum = self.agent_datum_mapping[agent_id]
            person_names = [n for n in datum.name if n in datum.message]
            messages = self.agent_core_memory_messages[agent_id]

            successful_core_memory_appends = 0
            golden_core_memory_appends = 0
            total_core_memory_appends = 0
            for message in messages:
                if (
                    message.message_type == "tool_call_message"
                    and message.tool_call.name == "core_memory_append"
                ):
                    total_core_memory_appends += 1
                    # this is a good core memory append
                    if any(
                        name in message.tool_call.arguments for name in person_names
                    ):
                        # TODO(shangyin): this is too relexed! Need to check if it succeed, and in the right block.
                        successful_core_memory_appends += 1

            for fact in datum.supporting_fact:
                if any(name in fact for name in person_names):
                    golden_core_memory_appends += 1

            successful_core_memory_appends_all.append(successful_core_memory_appends)
            golden_core_memory_appends_all.append(golden_core_memory_appends)
            total_core_memory_appends_all.append(total_core_memory_appends)

        return {
            "successful_core_memory_appends": successful_core_memory_appends_all,
            "golden_core_memory_appends": golden_core_memory_appends_all,
            "total_core_memory_appends": total_core_memory_appends_all,
        }


# configuration: testing archival memory
archival_benchmark = NeuralDatabaseBenchmark(test_archival_memory=True)
confidence_benchmark = NeuralDatabaseBenchmark(test_confidence=True)
# configuration: before creating the agent, we put all supporting facts in the core memory
core_memory_benchmark = NeuralDatabaseBenchmark(test_core_memory_in_memory=True)
core_memory_benchmark_hard = NeuralDatabaseBenchmark(test_core_memory_hard=True)

core_memory_append_benchmark = NeuralDatabaseBenchmark(test_core_memory_append=True)
