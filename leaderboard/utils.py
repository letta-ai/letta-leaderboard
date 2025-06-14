import argparse
import asyncio
from dataclasses import dataclass
import json
import os
import re
from typing import Dict, List

import openai

from letta_client import AsyncLetta
from rich import print


@dataclass
class EvaluationResult:
    score: float
    input_tokens: int
    output_tokens: int
    agent_ids: list[str]
    individual_scores: list[float]
    # agent_id -> (input_message, output_messages, all_messages)
    history: dict[str, tuple[str, list, list]]
    # benchmark specific
    usage: dict = None


@dataclass
class UsageStatistics:
    run_stat: dict
    agent_stat: dict


async def total_archival_usage(
    client: AsyncLetta, agent_ids: list[str], individual_scores: list[float]
):
    total_archival_memory = 0
    total_archival_score = 0
    agent_archival_ids = []
    for agent_id, score in zip(agent_ids, individual_scores):
        messages = await client.agents.messages.list(agent_id=agent_id)
        for message in messages:
            if (
                message.message_type == "tool_call_message"
                and message.tool_call.name == "archival_memory_search"
            ):
                total_archival_memory += 1
                total_archival_score += score
                agent_archival_ids.append(agent_id)
                break
    return {
        "total_archival_count": total_archival_memory,
        "total_archival_score": total_archival_score,
        "agent_archival_ids": agent_archival_ids,
    }


async def add_archival_usage_to_json(
    client: AsyncLetta,
    agent_ids: list[str],
    individual_scores: list[float],
    file_path: str,
):
    with open(file_path) as f:
        data = json.load(f)
    data["archival_usage"] = await total_archival_usage(
        client, agent_ids, individual_scores
    )
    with open(file_path, "w") as f:
        json.dump(data, f)


def write_result_to_json(
    result: EvaluationResult, client_settings: dict, model: str, output_file: str
):
    with open(output_file, "w") as f:
        json.dump(
            {
                "score": result.score,
                "model": model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "client_settings": client_settings,
                "agent_ids": result.agent_ids,
                "individual_scores": result.individual_scores,
            },
            f,
        )
    print(f"[green]Result written to {output_file}[/green]")


def write_result(
    result: EvaluationResult, client_settings: dict, model: str, output_file: str
):
    write_result_to_json(result, client_settings, model, output_file)
    output_path = os.path.dirname(output_file)
    subdir = os.path.splitext(os.path.basename(output_file))[0]
    agent_dir = os.path.join(output_path, subdir)
    os.makedirs(agent_dir, exist_ok=True)
    for agent_id, (
        input_message,
        output_messages,
        all_messages,
    ) in result.history.items():
        agent_file_path = os.path.join(agent_dir, f"{agent_id}.json")
        with open(agent_file_path, "w") as f:
            json.dump(
                {
                    "input": input_message,
                    "output": output_messages,
                    "all_messages": all_messages,
                },
                f,
                indent=2,
            )


def write_usage_statistics(file_path: str, usage: UsageStatistics):
    with open(f"{file_path}.json") as f:
        data = json.load(f)
    data["run_stat"] = usage.run_stat
    with open(f"{file_path}.json", "w") as f:
        json.dump(data, f)
    for agent_id, stat in usage.agent_stat.items():
        with open(f"{file_path}/{agent_id}.json") as f:
            agent_data = json.load(f)
        agent_data["agent_stat"] = stat
        with open(f"{file_path}/{agent_id}.json", "w") as f:
            json.dump(agent_data, f, indent=2)


class Dotdict(dict):
    def __getattr__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            ) from e

    def __setattr__(self, key, value):
        super().__setitem__(key, value)

    def __delattr__(self, key):
        try:
            super().__delitem__(key)
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            ) from e


GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))


async def grade_sample(question: str, target: str, predicted_answer: str, custom_template: str = None) -> str:
    if custom_template:
        # Use custom template and format it with the provided values
        grader_prompt = custom_template.format(
            question=question, target=target, predicted_answer=predicted_answer
        )
    else:
        # Use default grading template
        grader_prompt = GRADER_TEMPLATE.format(
            question=question, target=target, predicted_answer=predicted_answer
        )
    
    prompt_messages = [dict(content=grader_prompt, role="user")]
    grading_response = await request_openai(prompt_messages)
    match = re.search(r"(A|B|C)", grading_response)
    return match.group(0) if match else "C"


async def request_openai(
    message_list: List[Dict[str, str]],
    model: str = "gpt-4.1",
    temperature: float = 0,
    max_tokens: int = 2048,
    system_message: str = "You are a helpful assistant.",
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = openai.AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

    if system_message:
        message_list = [{"role": "system", "content": system_message}] + message_list

    trial = 0
    while True:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=message_list,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except openai.BadRequestError as e:
            print("Bad Request Error", e)
            return ""
        except openai.RateLimitError as e:
            backoff = 2**trial
            print(f"Rate limit, retrying {trial} after {backoff} sec", e)
            await asyncio.sleep(backoff)
            trial += 1
        except Exception as e:
            print("Unexpected error", e)
            return ""

def collect_stat(result_dir: str, model: str, apply_penalty: bool):
    scores = []
    total_input_tokens = 0
    total_output_tokens = 0
    i = 0

    while i < 100:
        i += 1
        json_path = os.path.join(result_dir, f"{model}_{i}.json")
        if not os.path.isfile(json_path):
            break

        with open(json_path, "r") as f:
            data = json.load(f)

        total_input_tokens += data["input_tokens"]
        total_output_tokens += data["output_tokens"]

        if apply_penalty:
            individual_scores = data["individual_scores"]
            agent_ids = data["agent_ids"]
            total_score = 0

            for individual_score, agent_id in zip(individual_scores, agent_ids):
                if individual_score == 0:
                    continue

                agent_path = os.path.join(
                    result_dir, f"{model}_{i}", f"{agent_id}.json"
                )
                if not os.path.isfile(agent_path):
                    continue

                with open(agent_path, "r") as agent_f:
                    agent_messages = json.load(agent_f)

                this_score = 1

                if "core_memory_read" in result_dir:
                    for message in agent_messages["output"]:
                        if message.get("message_type") == "tool_call_message":
                            this_score = 0
                            break
                if "core_memory_write" in result_dir:
                    for message in agent_messages["output"]:
                        if message.get("message_type") == "tool_call_message":
                            if message.get("tool_call").get("name") not in ["archival_memory_search", "conversation_search"]:
                                this_score = 0
                                break
                if "core_memory_update" in result_dir:
                    for message in agent_messages["output"]:
                        if message.get("message_type") == "tool_call_message":
                            this_score = 0
                            break

                if "archival" in result_dir:
                    any_archival_search = False
                    for message in agent_messages["output"]:
                        if message.get("message_type") == "tool_call_message":
                            tool = message.get("tool_call", {}).get("name")
                            if tool not in ["archival_memory_search", "conversation_search"]:
                                this_score = 0
                            if tool == "archival_memory_search":
                                any_archival_search = True
                    if not any_archival_search:
                        this_score = 0

                total_score += this_score

            scores.append(total_score)
        else:
            scores.append(data["score"])

    mean_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0

    return (
        model,
        mean_score,
        min_score,
        max_score,
        total_input_tokens,
        total_output_tokens,
    )


def collect_stats(parent_dir: str, model: str, apply_penalty: bool):
    separator = "=" * 50
    print(f"{separator}")
    print(f"=== Results for {model} ===")
    print(f"{separator}")
    
    # Helper function to collect and format results
    def collect_and_format(apply_penalty_flag: bool, section_name: str):
        results = {}
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Get all subdirectories that start with "letta_bench_"
        subdirs = []
        for item in os.listdir(parent_dir):
            item_path = os.path.join(parent_dir, item)
            if os.path.isdir(item_path) and item.startswith("letta_bench_"):
                subdirs.append(item)
        
        # Process each subdirectory
        for subdir in subdirs:
            result_dir = os.path.join(parent_dir, subdir)
            (
                returned_model,
                mean_score,
                min_score,
                max_score,
                input_tokens,
                output_tokens,
            ) = collect_stat(result_dir, model, apply_penalty_flag)
            
            # Remove "letta_bench_" prefix and dataset size suffix from directory name
            benchmark_name = subdir.replace("letta_bench_", "")
            # Remove dataset size suffix (e.g., "_100", "_50", etc.)
            benchmark_name = re.sub(r"_\d+$", "", benchmark_name)
            results[benchmark_name] = mean_score
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        
        # Calculate average score
        if results:
            average_score = sum(results.values()) / len(results)
        else:
            average_score = 0
        
        # Print results in the requested format
        print(f"\n--- {section_name} ---")
        print(f"  average: {average_score:.2f}")
        print(f"  total_input_tokens: {total_input_tokens}")
        print(f"  total_output_tokens: {total_output_tokens}")
        
        # Print individual benchmark scores
        for benchmark_name, score in sorted(results.items()):
            print(f"  {benchmark_name}: {score:.2f}")
        
        return results, average_score, total_input_tokens, total_output_tokens
    
    # Collect results with penalty applied
    results_with_penalty, avg_with_penalty, tokens_in_with_penalty, tokens_out_with_penalty = collect_and_format(
        True, "With Penalty Applied"
    )
    
    # Collect results without penalty applied
    results_without_penalty, avg_without_penalty, tokens_in_without_penalty, tokens_out_without_penalty = collect_and_format(
        False, "Without Penalty Applied"
    )
    
    print(f"\n{separator}")
    
    # Return both sets of results
    return {
        "with_penalty": {
            "results": results_with_penalty,
            "average": avg_with_penalty,
            "input_tokens": tokens_in_with_penalty,
            "output_tokens": tokens_out_with_penalty,
        },
        "without_penalty": {
            "results": results_without_penalty,
            "average": avg_without_penalty,
            "input_tokens": tokens_in_without_penalty,
            "output_tokens": tokens_out_without_penalty,
        }
    }


if __name__ == "__main__":
    import asyncio

    async def main():
        arg = argparse.ArgumentParser()
        arg.add_argument("--port", type=int, default=8283)
        arg.add_argument("--delete_all_agents", action="store_true")
        arg.add_argument("--search_agent_name_by_id", type=str)
        arg.add_argument("--get_results_for_model", type=str)
        arg.add_argument("--get_benchmark_results_for_model", type=str)
        arg.add_argument("--result_dir", type=str)
        arg.add_argument("--benchmark_name", type=str)

        args = arg.parse_args()

        client = AsyncLetta(base_url=f"http://localhost:{args.port}")

        if args.delete_all_agents:
            agents = await client.agents.list(limit=5000)
            for agent in agents:
                await client.agents.delete(agent.id)

        if args.search_agent_name_by_id:
            agent_id = args.search_agent_name_by_id
            agent_state = await client.agents.retrieve(agent_id)
            print(agent_state.name)

        if args.get_results_for_model:
            model = args.get_results_for_model
            parent_dir = args.result_dir
            collect_stats(parent_dir, model, True)

        if args.get_benchmark_results_for_model:
            model = args.get_benchmark_results_for_model
            result_dir = args.result_dir
            (
                model,
                mean_score,
                min_score,
                max_score,
                total_input_tokens,
                total_output_tokens,
            ) = collect_stat(result_dir, model, True)
            print(
                f"{model},{round(mean_score, 2)},{total_input_tokens},{total_output_tokens}"
            )

    asyncio.run(main())
