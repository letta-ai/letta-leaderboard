import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from dataclasses import dataclass
import datetime
import importlib
import json
import os
from typing import Callable, Generator, Union
from tqdm import tqdm
from letta_client import LettaResponse, MessageCreate
from leaderboard.agent import create_base_agent
from leaderboard.benchmark import Benchmark
from letta_client import Letta, LlmConfig, EmbeddingConfig
from rich import print
import traceback

from leaderboard.utils import (
    EvaluationResult,
    add_to_json,
    write_result,
    write_result_to_json,
)


def extract_last_message(response: LettaResponse) -> str:
    for message in response.messages[::-1]:
        if message.message_type == "assistant_message":
            return message.content
    print(f"[red]No message found in response {response}[/red]")
    return ""


def evaluate(
    benchmark: Benchmark,
    client: Letta,
    create_agent_fun: Callable[[Letta], str],
) -> EvaluationResult:
    total_score = 0
    total = len(benchmark.dataset)
    individual_scores = []

    progress_bar = tqdm(benchmark.dataset, desc=f"Score: {total_score}/{total}")
    agent_ids = []

    input_tokens = 0
    output_tokens = 0

    for datum in progress_bar:
        agent_id = create_agent_fun(client)
        benchmark.setup_agent(
            datum, client, agent_id
        )  # sets up the agent for current data
        response = client.send_message(
            agent_id=agent_id, message=datum.message, role="user"
        )
        predicted_answer = extract_last_message(response)
        current_score = benchmark.metric(predicted_answer, datum.answer)
        individual_scores.append(current_score)
        total_score += current_score
        input_tokens += response.usage.prompt_tokens
        output_tokens += response.usage.completion_tokens

        progress_bar.set_description(f"Score: {total_score}/{total}")
        progress_bar.refresh()
        agent_ids.append(agent_id)

    return EvaluationResult(
        score=total_score,
        agent_ids=agent_ids,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        individual_scores=individual_scores,
    )


def run_single_data(datum, create_agent_fun, client, benchmark):
    agent_id = create_agent_fun(client)
    benchmark.setup_agent(datum, client, agent_id)  # sets up the agent for current data
    try:
        response = client.agents.messages.create(
            agent_id=agent_id,
            messages=[
                MessageCreate(
                    role="user",
                    content=datum.message,
                )
            ],
        )
        predicted_answer = extract_last_message(response)
        score = benchmark.metric(
            predicted_answer, datum.answer, datum.message, agent_id
        )
        # print the score in red
        print("[red]Score: " + str(score) + "[/red]")
        return (
            agent_id,
            score,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
    except Exception as e:
        print(f"[red]Error in run_single_data: {e}[/red]")
        return (
            agent_id,
            0,
            0,
            0,
        )


def evaluate_multithread(
    benchmark: Benchmark,
    client: Letta,
    create_agent_fun,
    num_thread=8,
    timeout=60,
):
    total = len(benchmark.dataset)
    progress_bar = tqdm(total=total, desc="Score: 0/" + str(total))

    agent_ids = []
    individual_scores = []
    input_tokens = 0
    output_tokens = 0
    total_score = 0

    def process_datum_with_retry(datum, max_retries=5, timeout=None):
        for attempt in range(max_retries):
            try:
                return process_datum(datum)
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise e

    def process_datum(datum):
        agent_id = create_agent_fun(client, datum)
        benchmark.setup_agent(
            datum, client, agent_id
        )  # sets up the agent for current data
        response = benchmark.get_response(client, agent_id, datum)
        # FIXME(shangyin) if get response does something this could fail
        input_message = datum.message
        response_messages = [m.model_dump(mode="json") for m in response.messages]
        all_messages = [
            m.model_dump(mode="json")
            for m in client.agents.messages.list(agent_id=agent_id, limit=100)
        ]
        predicted_answer = extract_last_message(response)
        score = benchmark.metric(predicted_answer, datum.answer, datum, agent_id)
        # print the score in red
        print("[red]Score: " + str(score) + "[/red]")
        return (
            agent_id,
            score,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            input_message,
            response_messages,
            all_messages,
        )

    results = []
    NUM_RETRIES=5
    print("Starting at: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        future_to_datum = {
            executor.submit(process_datum_with_retry, datum, NUM_RETRIES): datum for datum in benchmark.dataset
        }

        total_success = 0

        history = {}

        agent_id = None

        for future in as_completed(future_to_datum):
            try:
                (
                    agent_id,
                    score,
                    input_t,
                    output_t,
                    input_message,
                    response_messages,
                    all_messages,
                ) = future.result(timeout=timeout * NUM_RETRIES)
                agent_ids.append(agent_id)
                individual_scores.append(score)
                total_score += score
                input_tokens += input_t
                output_tokens += output_t
                total_success += 1
                history[agent_id] = (input_message, response_messages, all_messages)
            except Exception as e:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[red]Error in future: {e} at " + current_time)
                print(traceback.format_exc())
                print(f"[red] agent id is {agent_id}[/red]")
                continue  # Skip failed future

            progress_bar.set_description(f"Score: {total_score}/{total}")
            progress_bar.update(1)
            progress_bar.refresh()

    print(f"Total Success: {total_success}/{total}")

    progress_bar.close()
    return EvaluationResult(
        score=total_score,
        agent_ids=agent_ids,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        individual_scores=individual_scores,
        history=history,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        help="the name of the benchmark - assuming benchmark_name/benchmark_name_benchmark.py exists",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="""Model to use. Available: [azure-gpt-4o-mini,
            bedrock-claude-3-5-sonnet,
            claude-3-5-haiku,
            claude-3-5-sonnet,
            deepseek-reasoner,
            gemini-pro,
            gemini-vertex,
            groq,
            letta-hosted,
            ollama,
            openai-gpt-3.5-turbo,
            openai-gpt-4o-mini,
            openai-gpt-4o,
            together-llama-3-1-405b,
            together-llama-3-70b,
            together-mistral-small-24B-Instruct-2501,
            xai-grok-2,]""",
        default="openai-gpt-4o-mini",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        help="Number of threads to use for evaluation",
        default=16,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file to write the result",
        default="",
    )

    parser.add_argument(
        "--dataset_size",
        type=int,
        help="Size of the dataset to evaluate",
        default=100,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for each evaluation",
        default=60,
    )

    parser.add_argument(
        "--benchmark_variable",
        type=str,
        help="The variable name of the benchmark to use",
        default="benchmark",
    )

    parser.add_argument(
        "--result_name_suffix",
        type=str,
        default="",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--repeat_from",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--letta_server",
        type=str,
        default="http://localhost:8283",
    )

    # import the benchmark
    args = parser.parse_args()

    client_settings = {
        "base_url": args.letta_server,
    }
    client = Letta(**client_settings)

    bench = importlib.import_module(
        f".{args.benchmark}.{args.benchmark}_benchmark", "leaderboard"
    )

    benchmark: Benchmark = getattr(bench, args.benchmark_variable)

    model = args.model

    model_config_path = f"leaderboard/llm_model_configs/{model}.json"

    with open(model_config_path) as f:
        model_config = json.load(f)

    llm_config = LlmConfig(**model_config)
    embedding_config = EmbeddingConfig(
        embedding_model="text-embedding-ada-002",
        embedding_endpoint_type="openai",
        embedding_endpoint="https://api.openai.com/v1",
        embedding_dim=1536,
        embedding_chunk_size=300,
    )

    if getattr(benchmark, "create_agent_fun", None):
        create_base_agent_fun = lambda client, datum: benchmark.create_agent_fun(
            client, datum, llm_config, embedding_config
        )

    else:
        create_base_agent_fun = lambda client, datum: create_base_agent(
            client, datum, llm_config, embedding_config
        )

    benchmark.truncate_dataset(args.dataset_size)

    for i in range(args.repeat_from, args.repeat):
        print(f"[green]Running evaluation {i + 1}/{args.repeat}[/green]")
        print("time out is " + str(args.timeout) + " seconds")
        result = evaluate_multithread(
            benchmark,
            client,
            create_base_agent_fun,
            num_thread=args.num_threads,
            timeout=args.timeout,
        )
        os.makedirs(
            f"results/{args.benchmark}_{args.benchmark_variable}_{args.dataset_size}",
            exist_ok=True,
        )
        outfile_path = (
            args.output_file + f"_{i+1}"
            if args.output_file
            else f"results/{args.benchmark}_{args.benchmark_variable}_{args.dataset_size}/{model}{args.result_name_suffix}_{i+1}.json"
        )
        write_result(result, client_settings, model, outfile_path)

        usage_statistics = benchmark.get_usage_statistics(
            client, result.agent_ids, evaluation_result=result
        )

        add_to_json(
            outfile_path,
            {
                "usage_statistics": usage_statistics,
            },
        )
