import argparse
import importlib
import json
import os
import datetime
import traceback
import asyncio
from typing import Callable, Any
from tqdm import tqdm
from rich import print

from letta_client import AsyncLetta, MessageCreate, LlmConfig, EmbeddingConfig
from leaderboard.agent import create_base_agent
from leaderboard.benchmark import Benchmark
from leaderboard.utils import EvaluationResult, write_result, write_usage_statistics


def extract_last_message(response: Any) -> str:
    for message in response.messages[::-1]:
        if message.message_type == "assistant_message":
            return message.content
    return ""


async def evaluate(
    benchmark: Benchmark,
    client: AsyncLetta,
    create_agent_fun: Callable[[AsyncLetta, Any], asyncio.Future],
) -> EvaluationResult:
    total_score = 0
    total = len(benchmark.dataset)
    individual_scores = []
    agent_ids = []
    input_tokens = 0
    output_tokens = 0

    progress_bar = tqdm(benchmark.dataset, desc=f"Score: {total_score}/{total}")

    for datum in progress_bar:
        agent_id = await create_agent_fun(client, datum)
        await benchmark.setup_agent(datum, client, agent_id)
        response = await client.agents.messages.create(
            agent_id=agent_id,
            messages=[MessageCreate(role="user", content=datum.message)],
        )
        predicted_answer = extract_last_message(response)
        current_score = await benchmark.metric(
            predicted_answer, datum.answer, datum, agent_id
        )
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


async def run_single_data(
    datum: Any,
    create_agent_fun: Callable[[AsyncLetta, Any], asyncio.Future],
    client: AsyncLetta,
    benchmark: Benchmark,
):
    agent_id = await create_agent_fun(client, datum)
    await benchmark.setup_agent(datum, client, agent_id)
    try:
        response = await client.agents.messages.create(
            agent_id=agent_id,
            messages=[MessageCreate(role="user", content=datum.message)],
        )
        predicted_answer = extract_last_message(response)
        score = await benchmark.metric(
            predicted_answer, datum.answer, datum.message, agent_id
        )
        return (
            agent_id,
            score,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
    except Exception as e:
        print(f"[red]Error in run_single_data: {e}[/red]")
        return (agent_id, 0, 0, 0)


async def evaluate_concurrent(
    benchmark: Benchmark,
    client: AsyncLetta,
    create_agent_fun: Callable[[AsyncLetta, Any], asyncio.Future],
    timeout: int = 60,
    max_concurrency: int = 16,
):
    total = len(benchmark.dataset)
    progress_bar = tqdm(total=total, desc=f"Score: 0/{total}")

    agent_ids = []
    individual_scores = []
    input_tokens = 0
    output_tokens = 0
    total_score = 0
    history: dict[str, tuple[str, list, list]] = {}

    MAX_RETRIES = 3

    async def process_datum(datum: Any):
        agent_id = await create_agent_fun(client, datum)
        await benchmark.setup_agent(datum, client, agent_id)
        response = await benchmark.get_response(client, agent_id, datum)
        input_message = datum.message
        response_messages = [m.model_dump(mode="json") for m in response.messages]
        all_messages = [
            m.model_dump(mode="json")
            for m in await client.agents.messages.list(agent_id=agent_id, limit=100)
        ]
        predicted_answer = extract_last_message(response)
        score = await benchmark.metric(predicted_answer, datum.answer, datum, agent_id)
        return (
            agent_id,
            score,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            input_message,
            response_messages,
            all_messages,
        )

    async def process_with_retry(datum: Any):
        for attempt in range(MAX_RETRIES):
            try:
                return await process_datum(datum)
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise

    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_with_semaphore(datum: Any):
        async with semaphore:
            return await process_with_retry(datum)

    tasks = [asyncio.create_task(process_with_semaphore(d)) for d in benchmark.dataset]

    for task in asyncio.as_completed(tasks):
        try:
            (
                agent_id,
                score,
                in_t,
                out_t,
                input_message,
                response_messages,
                all_messages,
            ) = await asyncio.wait_for(task, timeout * MAX_RETRIES)
            agent_ids.append(agent_id)
            individual_scores.append(score)
            total_score += score
            input_tokens += in_t
            output_tokens += out_t
            history[agent_id] = (input_message, response_messages, all_messages)
        except Exception as e:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[red]Error in task: {e} at {now}[/red]")
            print(traceback.format_exc())
            continue

        progress_bar.update(1)
        progress_bar.set_description(f"Score: {total_score}/{total}")
        progress_bar.refresh()

    progress_bar.close()

    return EvaluationResult(
        score=total_score,
        agent_ids=agent_ids,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        individual_scores=individual_scores,
        history=history,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Name of the benchmark module (e.g. foo/foo_benchmark.py)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-gpt-4o-mini",
        help="Model to use",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Base output file path",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=100,
        help="Number of datapoints to run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-task timeout in seconds",
    )
    parser.add_argument(
        "--benchmark_variable",
        type=str,
        default="benchmark",
        help="Variable name in the module",
    )
    parser.add_argument(
        "--result_name_suffix",
        type=str,
        default="",
        help="Suffix to append to result filenames",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to repeat",
    )
    parser.add_argument(
        "--repeat_from",
        type=int,
        default=0,
        help="Start index for repeat",
    )
    parser.add_argument(
        "--letta_server",
        type=str,
        default="http://localhost:8283",
        help="Base URL for Letta server",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=16,
        help="Maximum number of concurrent evaluation tasks",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Result output parent directory name",
    )
    args = parser.parse_args()

    client_settings = {"base_url": args.letta_server}
    client = AsyncLetta(**client_settings)

    bench_mod = importlib.import_module(
        f".{args.benchmark}.{args.benchmark}_benchmark", "leaderboard"
    )
    benchmark: Benchmark = getattr(bench_mod, args.benchmark_variable)

    model_config_path = f"leaderboard/llm_model_configs/{args.model}.json"
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

    # Create agent_config dict containing llm_config and embedding_config
    agent_config = {
        "llm_config": llm_config,
        "embedding_config": embedding_config,
        "agent_type": "memgpt_v2_agent",
    }

    # Verify agent_config contains required keys
    assert "llm_config" in agent_config, "agent_config must contain 'llm_config'"
    assert "embedding_config" in agent_config, "agent_config must contain 'embedding_config'"
    assert agent_config["llm_config"] is not None, "llm_config cannot be None"
    assert agent_config["embedding_config"] is not None, "embedding_config cannot be None"

    if getattr(benchmark, "create_agent_fun", None):

        def create_base_agent_fun(c, d):
            return benchmark.create_agent_fun(c, d, agent_config)
    else:
        def create_base_agent_fun(c, d):
            return create_base_agent(c, d, agent_config)

    benchmark.truncate_dataset(args.dataset_size)

    for i in range(args.repeat_from, args.repeat):
        print(
            f"[green]Running eval {i + 1}/{args.repeat} for {args.benchmark_variable}[/green]"
        )
        result = await evaluate_concurrent(
            benchmark,
            client,
            create_base_agent_fun,
            timeout=args.timeout,
            max_concurrency=args.max_concurrency,
        )
        out_dir = (
            f"{args.out_dir}/{args.benchmark}_{args.benchmark_variable}_{args.dataset_size}"
        )
        os.makedirs(out_dir, exist_ok=True)
        base = (
            args.output_file + f"_{i + 1}"
            if args.output_file
            else f"{out_dir}/{args.model}{args.result_name_suffix}_{i + 1}"
        )
        write_result(result, client_settings, args.model, f"{base}.json")

        usage_stats = await benchmark.get_usage_statistics(
            client, result.agent_ids, evaluation_result=result
        )
        write_usage_statistics(base, usage_stats)


if __name__ == "__main__":
    asyncio.run(main())
