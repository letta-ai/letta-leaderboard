from typing import Any
from letta_client import AsyncLetta, LlmConfig, EmbeddingConfig


async def create_base_agent(
    client: AsyncLetta,
    datum: Any,
    llm_config: LlmConfig,
    embedding_config: EmbeddingConfig,
) -> str:
    agent = await client.agents.create(
        llm_config=llm_config, embedding_config=embedding_config
    )
    return agent.id
