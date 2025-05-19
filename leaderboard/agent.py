from typing import Any
from letta_client import Letta, LlmConfig, EmbeddingConfig


def create_base_agent(
    client: Letta,
    datum: Any,
    llm_config: LlmConfig,
    embedding_config: EmbeddingConfig,
) -> str:
    agent_id = client.agents.create(
        llm_config=llm_config, embedding_config=embedding_config
    ).id
    return agent_id
