from typing import Any
from letta_client import AsyncLetta, LlmConfig, EmbeddingConfig


async def create_base_agent(
    client: AsyncLetta,
    datum: Any,
    agent_config: dict,
) -> str:
    # Ensure agent_config contains required keys
    assert "llm_config" in agent_config, "agent_config must contain 'llm_config'"
    assert "embedding_config" in agent_config, "agent_config must contain 'embedding_config'"
    
    agent = await client.agents.create(**agent_config)
    return agent.id
