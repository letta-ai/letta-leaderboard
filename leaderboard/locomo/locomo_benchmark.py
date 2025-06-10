"""
LoCoMo (Long-term Conversational Memory) Question Answering Benchmark

This benchmark evaluates LLM agents' ability to answer questions based on 
very long-term conversational memory across multiple sessions.

Based on the paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
by Maharana et al. (2024)
"""

from typing import List, Dict, Optional, Literal
import json
import asyncio
from pathlib import Path

from letta_client import (
    CreateBlock,
    AsyncLetta,
    LettaResponse,
    MessageCreate,
)
from leaderboard.benchmark import Benchmark
from leaderboard.evaluate import EvaluationResult
from leaderboard.utils import (
    Dotdict,
    grade_sample,
    UsageStatistics,
)


class LoCoMoQABenchmark(Benchmark):
    """
    LoCoMo Question Answering benchmark for evaluating very long-term conversational memory.
    
    Tests agents' ability to recall and utilize information from extensive conversation histories
    to answer questions accurately.
    """
    
    def __init__(self, data_path: Optional[str] = None, chunking_strategy: Literal["turn", "session", "time_window"] = "session"):
        """
        Initialize LoCoMo QA benchmark.
        
        Args:
            data_path: Path to LoCoMo dataset file (locomo10.json)
            chunking_strategy: How to chunk the conversation history into messages
                - "turn": Each individual turn as a separate message
                - "session": Group turns by session 
                - "time_window": Group turns by time windows
        """
        self.data_path = data_path or "leaderboard/locomo/locomo10.json"
        self.chunking_strategy = chunking_strategy
        self.raw_data = self._load_dataset()
        self.agent_datum_mapping: Dict[str, Dotdict] = {}
        self.template_agent_id: Optional[str] = None
        self.template_agent_lock = asyncio.Lock()
        self.template_message_ids: List[str] = []
        self.dataset = self._build_dataset()
        self.benchmark_type = "feature"
    
    def _load_dataset(self) -> List[Dict]:
        """Load the LoCoMo dataset from JSON file."""
        dataset_path = Path(self.data_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"LoCoMo dataset not found at {self.data_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def _build_dataset(self) -> List[Dotdict]:
        """
        Build dataset for question answering task.
        
        Returns:
            List of QA data points with conversation context
        """
        data: List[Dotdict] = []
        
        for sample in self.raw_data:
            conversation_history = self._extract_conversation_history(sample)
            
            # Process each QA pair in the sample
            for qa_item in sample.get("qa", []):
                data.append(Dotdict({
                    "sample_id": sample["sample_id"],
                    "conversation_history": conversation_history,
                    "message": qa_item["question"],
                    "answer": qa_item["answer"],
                    "category": qa_item.get("category", "unknown"),
                    "evidence": qa_item.get("evidence", []),
                    "speakers": {
                        "speaker_a": sample.get("conversation", {}).get("speaker_a", "Speaker A"),
                        "speaker_b": sample.get("conversation", {}).get("speaker_b", "Speaker B")
                    }
                }))
        
        return data
    
    def _extract_conversation_history(self, sample: Dict) -> List[Dict]:
        """Extract and format conversation history from sample."""
        conversation = sample.get("conversation", {})
        history = []
        
        # Get all sessions in chronological order
        session_keys = sorted([k for k in conversation.keys() if k.startswith("session_")])
        
        for session_key in session_keys:
            session_data = conversation[session_key]
            if isinstance(session_data, list):
                for turn in session_data:
                    history.append({
                        "session": session_key,
                        "speaker": turn.get("speaker", ""),
                        "text": turn.get("text", ""),
                        "dia_id": turn.get("dia_id", ""),
                        "img_url": turn.get("img_url", ""),
                        "blip_caption": turn.get("blip_caption", ""),
                        "timestamp": turn.get("timestamp", ""),
                    })
        
        return history
    
    def _chunk_conversation_by_turn(self, conversation_history: List[Dict]) -> List[str]:
        """Chunk conversation with each turn as a separate message."""
        chunks = []
        for turn in conversation_history:
            content = f"{turn['speaker']}: {turn['text']}"
            
            # Add timestamp if available
            if turn.get('timestamp'):
                content = f"Timestamp: {turn['timestamp']} {content}"
            
            # Add image context if available
            if turn.get('img_url') and turn.get('blip_caption'):
                content += f" [Image: {turn['blip_caption']}]"
            
            chunks.append(content)
        
        return chunks
    
    def _chunk_conversation_by_session(self, conversation_history: List[Dict]) -> List[str]:
        """Chunk conversation by grouping turns within the same session."""
        chunks = []
        current_session = None
        current_chunk_turns = []
        
        for turn in conversation_history:
            if turn['session'] != current_session:
                # Save previous session chunk if exists
                if current_chunk_turns:
                    session_content = self._format_session_chunk(current_chunk_turns, current_session)
                    chunks.append(session_content)
                
                # Start new session
                current_session = turn['session']
                current_chunk_turns = [turn]
            else:
                current_chunk_turns.append(turn)
        
        # Add final session chunk
        if current_chunk_turns:
            session_content = self._format_session_chunk(current_chunk_turns, current_session)
            chunks.append(session_content)
        
        return chunks
    
    def _format_session_chunk(self, turns: List[Dict], session: str) -> str:
        """Format a group of turns from the same session into a single chunk."""
        turn_texts = []
        session_timestamp = None
        
        for turn in turns:
            speaker_text = f"{turn['speaker']}: {turn['text']}"
            
            # Add image context if available
            if turn.get('img_url') and turn.get('blip_caption'):
                speaker_text += f" [Image: {turn['blip_caption']}]"
            
            turn_texts.append(speaker_text)
            
            # Use first timestamp as session timestamp
            if not session_timestamp and turn.get('timestamp'):
                session_timestamp = turn['timestamp']
        
        content = ", ".join(turn_texts)
        
        # Add timestamp at the beginning if available
        if session_timestamp:
            content = f"Timestamp: {session_timestamp} {content}"
        
        return content
    
    def _chunk_conversation_by_time_window(self, conversation_history: List[Dict], window_size: int = 10) -> List[str]:
        """Chunk conversation by grouping turns in time windows."""
        chunks = []
        
        for i in range(0, len(conversation_history), window_size):
            window_turns = conversation_history[i:i + window_size]
            turn_texts = []
            window_timestamp = None
            
            for turn in window_turns:
                speaker_text = f"{turn['speaker']}: {turn['text']}"
                
                # Add image context if available
                if turn.get('img_url') and turn.get('blip_caption'):
                    speaker_text += f" [Image: {turn['blip_caption']}]"
                
                turn_texts.append(speaker_text)
                
                # Use first timestamp as window timestamp
                if not window_timestamp and turn.get('timestamp'):
                    window_timestamp = turn['timestamp']
            
            content = " ".join(turn_texts)
            
            # Add timestamp at the beginning if available
            if window_timestamp:
                content = f"Timestamp: {window_timestamp} {content}"
            
            chunks.append(content)
        
        return chunks
    
    def _create_conversation_messages(self, datum: Dotdict) -> List[MessageCreate]:
        """Create conversation messages based on chunking strategy."""
        if self.chunking_strategy == "turn":
            chunks = self._chunk_conversation_by_turn(datum.conversation_history)
        elif self.chunking_strategy == "session":
            chunks = self._chunk_conversation_by_session(datum.conversation_history)
        elif self.chunking_strategy == "time_window":
            chunks = self._chunk_conversation_by_time_window(datum.conversation_history)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")
        
        # Convert chunks to user messages
        messages = []
        for chunk in chunks:
            messages.append(MessageCreate(role="user", content=chunk))
        
        # Add evaluation prompt
        eval_prompt = (
            "Based on the conversation history provided above, please answer the following question. "
            "Use only the information from the conversation to provide your answer."
        )
        messages.append(MessageCreate(role="user", content=eval_prompt))
        
        return messages

    async def setup_agent(self, datum: Dotdict, client: AsyncLetta, agent_id: str) -> None:
        """
        Setup agent - no additional setup needed since conversation history 
        is already loaded during agent creation or copied from template.
        """
        pass

    async def create_agent_fun(
        self,
        client: AsyncLetta,
        datum: Dotdict,
        llm_config,
        embedding_config,
    ) -> str:
        """
        Create agent configured for LoCoMo Question Answering.
        
        If this is the first agent, create template agent and send all conversation messages.
        For subsequent agents, create fresh agents and copy template agent's message history.
        """
        
        async with self.template_agent_lock:
            if self.template_agent_id is None:
                # Create template agent
                memory_blocks = [
                    CreateBlock(
                        label="Task Instructions",
                        value="You are tasked with answering questions about conversation history. "
                              "The conversation history will be provided as a series of messages. "
                              "Read through all the conversation context carefully before answering any questions."
                    ),
                    CreateBlock(
                        label="Conversation Context",
                        value=f"This conversation involves {datum.speakers['speaker_a']} and {datum.speakers['speaker_b']}. "
                              f"The conversation history contains {len(datum.conversation_history)} turns across multiple sessions."
                    )
                ]
                
                template_agent = await client.agents.create(
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                    memory_blocks=memory_blocks,
                )
                
                # Send all conversation chunks to template agent
                conversation_messages = self._create_conversation_messages(datum)
                
                # Send messages and get response to populate message history
                response = await client.agents.messages.create(
                    agent_id=template_agent.id,
                    messages=conversation_messages,
                )
                
                # Get the message history from template agent to copy to others
                message_history = await client.agents.messages.list(template_agent.id)
                self.template_message_ids = [msg.id for msg in message_history if hasattr(msg, 'id')]
                
                self.template_agent_id = template_agent.id
                self.agent_datum_mapping[template_agent.id] = datum
                return template_agent.id
        
        # Template agent exists, create new agent and copy message history
        memory_blocks = [
            CreateBlock(
                label="Task Instructions",
                value="You are tasked with answering questions about conversation history. "
                      "The conversation history will be provided as a series of messages. "
                      "Read through all the conversation context carefully before answering any questions."
            ),
            CreateBlock(
                label="Conversation Context",
                value=f"This conversation involves {datum.speakers['speaker_a']} and {datum.speakers['speaker_b']}. "
                      f"The conversation history contains {len(datum.conversation_history)} turns across multiple sessions."
            )
        ]
        
        # Create fresh agent
        agent = await client.agents.create(
            llm_config=llm_config,
            embedding_config=embedding_config,
            memory_blocks=memory_blocks,
        )
        
        # Copy message history from template agent using modify()
        if self.template_message_ids:
            try:
                await client.agents.modify(
                    agent_id=agent.id,
                    message_ids=self.template_message_ids,
                )
            except Exception as e:
                # If message_ids copying fails, fallback to sending messages directly
                print(f"Warning: Failed to copy message history via modify(), falling back to direct messaging: {e}")
                conversation_messages = self._create_conversation_messages(datum)
                await client.agents.messages.create(
                    agent_id=agent.id,
                    messages=conversation_messages,
                )
        
        self.agent_datum_mapping[agent.id] = datum
        return agent.id
    
    async def metric(
        self, predicted: str, true: str, datum: Dotdict, agent_id: str
    ) -> float:
        """
        Evaluate QA prediction against ground truth answer.
        
        Uses the existing grading system to determine if the predicted answer
        matches the expected answer.
        """
        result = await grade_sample(datum.message, true, predicted)
        return 1.0 if result == "A" else 0.0
    
    async def get_response(
        self,
        client: AsyncLetta,
        agent_id: str,
        datum: Dotdict,
    ) -> LettaResponse:
        """Get response from agent for the QA question."""
        # The conversation history is already loaded in the agent
        # Just send the actual question
        return await client.agents.messages.create(
            agent_id=agent_id,
            messages=[MessageCreate(
                role="user",
                content=datum.message,
            )],
        )
    
    async def get_usage_statistics(
        self, client: AsyncLetta, agent_ids: List[str], evaluation_result: EvaluationResult
    ) -> UsageStatistics:
        """Get usage statistics for the QA evaluation."""
        return UsageStatistics(
            {},
            {
                agent_id: {
                    "task_type": "question_answering",
                    "chunking_strategy": self.chunking_strategy,
                    "num_conversation_turns": len(self.agent_datum_mapping.get(agent_id, {}).get("conversation_history", [])),
                    "sample_id": self.agent_datum_mapping.get(agent_id, {}).get("sample_id", ""),
                    "question_category": self.agent_datum_mapping.get(agent_id, {}).get("category", "unknown")
                }
                for agent_id in agent_ids
                if agent_id in self.agent_datum_mapping
            },
        )


# Benchmark instances for different chunking strategies
locomo_qa_benchmark = LoCoMoQABenchmark(chunking_strategy="session")
locomo_qa_turn_benchmark = LoCoMoQABenchmark(chunking_strategy="turn")
locomo_qa_time_benchmark = LoCoMoQABenchmark(chunking_strategy="time_window")
