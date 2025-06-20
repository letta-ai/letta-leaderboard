"""
LoCoMo (Long-term Conversational Memory) Question Answering Benchmark

This benchmark evaluates LLM agents' ability to answer questions based on 
very long-term conversational memory across multiple sessions.

Based on the paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
by Maharana et al. (2024)
"""

import time
import os
import tempfile
import re
from typing import List, Dict, Optional, Literal
import json
import asyncio
from pathlib import Path

from letta_client import (
    CreateBlock,
    AsyncLetta,
    EmbeddingConfig,
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
        self.template_agent_ids: Dict[str, str] = {}  # sample_id -> agent_id mapping
        self.template_agent_locks: Dict[str, asyncio.Lock] = {}  # sample_id -> lock mapping
        self.template_message_ids: Dict[str, List[str]] = {}  # sample_id -> message_ids mapping
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
                # Handle different answer formats based on category
                category = qa_item.get("category", "unknown")
                if category == 5:
                    # Adversarial questions use "adversarial_answer"
                    continue
                    # answer = qa_item.get("adversarial_answer", "")
                else:
                    # Regular questions use "answer"
                    answer = qa_item.get("answer", "")
                
                data.append(Dotdict({
                    "sample_id": sample["sample_id"],
                    "conversation_history": conversation_history,
                    "message": qa_item["question"],
                    "answer": answer,
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
            # Get session timestamp
            session_timestamp_key = f"{session_key}_date_time"
            session_timestamp = conversation.get(session_timestamp_key, "")
            
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
                        "timestamp": session_timestamp,  # Use session-level timestamp
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
        
        content = "\n".join(turn_texts)
        
        # Add timestamp at the beginning if available
        if session_timestamp:
            content = f"Timestamp: {session_timestamp}\n{content}"
        
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
        
        # TODO: Consider adding an evaluation prompt to help guide the agent's responses
        # eval_prompt = (
        #     "Based on the conversation history provided above, please answer the following question. "
        #     "Use only the information from the conversation to provide your answer."
        # )
        
        return messages

    async def _find_send_message_tool_id(self, client: AsyncLetta, agent_id: str) -> Optional[str]:
        """Find the send_message tool ID for an agent."""
        tools = await client.agents.tools.list(agent_id)
        for tool in tools:
            if tool.name == "send_message":
                return tool.id
        return None

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
        agent_config: dict,
    ) -> str:
        """
        Create agent configured for LoCoMo Question Answering.
        
        If this is the first agent for this sample_id, create template agent and send all conversation messages.
        For subsequent agents with the same sample_id, create fresh agents and copy template agent's message history.
        """
        # Ensure agent_config contains required keys
        assert "llm_config" in agent_config, "agent_config must contain 'llm_config'"
        assert "embedding_config" in agent_config, "agent_config must contain 'embedding_config'"
        
        print(f"Creating agent for sample {datum.sample_id}")
        
        # Initialize lock for this sample_id if not exists
        if datum.sample_id not in self.template_agent_locks:
            self.template_agent_locks[datum.sample_id] = asyncio.Lock()
        
        async with self.template_agent_locks[datum.sample_id]:
            if datum.sample_id not in self.template_agent_ids:
                print(f"Creating template agent (first agent) for sample {datum.sample_id}")
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
                    memory_blocks=memory_blocks,
                    **agent_config,
                )

                print(f"Created template agent: {template_agent.id}")

                # Detach send_message tool from template agent to prevent it from responding
                send_message_tool_id = await self._find_send_message_tool_id(client, template_agent.id)
                if send_message_tool_id:
                    await client.agents.tools.detach(template_agent.id, send_message_tool_id)

                # Send all conversation chunks to template agent one by one
                conversation_messages = self._create_conversation_messages(datum)
                print(f"Sending {len(conversation_messages)} conversation messages to template agent")
                
                # Send messages one by one
                for i, message in enumerate(conversation_messages):
                    print(f"Sending message {i+1}/{len(conversation_messages)} to template agent")
                    await client.agents.messages.create(
                        agent_id=template_agent.id,
                        messages=[message],
                    )
                    print(f"Sent message {i+1}/{len(conversation_messages)} to template agent")
                
                print("All conversation messages sent to template agent")
                
                # Get the message history from template agent to copy to others
                message_history = await client.agents.messages.list(template_agent.id, limit=1000)
                
                # Debug print all messages
                print(f"\n=== DEBUG: Message History for sample {datum.sample_id} ===")
                print(f"Total messages retrieved: {len(message_history)}")
                
                system_messages = []
                valid_messages = []
                
                for i, msg in enumerate(message_history):
                    has_id = hasattr(msg, 'id')
                    msg_type = getattr(msg, 'message_type', None)
                    msg_id = getattr(msg, 'id', 'NO_ID')
                    
                    print(f"Message {i+1}: ID={msg_id}, has_id={has_id}, message_type={msg_type}")
                    
                    # do we need to filter out system messages and reasoning messages?
                    if has_id and msg_type != 'system_message':
                        valid_messages.append(msg.id)
                    else:
                        system_messages.append({
                            'index': i+1,
                            'id': msg_id,
                            'has_id': has_id,
                            'message_type': msg_type
                        })
                
                print(f"\nFiltered out {len(system_messages)} messages:")
                for sys_msg in system_messages:
                    print(f"  - Message {sys_msg['index']}: ID={sys_msg['id']}, has_id={sys_msg['has_id']}, type={sys_msg['message_type']}")
                
                print(f"\nKeeping {len(valid_messages)} valid message IDs:")
                for j, msg_id in enumerate(valid_messages):
                    print(f"  - Valid message {j+1}: {msg_id}")
                
                print("=== END DEBUG ===\n")
                
                self.template_message_ids[datum.sample_id] = valid_messages
                print(f"Retrieved {len(self.template_message_ids[datum.sample_id])} message IDs from template agent")
                
                self.template_agent_ids[datum.sample_id] = template_agent.id
                self.agent_datum_mapping[template_agent.id] = datum
                return template_agent.id
        
        # Template agent exists, create new agent and copy message history
        print(f"Creating new agent and copying from template agent {self.template_agent_ids[datum.sample_id]}")
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
            memory_blocks=memory_blocks,
            **agent_config,
        )
        print(f"Created new agent: {agent.id}")
        
        # Detach send_message tool from regular agent initially
        send_message_tool_id = await self._find_send_message_tool_id(client, agent.id)
        if send_message_tool_id:
            await client.agents.tools.detach(agent.id, send_message_tool_id)
        
        # Copy message history from template agent using modify()
        if self.template_message_ids[datum.sample_id]:
            try:
                print(f"Copying {len(self.template_message_ids[datum.sample_id])} messages from template agent")
                await client.agents.modify(
                    agent_id=agent.id,
                    message_ids=self.template_message_ids[datum.sample_id],
                )
                print("Successfully copied message history via modify()")
            except Exception as e:
                # If message_ids copying fails, fallback to sending messages directly
                print(f"Warning: Failed to copy message history via modify(), falling back to direct messaging: {e}")
                conversation_messages = self._create_conversation_messages(datum)
                # Send messages one by one
                for i, message in enumerate(conversation_messages):
                    await client.agents.messages.create(
                        agent_id=agent.id,
                        messages=[message],
                    )
                print("Fallback message sending complete")
        
        # Reattach send_message tool to regular agent for evaluation
        if send_message_tool_id:
            await client.agents.tools.attach(agent.id, send_message_tool_id)
        
        self.agent_datum_mapping[agent.id] = datum
        return agent.id
    
    async def metric(
        self, predicted: str, true: str, datum: Dotdict, agent_id: str
    ) -> float:
        """
        Evaluate QA prediction against ground truth answer.
        
        For adversarial questions (category 5), evaluates if the agent correctly
        identifies the question as unanswerable. For other categories, uses
        standard grading system.
        """
        if datum.category == 5:
            # Adversarial questions - check if agent refuses to answer or indicates unanswerable
            return await self._evaluate_adversarial_response(predicted, datum.message)
        else:
            # Regular questions - use standard grading
            result = await grade_sample(datum.message, true, predicted)
            return 1.0 if result == "A" else 0.0
    
    async def _evaluate_adversarial_response(self, predicted: str, question: str) -> float:
        """
        Evaluate if the agent correctly handles an adversarial question.
        
        Returns 1.0 if the agent appropriately refuses to answer or indicates 
        the question is unanswerable, 0.0 otherwise.
        """
        adversarial_template = """You are evaluating whether an AI agent correctly identified an adversarial/unanswerable question.

Question: {question}
Agent Response: {predicted_answer}

Adversarial questions are designed to trick the agent into providing wrong answers based on information not present in the conversation history. The agent should correctly identify these as unanswerable.

Rate the response:
- Give "A" if the agent correctly refuses to answer, says the information is not available, indicates uncertainty, or explicitly states the question cannot be answered based on the given information.
- Give "B" if the agent attempts to answer the question with specific information that wasn't provided in the conversation.

Focus on whether the agent appropriately handled the adversarial nature of the question rather than the specific content of any answer.

Just return the letter "A" or "B", with no text around it."""

        # Use grade_sample with custom template - it will format with actual values
        result = await grade_sample(question, "", predicted, custom_template=adversarial_template)
        
        # For adversarial questions, we only expect A or B responses
        # A = correctly identified as unanswerable, B = incorrectly answered
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
                    "question_category": self.agent_datum_mapping.get(agent_id, {}).get("category", "unknown"),
                    "correct_answer": self.agent_datum_mapping.get(agent_id, {}).get("answer", "")
                }
                for agent_id in agent_ids
                if agent_id in self.agent_datum_mapping
            },
        )


class LoCoMoQAFileBenchmark(LoCoMoQABenchmark):
    """
    LoCoMo Question Answering benchmark for evaluating very long-term conversational memory.
    
    Similar to LoCoMoQABenchmark, but uses a file-based memory solution.
    """

    def __init__(self, data_path: Optional[str] = None, chunking_strategy: Literal["turn", "session", "time_window"] = "session"):
        super().__init__(data_path, chunking_strategy)
        self.data_source_ids = {}
        self.sample_file_paths = {}  # sample_id -> list of file paths
        
        # Create files for all samples during initialization
        self._create_files_for_all_samples()

    def _create_files_for_all_samples(self):
        """Create conversation files for all samples during initialization."""
        import os
        from pathlib import Path
        
        # Create data directory if it doesn't exist
        data_dir = Path("leaderboard/locomo/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Group dataset by sample_id to avoid duplicates
        samples_by_id = {}
        for datum in self.dataset:
            if datum.sample_id not in samples_by_id:
                samples_by_id[datum.sample_id] = datum
        
        # Create files for each unique sample
        for sample_id, datum in samples_by_id.items():
            print(f"Creating files for sample {sample_id}")
            
            # Create sample-specific directory
            sample_dir = data_dir / f"sample_{sample_id}"
            sample_dir.mkdir(exist_ok=True)
            
            # Prepare conversation chunks with timestamps
            conversation_chunks = self._create_conversation_string_with_timestamp(datum.conversation_history)
            
            # Create files for each chunk
            file_paths = []
            for i, (content, timestamp) in enumerate(conversation_chunks):
                # Create reader-friendly filename
                if timestamp:
                    # Just use the timestamp as filename, replacing problematic characters
                    filename = timestamp.replace("/", "-").replace(":", "-") + ".txt"
                else:
                    filename = f"session_{i+1:03d}.txt"
                
                file_path = sample_dir / filename
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                file_paths.append(str(file_path))
                print(f"Created file {file_path} for timestamp {timestamp}")
            
            # Store file paths for this sample
            self.sample_file_paths[sample_id] = file_paths
            print(f"Created {len(file_paths)} files for sample {sample_id}")

    def _create_conversation_string_with_timestamp(self, conversation_history: List[Dict]) -> list[tuple[str, str]]:
        """Create a list of (content, timestamp) tuples representing the conversation history."""
        if self.chunking_strategy == "turn":
            return self._chunk_by_turn_with_timestamp(conversation_history)
        elif self.chunking_strategy == "session":
            return self._chunk_by_session_with_timestamp(conversation_history)
        elif self.chunking_strategy == "time_window":
            return self._chunk_by_time_window_with_timestamp(conversation_history)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")

    def _chunk_by_turn_with_timestamp(self, conversation_history: List[Dict]) -> list[tuple[str, str]]:
        """Chunk conversation with each turn as a separate message, returning (content, timestamp)."""
        chunks = []
        for turn in conversation_history:
            content = f"{turn['speaker']}: {turn['text']}"
            if turn.get('img_url') and turn.get('blip_caption'):
                content += f" [Image: {turn['blip_caption']}]"
            
            timestamp = turn.get('timestamp', "")
            chunks.append((content, timestamp))
        return chunks

    def _format_session_chunk_with_timestamp(self, turns: List[Dict]) -> tuple[str, str]:
        """Format turns from a session into a content string and a timestamp."""
        turn_texts = []
        session_timestamp = ""
        
        for turn in turns:
            speaker_text = f"{turn['speaker']}: {turn['text']}"
            if turn.get('img_url') and turn.get('blip_caption'):
                speaker_text += f" [Image: {turn['blip_caption']}]"
            turn_texts.append(speaker_text)
            
            if not session_timestamp and turn.get('timestamp'):
                session_timestamp = turn['timestamp']
        
        content = "\n".join(turn_texts)
        return content, session_timestamp

    def _chunk_by_session_with_timestamp(self, conversation_history: List[Dict]) -> list[tuple[str, str]]:
        """Chunk conversation by session, returning (content, timestamp)."""
        chunks = []
        current_session = None
        current_chunk_turns = []
        
        for turn in conversation_history:
            if turn['session'] != current_session:
                if current_chunk_turns:
                    session_content, session_timestamp = self._format_session_chunk_with_timestamp(current_chunk_turns)
                    chunks.append((session_content, session_timestamp))
                
                current_session = turn['session']
                current_chunk_turns = [turn]
            else:
                current_chunk_turns.append(turn)
        
        if current_chunk_turns:
            session_content, session_timestamp = self._format_session_chunk_with_timestamp(current_chunk_turns)
            chunks.append((session_content, session_timestamp))
            
        return chunks

    def _chunk_by_time_window_with_timestamp(self, conversation_history: List[Dict], window_size: int = 10) -> list[tuple[str, str]]:
        """Chunk conversation by time window, returning (content, timestamp)."""
        chunks = []
        for i in range(0, len(conversation_history), window_size):
            window_turns = conversation_history[i:i + window_size]
            turn_texts = []
            window_timestamp = ""
            
            for turn in window_turns:
                speaker_text = f"{turn['speaker']}: {turn['text']}"
                if turn.get('img_url') and turn.get('blip_caption'):
                    speaker_text += f" [Image: {turn['blip_caption']}]"
                turn_texts.append(speaker_text)
                
                if not window_timestamp and turn.get('timestamp'):
                    window_timestamp = turn['timestamp']
            
            content = "\n".join(turn_texts)
            chunks.append((content, window_timestamp))
        return chunks

    async def create_agent_fun(
        self,
        client: AsyncLetta,
        datum: Dotdict,
        agent_config: dict,
    ) -> str:
        # Ensure agent_config contains required keys
        assert "llm_config" in agent_config, "agent_config must contain 'llm_config'"
        assert "embedding_config" in agent_config, "agent_config must contain 'embedding_config'"
        # Initialize lock for this sample_id if not exists
        if datum.sample_id not in self.template_agent_locks:
            self.template_agent_locks[datum.sample_id] = asyncio.Lock()
        
        embedding_config = EmbeddingConfig(
            embedding_model="text-embedding-3-large",
            embedding_endpoint_type="openai",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_dim=1536,
            embedding_chunk_size=100000,
        )
        
        async with self.template_agent_locks[datum.sample_id]:
            if datum.sample_id not in self.data_source_ids:
                # First time for this sample_id: create a source and upload files
                print(f"Creating new data source for sample {datum.sample_id}")
                source = await client.sources.create(
                    name=f"Conversation Context for {datum.sample_id}",
                    embedding_config=embedding_config,
                    description=f"The conversation history, containing timestamped messages from {datum.speakers['speaker_a']} and {datum.speakers['speaker_b']}."
                )

                # Upload the pre-created files
                file_paths = self.sample_file_paths[datum.sample_id]
                
                for file_path in file_paths:
                    print(f"Uploading file {file_path}")

                    with open(file_path, "rb") as f:
                        job = await client.sources.files.upload(
                            source_id=source.id,
                            file=f,
                        )

                        while job.status != "completed":
                            print(f"Waiting for job {job.id} to complete... Current status: {job.status}")
                            time.sleep(1)
                            job = await client.jobs.retrieve(job_id=job.id)
                
                self.data_source_ids[datum.sample_id] = source.id
            
        source_id = self.data_source_ids[datum.sample_id]
        print(f"Using data source {source_id} for sample {datum.sample_id}")

        # create a new agent for this sample_id, with file capabilities
        agent_config.pop("agent_type", None)
        # no search_files tool now
        memory_blocks = [
            CreateBlock(
                label="Task Instructions",
                value="You are tasked with answering questions about conversation history."
                        "The conversation history will be provided as a series of files. "
                        "The files will only contain the conversation history, one person per line."
                        "Read through all the conversation context carefully before answering any questions."
                        "You can use the open_file, close_file, grep, search_files, and send_message tools to answer the question."
                        "For grep, try multiple times if you did not find the answer in the file."
                        "Please provide precise answers to the questions, do not hallucinate, and note the time! The timestamp is in the file name."
                        "Always refer to a file to answer the question, do not make up information."
                        "Avoid general searches, like just searching for a person's name."
            )
        ]

        with open("leaderboard/locomo/locomo_agent.txt", "r") as f:
            system_prompt = f.read()

        agent = await client.agents.create(
            agent_type="memgpt_v2_agent",
            include_base_tools=False,
            tools=["grep", "search_files", "send_message"],
            # "open_file", "close_file",
            # system=system_prompt,
            # memory_blocks=memory_blocks,
            **agent_config,
        )

        print(f"Created agent {agent.id} for sample {datum.sample_id}")

        # attach the source to the agent
        await client.agents.sources.attach(agent.id, source_id)

        self.agent_datum_mapping[agent.id] = datum
        return agent.id



# Benchmark instances for different chunking strategies
locomo_qa_benchmark = LoCoMoQABenchmark(chunking_strategy="session")
locomo_qa_benchmark_file = LoCoMoQAFileBenchmark(chunking_strategy="session")
# locomo_qa_turn_benchmark = LoCoMoQABenchmark(chunking_strategy="turn")
# locomo_qa_time_benchmark = LoCoMoQABenchmark(chunking_strategy="time_window")
