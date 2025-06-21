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
from openai import AsyncOpenAI, OpenAI
import concurrent.futures

from letta_client import (
    ContinueToolRule,
    CreateBlock,
    AsyncLetta,
    EmbeddingConfig,
    InitToolRule,
    LettaResponse,
    MessageCreate,
    TerminalToolRule,
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

def answer_question(answer: str) -> str:
    """
    Returns the provided answer to the user.
    Your answer should be as precise as possible. For example, "this month" is not precise, "June 2025" is precise.
    
    Args:
        answer (str): The answer to return.
        
    Returns:
        str: The provided answer.
    """
    return answer


class LoCoMoQAFileBenchmark(LoCoMoQABenchmark):
    """
    LoCoMo Question Answering benchmark for evaluating very long-term conversational memory.
    
    Similar to LoCoMoQABenchmark, but uses a file-based memory solution.
    """

    def __init__(self, data_path: Optional[str] = None, chunking_strategy: Literal["turn", "session", "time_window", "secom"] = "session", use_summary: bool = False):
        super().__init__(data_path, chunking_strategy)
        self.data_source_ids = {}
        self.sample_file_paths = {}  # sample_id -> list of file paths
        
        # Create files for all samples during initialization
        if chunking_strategy == "session":
            self._create_files_for_all_samples(use_summary=use_summary)
        elif chunking_strategy == "secom":
            self._create_secom_files_for_all_samples()
        self.tool_functions.append(answer_question)

    def _create_secom_files_for_all_samples(self):
        """Create SeCom-segmented conversation files for all samples during initialization."""
        import os
        from pathlib import Path
        import concurrent.futures
        
        # Create data directory if it doesn't exist
        data_dir = Path("leaderboard/locomo/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Group dataset by sample_id to avoid duplicates
        samples_by_id = {}
        for datum in self.dataset:
            if datum.sample_id not in samples_by_id:
                samples_by_id[datum.sample_id] = datum
        
        # Collect all file creation tasks
        file_creation_tasks = []
        
        # Prepare tasks for all samples
        for sample_id, datum in samples_by_id.items():
            print(f"Preparing SeCom segmented files for sample {sample_id}")
            
            # Initialize sample_file_paths for this sample
            self.sample_file_paths[sample_id] = []
            
            # Create sample-specific directory
            sample_dir = data_dir / f"sample_{sample_id}"
            sample_dir.mkdir(exist_ok=True)
            
            # Check if summary file exists
            summary_file = sample_dir / f"{sample_id}_segment_summary.txt"
            if summary_file.exists():
                print(f"Found existing segment summary for sample {sample_id}, skipping segmentation")
                # Read filenames from summary file
                with open(summary_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        filename = line.strip()
                        if filename:
                            file_path = sample_dir / filename
                            self.sample_file_paths[sample_id].append(str(file_path))
                continue
            
            # Get conversation segments using SeCom approach
            conversation_segments = self._segment_conversation_secom(datum.conversation_history)
            
            # Create task for each segment and collect filenames
            segment_filenames = []
            for i, (content, timestamp) in enumerate(conversation_segments):
                # Determine filename based on segment index
                filename = f"segment_{i+1:03d}.txt"
                if timestamp:
                    filename = f"{timestamp.replace('/', '-').replace(':', '-')}_{i+1:03d}.txt"
                
                file_path = sample_dir / filename
                segment_filenames.append(filename)
                
                # Add to sample_file_paths immediately
                self.sample_file_paths[sample_id].append(str(file_path))
                
                # Only create task if file doesn't already exist
                if not file_path.exists():
                    file_creation_tasks.append({
                        'sample_id': sample_id,
                        'content': content,
                        'timestamp': timestamp,
                        'file_path': file_path,
                        'segment_index': i
                    })
                else:
                    print(f"File {file_path} already exists, skipping task creation")
            
            # Create summary file with segment filenames
            with open(summary_file, 'w', encoding='utf-8') as f:
                for filename in segment_filenames:
                    f.write(f"{filename}\n")
            print(f"Created segment summary file for sample {sample_id}: {len(segment_filenames)} segments")
        
        # Sort file paths for consistent ordering
        for sample_id in self.sample_file_paths:
            self.sample_file_paths[sample_id] = sorted(self.sample_file_paths[sample_id])
            print(f"Sample {sample_id}: {len(self.sample_file_paths[sample_id])} SeCom segments prepared")

        # Define the complete file creation function
        def create_single_file(task_data):
            """Create a single file with segment content."""
            sample_id = task_data['sample_id']
            content = task_data['content']
            timestamp = task_data['timestamp']
            file_path = task_data['file_path']
            segment_index = task_data['segment_index']
            
            try:                
                # Write file with timestamp and content
                with open(file_path, "w", encoding="utf-8") as f:
                    # Line 1: Timestamp or segment info
                    if timestamp:
                        f.write(f"Timestamp: {timestamp}\n")
                    else:
                        f.write(f"Segment {segment_index+1}\n")
                    
                    # Write the segment content
                    f.write(content)
                
                print(f"Created SeCom segment file {file_path} for sample {sample_id}")
                return True  # Successfully created
                
            except Exception as e:
                print(f"Error creating SeCom segment file for sample {sample_id}: {e}")
                return False  # Failed
        
        # Process all file creation tasks in parallel
        print(f"Processing {len(file_creation_tasks)} SeCom file creation tasks in parallel...")
        
        # Execute file creation tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Submit all tasks
            futures = [executor.submit(create_single_file, task) for task in file_creation_tasks]
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Just ensure completion

    def _segment_conversation_secom(self, conversation_history: List[Dict]) -> list[tuple[str, str]]:
        """
        Segment conversation using SeCom approach - topically coherent segments within sessions.
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            List of (content, timestamp) tuples for each segment
        """
        if not conversation_history:
            return []
        
        # First group by sessions (like the existing session-based approach)
        sessions = self._group_conversation_by_sessions(conversation_history)
        
        # Apply SeCom segmentation within each session
        all_segments = []
        for session_turns in sessions:
            session_segments = self._segment_single_session_secom(session_turns)
            all_segments.extend(session_segments)
        
        return all_segments
    
    def _group_conversation_by_sessions(self, conversation_history: List[Dict]) -> List[List[Dict]]:
        """Group conversation turns by session."""
        sessions = []
        current_session = None
        current_session_turns = []
        
        for turn in conversation_history:
            if turn['session'] != current_session:
                # Save previous session if exists
                if current_session_turns:
                    sessions.append(current_session_turns)
                
                # Start new session
                current_session = turn['session']
                current_session_turns = [turn]
            else:
                current_session_turns.append(turn)
        
        # Add final session
        if current_session_turns:
            sessions.append(current_session_turns)
        
        return sessions
    
    def _segment_single_session_secom(self, session_turns: List[Dict]) -> list[tuple[str, str]]:
        """
        Apply SeCom segmentation to a single session.
        
        Args:
            session_turns: List of turns from a single session
            
        Returns:
            List of (content, timestamp) tuples for segments within this session
        """
        if not session_turns:
            return []
        
        # If session is too short, treat as single segment
        if len(session_turns) <= 3:
            segment_content = self._format_segment_content(session_turns)
            segment_timestamp = session_turns[0].get('timestamp', '')
            return [(segment_content, segment_timestamp)]
        
        # Convert session to text format for segmentation
        session_text = self._format_session_for_segmentation(session_turns)
        
        # Get segment boundaries using GPT-4 for this session
        segment_boundaries = self._get_session_segments_sync(session_text, session_turns)
        
        # Create segments based on boundaries
        segments = []
        for start_idx, end_idx in segment_boundaries:
            segment_turns = session_turns[start_idx:end_idx+1]
            segment_content = self._format_segment_content(segment_turns)
            segment_timestamp = segment_turns[0].get('timestamp', '') if segment_turns else ''
            segments.append((segment_content, segment_timestamp))
        
        return segments
    
    def _format_session_for_segmentation(self, session_turns: List[Dict]) -> str:
        """Format session turns for segmentation analysis."""
        formatted_turns = []
        for i, turn in enumerate(session_turns):
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            # Add image context if available
            if turn.get('img_url') and turn.get('blip_caption'):
                text += f" [Image: {turn['blip_caption']}]"
            formatted_turns.append(f"Turn {i+1}: {speaker}: {text}")
        
        return "\n".join(formatted_turns)
    
    def _get_session_segments_sync(self, session_text: str, session_turns: List[Dict]) -> List[tuple[int, int]]:
        """
        Use GPT-4 to identify topically coherent segments within a single session.
        
        Args:
            session_text: Formatted session text
            session_turns: Original session turns for context
            
        Returns:
            List of (start_index, end_index) tuples for each segment within this session
        """
        try:
            from openai import OpenAI
            client = OpenAI()
            
            # Create segmentation prompt for a single session
            segmentation_prompt = f"""You are a conversation segmentation expert. Your task is to identify topically coherent segments within a single conversation session.

Given the following conversation session, identify natural segments where the topic or focus changes. Each segment should contain turns that are topically related.

Instructions:
1. Identify segments by looking for topic shifts, context changes, or natural conversation boundaries WITHIN this session
2. Each segment should be coherent and focused on a specific topic or set of related topics
3. Segments should be substantial enough to be meaningful.
4. If the entire session is about one coherent topic, you can return it as a single segment
5. Return the segments as a list of turn ranges in the format: "Start-End" (e.g., "1-5", "6-12", "13-18")
6. Turn numbers start from 1 and refer to turns within this session only
7. Avoid too short segments (e.g. just 1 or 2 turns)
8  Segments should be longer if possible!

Session to segment:
{session_text}

Please provide the segment boundaries as a comma-separated list of ranges (e.g., "1-5, 6-12, 13-18"):"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying topically coherent conversation segments within individual sessions."},
                    {"role": "user", "content": segmentation_prompt}
                ],
                max_tokens=300,
                temperature=0
            )
            
            # Parse the response to get segment boundaries
            segments_text = response.choices[0].message.content.strip()
            return self._parse_segment_boundaries(segments_text, len(session_turns))
            
        except Exception as e:
            print(f"Warning: Failed to get GPT-4 session segmentation: {e}")
            # Fallback to simple heuristic segmentation for this session
            return self._fallback_session_segmentation(session_turns)
    
    def _fallback_session_segmentation(self, session_turns: List[Dict]) -> List[tuple[int, int]]:
        """Fallback segmentation for a single session when GPT-4 is not available."""
        # Simple heuristic: segment every 8-10 turns within the session
        segments = []
        current_start = 0
        segment_size = 8
        
        while current_start < len(session_turns):
            end_idx = min(current_start + segment_size - 1, len(session_turns) - 1)
            segments.append((current_start, end_idx))
            current_start = end_idx + 1
        
        return segments if segments else [(0, len(session_turns) - 1)]
    
    def _parse_segment_boundaries(self, segments_text: str, total_turns: int) -> List[tuple[int, int]]:
        """Parse GPT-4 response to extract segment boundaries."""
        segments = []
        
        try:
            # Extract ranges like "1-5, 6-12, 13-18"
            ranges = [r.strip() for r in segments_text.split(',')]
            
            for range_str in ranges:
                if '-' in range_str:
                    start_str, end_str = range_str.split('-', 1)
                    start_idx = max(0, int(start_str.strip()) - 1)  # Convert to 0-based
                    end_idx = min(total_turns - 1, int(end_str.strip()) - 1)  # Convert to 0-based
                    
                    if start_idx <= end_idx:
                        segments.append((start_idx, end_idx))
            
            # Ensure we cover all turns
            if not segments:
                segments = [(0, total_turns - 1)]
            else:
                # Fill any gaps
                segments = sorted(segments)
                filled_segments = []
                current_end = -1
                
                for start, end in segments:
                    if start > current_end + 1:
                        # Fill gap
                        filled_segments.append((current_end + 1, start - 1))
                    filled_segments.append((start, end))
                    current_end = end
                
                # Handle final gap
                if current_end < total_turns - 1:
                    filled_segments.append((current_end + 1, total_turns - 1))
                
                segments = filled_segments
                
        except Exception as e:
            print(f"Warning: Failed to parse segment boundaries: {e}")
            # Fallback to single segment
            segments = [(0, total_turns - 1)]
        
        return segments
    
    def _format_segment_content(self, segment_turns: List[Dict]) -> str:
        """Format a segment's turns into content string."""
        turn_texts = []
        
        for turn in segment_turns:
            speaker_text = f"{turn['speaker']}: {turn['text']}"
            
            # Add image context if available
            if turn.get('img_url') and turn.get('blip_caption'):
                speaker_text += f" [Image: {turn['blip_caption']}]"
            
            turn_texts.append(speaker_text)
        
        return "\n".join(turn_texts)
        
    def extract_last_message(self, response: LettaResponse) -> str:
        """ Extract last answer_question tool response """
        for message in response.messages[::-1]:
            if message.message_type == "tool_return_message":
                if message.name == "answer_question":
                    return message.tool_return
        return ""

    def _create_files_for_all_samples(self, use_summary: bool = False):
        """Create conversation files for all samples during initialization."""
        import os
        from pathlib import Path
        import concurrent.futures
        
        # Create data directory if it doesn't exist
        data_dir = Path("leaderboard/locomo/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Group dataset by sample_id to avoid duplicates
        samples_by_id = {}
        for datum in self.dataset:
            if datum.sample_id not in samples_by_id:
                samples_by_id[datum.sample_id] = datum
        
        # Collect all file creation tasks
        file_creation_tasks = []
        
        # Prepare tasks for all samples
        for sample_id, datum in samples_by_id.items():
            print(f"Preparing files for sample {sample_id}")
            
            # Initialize sample_file_paths for this sample
            self.sample_file_paths[sample_id] = []
            
            # Create sample-specific directory
            sample_dir = data_dir / f"sample_{sample_id}"
            sample_dir.mkdir(exist_ok=True)
            
            # Prepare conversation chunks with timestamps
            conversation_chunks = self._create_conversation_string_with_timestamp(datum.conversation_history)
            
            # Create task for each chunk
            for i, (content, timestamp) in enumerate(conversation_chunks):
                # Determine filename
                if timestamp:
                    filename = timestamp.replace("/", "-").replace(":", "-") + ".txt"
                else:
                    filename = f"session_{i+1:03d}.txt"
                
                file_path = sample_dir / filename
                
                # Add to sample_file_paths immediately
                self.sample_file_paths[sample_id].append(str(file_path))
                
                # Only create task if file doesn't already exist
                if not file_path.exists():
                    file_creation_tasks.append({
                        'sample_id': sample_id,
                        'content': content,
                        'timestamp': timestamp,
                        'file_path': file_path,
                        'chunk_index': i
                    })
                else:
                    print(f"File {file_path} already exists, skipping task creation")
        
        # Sort file paths for consistent ordering
        for sample_id in self.sample_file_paths:
            self.sample_file_paths[sample_id] = sorted(self.sample_file_paths[sample_id])
            print(f"Sample {sample_id}: {len(self.sample_file_paths[sample_id])} files prepared")

        # Define the complete file creation function
        def create_single_file(task_data):
            """Create a single file with all content and metadata."""
            sample_id = task_data['sample_id']
            content = task_data['content']
            timestamp = task_data['timestamp']
            file_path = task_data['file_path']
            chunk_index = task_data['chunk_index']
            
            try:                
                # Write file with timestamp, summary, and content
                with open(file_path, "w", encoding="utf-8") as f:
                    # Line 1: Timestamp
                    if timestamp:
                        f.write(f"Timestamp: {timestamp}\n")
                    else:
                        f.write(f"Session {chunk_index+1}\n")
                    
                    # Add summary after timestamp
                    if use_summary:
                        f.write(f"{self._get_content_summary_sync(content)}\n")
                    
                    # Original content
                    f.write(content)
                
                print(f"Created file {file_path} for sample {sample_id}")
                return True  # Successfully created
                
            except Exception as e:
                print(f"Error creating file for sample {sample_id}: {e}")
                return False  # Failed
        
        # Process all file creation tasks in parallel
        print(f"Processing {len(file_creation_tasks)} file creation tasks in parallel...")
        
        # Execute file creation tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Submit all tasks
            futures = [executor.submit(create_single_file, task) for task in file_creation_tasks]
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Just ensure completion, we don't need the result since paths are already stored

    def _get_content_summary_sync(self, content: str) -> str:
        """Use OpenAI to summarize conversation content into topics and main points (synchronous)."""
        try:
            client = OpenAI()  # Use synchronous client
            
            prompt = f"""Summarize the following conversation content. Format your response as follows:
1. First line: "Topics: [list all topics covered in this conversation separated by commas]"
2. Following lines: "Main points:" followed by each main conversation point on a separate line, prefixed with "- "

Be concise and focus on the key information exchanged.

Conversation content:
{content}"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Warning: Failed to get OpenAI summary: {e}")
            # Fallback to simple summary
            lines = content.split('\n')
            speakers = set()
            for line in lines:
                if ':' in line:
                    speaker = line.split(':')[0].strip()
                    speakers.add(speaker)
            
            return f"Topics: conversation between {', '.join(speakers)}\nMain points:\n- Conversation content (summary failed)"

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

        agent_config["embedding_config"] = embedding_config

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
                
                # Upload all files first and collect jobs
                jobs = []
                for file_path in file_paths:
                    print(f"Uploading file {file_path}")
                    with open(file_path, "rb") as f:
                        job = await client.sources.files.upload(
                            source_id=source.id,
                            file=f,
                        )
                        jobs.append(job)
                
                # Wait for all jobs to complete
                while True:
                    all_completed = True
                    for job in jobs:
                        updated_job = await client.jobs.retrieve(job_id=job.id)
                        if updated_job.status not in ["completed", "failed"]:
                            all_completed = False
                            break
                    
                    if all_completed:
                        break
                    
                    print(f"Waiting for {len(jobs)} jobs to complete...")
                    time.sleep(1)
                
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
                        "The files will only contain a summary of the conversation history, and the whole conversation."
                        "Read through all the conversation context carefully before answering any questions."
                        "You can use the grep, search_files, and send_message tools to answer the question."
                        "For grep, try multiple times if you did not find the answer in the file."
                        "Please provide precise answers to the questions, do not hallucinate, and note the time! The timestamp is in the file name."
                        "Always refer to a file to answer the question, do not make up information."
                        "Avoid general searches, like just searching for a person's name."
                        "Try directly search the question in the conversation first, before you try some other queries."
                        "DO NOT RELY ON THE SUMMARY, IT IS NOT ACCURATE. WHEN IN DOUBE USE THE WHOLE CONVERSATION TO ANSWER THE QUESTION."
                        "Your final send_message should be a precise and concise answer to the question, do not include any other information."
                        "When expressing time, use a specific time."
            )
        ]

        with open("leaderboard/locomo/locomo_agent.txt", "r") as f:
            system_prompt = f.read()

        agent = await client.agents.create(
            agent_type="locomo_agent",
            include_base_tools=False,
            tools=["search_files", "answer_question"],
            # "grep", "open_file", "close_file",
            system=system_prompt,
            # memory_blocks=memory_blocks,
            tool_rules=[
                TerminalToolRule(tool_name="answer_question"),
                ContinueToolRule(tool_name="grep"),
                ContinueToolRule(tool_name="search_files"),
                InitToolRule(tool_name="search_files"),
            ],
            **agent_config,
        )

        all_tools = await client.agents.tools.list(agent_id=agent.id)
        print(f"Agent {agent.id} created with tools {[tool.name for tool in all_tools]}")

        print(f"Created agent {agent.id} for sample {datum.sample_id}")

        # attach the source to the agent
        await client.agents.sources.attach(agent.id, source_id)

        self.agent_datum_mapping[agent.id] = datum
        return agent.id



# Benchmark instances for different chunking strategies
locomo_qa_benchmark = LoCoMoQABenchmark(chunking_strategy="session")
locomo_qa_benchmark_file = LoCoMoQAFileBenchmark(chunking_strategy="session")
locomo_qa_benchmark_secom = LoCoMoQAFileBenchmark(chunking_strategy="secom")
# locomo_qa_turn_benchmark = LoCoMoQABenchmark(chunking_strategy="turn")
# locomo_qa_time_benchmark = LoCoMoQABenchmark(chunking_strategy="time_window")
