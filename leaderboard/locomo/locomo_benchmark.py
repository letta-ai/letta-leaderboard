"""
LoCoMo (Long-term Conversational Memory) Question Answering Benchmark

This benchmark evaluates LLM agents' ability to answer questions based on
very long-term conversational memory across multiple sessions.

Based on the paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
by Maharana et al. (2024)
"""

import time
from typing import List, Dict, Optional, Literal
import json
import asyncio
from pathlib import Path
from openai import OpenAI
import concurrent.futures

from letta_client import (
    ContinueToolRule,
    CreateBlock,
    AsyncLetta,
    EmbeddingConfig,
    InitToolRule,
    LettaResponse,
    TerminalToolRule,
)
from leaderboard.benchmark import Benchmark
from leaderboard.evaluate import EvaluationResult
from leaderboard.utils import (
    Dotdict,
    grade_sample,
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


class LoCoMoQAFileBenchmark(Benchmark):
    """
    LoCoMo Question Answering benchmark for evaluating very long-term conversational memory.

    Similar to LoCoMoQABenchmark, but uses a file-based memory solution.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        chunking_strategy: Literal[
            "turn", "session", "time_window", "secom"
        ] = "session",
        use_summary: bool = False,
    ):
        self.data_path = data_path or "leaderboard/locomo/locomo10.json"
        self.chunking_strategy = chunking_strategy
        self.raw_data = self._load_dataset()
        self.dataset = self._build_dataset()
        self.template_agent_locks: Dict[str, asyncio.Lock] = {}
        self.benchmark_type = "feature"
        self.data_source_ids = {}
        self.sample_file_paths = {}  # sample_id -> list of file paths

        # Create files for all samples during initialization
        if chunking_strategy == "session":
            self._create_files_for_all_samples(use_summary=use_summary)
        elif chunking_strategy == "secom":
            self._create_secom_files_for_all_samples()
        self.tool_functions.append(answer_question)

    def _load_dataset(self) -> List[Dict]:
        """Load the LoCoMo dataset from JSON file."""
        dataset_path = Path(self.data_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"LoCoMo dataset not found at {self.data_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    async def metric(
        self, predicted: str, true: str, datum: Dotdict, agent_id: str
    ) -> float:
        result = await grade_sample(datum.message, true, predicted)
        return 1.0 if result == "A" else 0.0

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

                data.append(
                    Dotdict(
                        {
                            "sample_id": sample["sample_id"],
                            "conversation_history": conversation_history,
                            "message": qa_item["question"],
                            "answer": answer,
                            "category": qa_item.get("category", "unknown"),
                            "evidence": qa_item.get("evidence", []),
                            "speakers": {
                                "speaker_a": sample.get("conversation", {}).get(
                                    "speaker_a", "Speaker A"
                                ),
                                "speaker_b": sample.get("conversation", {}).get(
                                    "speaker_b", "Speaker B"
                                ),
                            },
                        }
                    )
                )

        return data

    def _extract_conversation_history(self, sample: Dict) -> List[Dict]:
        """Extract and format conversation history from sample."""
        conversation = sample.get("conversation", {})
        history = []

        # Get all sessions in chronological order
        session_keys = sorted(
            [k for k in conversation.keys() if k.startswith("session_")]
        )

        for session_key in session_keys:
            # Get session timestamp
            session_timestamp_key = f"{session_key}_date_time"
            session_timestamp = conversation.get(session_timestamp_key, "")

            session_data = conversation[session_key]
            if isinstance(session_data, list):
                for turn in session_data:
                    history.append(
                        {
                            "session": session_key,
                            "speaker": turn.get("speaker", ""),
                            "text": turn.get("text", ""),
                            "dia_id": turn.get("dia_id", ""),
                            "img_url": turn.get("img_url", ""),
                            "blip_caption": turn.get("blip_caption", ""),
                            "timestamp": session_timestamp,  # Use session-level timestamp
                        }
                    )

        return history

    def _create_secom_files_for_all_samples(self):
        """Create SeCom-segmented conversation files for all samples during initialization."""

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
            # Initialize sample_file_paths for this sample
            self.sample_file_paths[sample_id] = []

            # Create sample-specific directory
            sample_dir = data_dir / f"sample_{sample_id}"
            sample_dir.mkdir(exist_ok=True)

            # Check if summary file exists
            summary_file = sample_dir / f"{sample_id}_segment_summary.txt"
            if summary_file.exists():
                # Read filenames from summary file
                with open(summary_file, "r", encoding="utf-8") as f:
                    for line in f:
                        filename = line.strip()
                        if filename:
                            file_path = sample_dir / filename
                            self.sample_file_paths[sample_id].append(str(file_path))
                continue

            # Get conversation segments using SeCom approach
            conversation_segments = self._segment_conversation_secom(
                datum.conversation_history
            )

            # Create task for each segment and collect filenames
            segment_filenames = []
            for i, (content, timestamp) in enumerate(conversation_segments):
                # Determine filename based on segment index
                filename = f"segment_{i+1:03d}.txt"
                if timestamp:
                    filename = (
                        f"{timestamp.replace('/', '-').replace(':', '-')}_{i+1:03d}.txt"
                    )

                file_path = sample_dir / filename
                segment_filenames.append(filename)

                # Add to sample_file_paths immediately
                self.sample_file_paths[sample_id].append(str(file_path))

                # Only create task if file doesn't already exist
                if not file_path.exists():
                    file_creation_tasks.append(
                        {
                            "sample_id": sample_id,
                            "content": content,
                            "timestamp": timestamp,
                            "file_path": file_path,
                            "segment_index": i,
                        }
                    )

            # Create summary file with segment filenames
            with open(summary_file, "w", encoding="utf-8") as f:
                for filename in segment_filenames:
                    f.write(f"{filename}\n")

        # Sort file paths for consistent ordering
        for sample_id in self.sample_file_paths:
            self.sample_file_paths[sample_id] = sorted(
                self.sample_file_paths[sample_id]
            )

        # Define the complete file creation function
        def create_single_file(task_data):
            """Create a single file with segment content."""
            sample_id = task_data["sample_id"]
            content = task_data["content"]
            timestamp = task_data["timestamp"]
            file_path = task_data["file_path"]
            segment_index = task_data["segment_index"]

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

                return True  # Successfully created

            except Exception as e:
                return False  # Failed

        # Execute file creation tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Submit all tasks
            futures = [
                executor.submit(create_single_file, task)
                for task in file_creation_tasks
            ]

            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Just ensure completion

    def _segment_conversation_secom(
        self, conversation_history: List[Dict]
    ) -> list[tuple[str, str]]:
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

    def _group_conversation_by_sessions(
        self, conversation_history: List[Dict]
    ) -> List[List[Dict]]:
        """Group conversation turns by session."""
        sessions = []
        current_session = None
        current_session_turns = []

        for turn in conversation_history:
            if turn["session"] != current_session:
                # Save previous session if exists
                if current_session_turns:
                    sessions.append(current_session_turns)

                # Start new session
                current_session = turn["session"]
                current_session_turns = [turn]
            else:
                current_session_turns.append(turn)

        # Add final session
        if current_session_turns:
            sessions.append(current_session_turns)

        return sessions

    def _segment_single_session_secom(
        self, session_turns: List[Dict]
    ) -> list[tuple[str, str]]:
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
            segment_timestamp = session_turns[0].get("timestamp", "")
            return [(segment_content, segment_timestamp)]

        # Convert session to text format for segmentation
        session_text = self._format_session_for_segmentation(session_turns)

        # Get segment boundaries using GPT-4 for this session
        segment_boundaries = self._get_session_segments_sync(
            session_text, session_turns
        )

        # Create segments based on boundaries
        segments = []
        for start_idx, end_idx in segment_boundaries:
            segment_turns = session_turns[start_idx : end_idx + 1]
            segment_content = self._format_segment_content(segment_turns)
            segment_timestamp = (
                segment_turns[0].get("timestamp", "") if segment_turns else ""
            )
            segments.append((segment_content, segment_timestamp))

        return segments

    def _format_session_for_segmentation(self, session_turns: List[Dict]) -> str:
        """Format session turns for segmentation analysis."""
        formatted_turns = []
        for i, turn in enumerate(session_turns):
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            # Add image context if available
            if turn.get("img_url") and turn.get("blip_caption"):
                text += f" [Image: {turn['blip_caption']}]"
            formatted_turns.append(f"Turn {i+1}: {speaker}: {text}")

        return "\n".join(formatted_turns)

    def _get_session_segments_sync(
        self, session_text: str, session_turns: List[Dict]
    ) -> List[tuple[int, int]]:
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
                    {
                        "role": "system",
                        "content": "You are an expert at identifying topically coherent conversation segments within individual sessions.",
                    },
                    {"role": "user", "content": segmentation_prompt},
                ],
                max_tokens=300,
                temperature=0,
            )

            # Parse the response to get segment boundaries
            segments_text = response.choices[0].message.content.strip()
            return self._parse_segment_boundaries(segments_text, len(session_turns))

        except Exception as e:
            # Fallback to simple heuristic segmentation for this session
            return self._fallback_session_segmentation(session_turns)

    def _fallback_session_segmentation(
        self, session_turns: List[Dict]
    ) -> List[tuple[int, int]]:
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

    def _parse_segment_boundaries(
        self, segments_text: str, total_turns: int
    ) -> List[tuple[int, int]]:
        """Parse GPT-4 response to extract segment boundaries."""
        segments = []

        try:
            # Extract ranges like "1-5, 6-12, 13-18"
            ranges = [r.strip() for r in segments_text.split(",")]

            for range_str in ranges:
                if "-" in range_str:
                    start_str, end_str = range_str.split("-", 1)
                    start_idx = max(0, int(start_str.strip()) - 1)  # Convert to 0-based
                    end_idx = min(
                        total_turns - 1, int(end_str.strip()) - 1
                    )  # Convert to 0-based

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
            # Fallback to single segment
            segments = [(0, total_turns - 1)]

        return segments

    def _format_segment_content(self, segment_turns: List[Dict]) -> str:
        """Format a segment's turns into content string."""
        turn_texts = []

        for turn in segment_turns:
            speaker_text = f"{turn['speaker']}: {turn['text']}"

            # Add image context if available
            if turn.get("img_url") and turn.get("blip_caption"):
                speaker_text += f" [Image: {turn['blip_caption']}]"

            turn_texts.append(speaker_text)

        return "\n".join(turn_texts)

    def extract_last_message(self, response: LettaResponse) -> str:
        """Extract last answer_question tool response"""
        for message in response.messages[::-1]:
            if message.message_type == "tool_return_message":
                if message.name == "answer_question":
                    return message.tool_return
        return ""

    def _create_files_for_all_samples(self, use_summary: bool = False):
        """Create conversation files for all samples during initialization."""
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
            # Initialize sample_file_paths for this sample
            self.sample_file_paths[sample_id] = []

            # Create sample-specific directory
            sample_dir = data_dir / f"sample_{sample_id}"
            sample_dir.mkdir(exist_ok=True)

            # Prepare conversation chunks with timestamps
            conversation_chunks = self._create_conversation_string_with_timestamp(
                datum.conversation_history
            )

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
                    file_creation_tasks.append(
                        {
                            "sample_id": sample_id,
                            "content": content,
                            "timestamp": timestamp,
                            "file_path": file_path,
                            "chunk_index": i,
                        }
                    )

        # Sort file paths for consistent ordering
        for sample_id in self.sample_file_paths:
            self.sample_file_paths[sample_id] = sorted(
                self.sample_file_paths[sample_id]
            )

        # Define the complete file creation function
        def create_single_file(task_data):
            """Create a single file with all content and metadata."""
            sample_id = task_data["sample_id"]
            content = task_data["content"]
            timestamp = task_data["timestamp"]
            file_path = task_data["file_path"]
            chunk_index = task_data["chunk_index"]

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

                return True  # Successfully created

            except Exception as e:
                return False  # Failed

        # Process all file creation tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Submit all tasks
            futures = [
                executor.submit(create_single_file, task)
                for task in file_creation_tasks
            ]

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
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes conversations concisely.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0,
            )

            return response.choices[0].message.content.strip()

        except Exception:
            # Fallback to simple summary
            lines = content.split("\n")
            speakers = set()
            for line in lines:
                if ":" in line:
                    speaker = line.split(":")[0].strip()
                    speakers.add(speaker)

            return f"Topics: conversation between {', '.join(speakers)}\nMain points:\n- Conversation content (summary failed)"

    def _create_conversation_string_with_timestamp(
        self, conversation_history: List[Dict]
    ) -> list[tuple[str, str]]:
        """Create a list of (content, timestamp) tuples representing the conversation history."""
        if self.chunking_strategy == "turn":
            return self._chunk_by_turn_with_timestamp(conversation_history)
        elif self.chunking_strategy == "session":
            return self._chunk_by_session_with_timestamp(conversation_history)
        elif self.chunking_strategy == "time_window":
            return self._chunk_by_time_window_with_timestamp(conversation_history)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")

    def _chunk_by_turn_with_timestamp(
        self, conversation_history: List[Dict]
    ) -> list[tuple[str, str]]:
        """Chunk conversation with each turn as a separate message, returning (content, timestamp)."""
        chunks = []
        for turn in conversation_history:
            content = f"{turn['speaker']}: {turn['text']}"
            if turn.get("img_url") and turn.get("blip_caption"):
                content += f" [Image: {turn['blip_caption']}]"

            timestamp = turn.get("timestamp", "")
            chunks.append((content, timestamp))
        return chunks

    def _format_session_chunk_with_timestamp(
        self, turns: List[Dict]
    ) -> tuple[str, str]:
        """Format turns from a session into a content string and a timestamp."""
        turn_texts = []
        session_timestamp = ""

        for turn in turns:
            speaker_text = f"{turn['speaker']}: {turn['text']}"
            if turn.get("img_url") and turn.get("blip_caption"):
                speaker_text += f" [Image: {turn['blip_caption']}]"
            turn_texts.append(speaker_text)

            if not session_timestamp and turn.get("timestamp"):
                session_timestamp = turn["timestamp"]

        content = "\n".join(turn_texts)
        return content, session_timestamp

    def _chunk_by_session_with_timestamp(
        self, conversation_history: List[Dict]
    ) -> list[tuple[str, str]]:
        """Chunk conversation by session, returning (content, timestamp)."""
        chunks = []
        current_session = None
        current_chunk_turns = []

        for turn in conversation_history:
            if turn["session"] != current_session:
                if current_chunk_turns:
                    session_content, session_timestamp = (
                        self._format_session_chunk_with_timestamp(current_chunk_turns)
                    )
                    chunks.append((session_content, session_timestamp))

                current_session = turn["session"]
                current_chunk_turns = [turn]
            else:
                current_chunk_turns.append(turn)

        if current_chunk_turns:
            session_content, session_timestamp = (
                self._format_session_chunk_with_timestamp(current_chunk_turns)
            )
            chunks.append((session_content, session_timestamp))

        return chunks

    def _chunk_by_time_window_with_timestamp(
        self, conversation_history: List[Dict], window_size: int = 10
    ) -> list[tuple[str, str]]:
        """Chunk conversation by time window, returning (content, timestamp)."""
        chunks = []
        for i in range(0, len(conversation_history), window_size):
            window_turns = conversation_history[i : i + window_size]
            turn_texts = []
            window_timestamp = ""

            for turn in window_turns:
                speaker_text = f"{turn['speaker']}: {turn['text']}"
                if turn.get("img_url") and turn.get("blip_caption"):
                    speaker_text += f" [Image: {turn['blip_caption']}]"
                turn_texts.append(speaker_text)

                if not window_timestamp and turn.get("timestamp"):
                    window_timestamp = turn["timestamp"]

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
        assert (
            "embedding_config" in agent_config
        ), "agent_config must contain 'embedding_config'"
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
                source = await client.sources.create(
                    name=f"Conversation Context for {datum.sample_id}",
                    embedding_config=embedding_config,
                    description=f"The conversation history, containing timestamped messages from {datum.speakers['speaker_a']} and {datum.speakers['speaker_b']}.",
                )

                # Upload the pre-created files
                file_paths = self.sample_file_paths[datum.sample_id]

                # Upload all files first and collect jobs
                jobs = []
                for file_path in file_paths:
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

                    time.sleep(1)

                self.data_source_ids[datum.sample_id] = source.id

        source_id = self.data_source_ids[datum.sample_id]

        # create a new agent for this sample_id, with file capabilities
        agent_config.pop("agent_type", None)
        # no search_files tool now

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

        # attach the source to the agent
        await client.agents.sources.attach(agent.id, source_id)

        return agent.id


# Benchmark instances for different chunking strategies
locomo_qa_benchmark_file = LoCoMoQAFileBenchmark(chunking_strategy="session")
locomo_qa_benchmark_secom = LoCoMoQAFileBenchmark(chunking_strategy="secom")
