import asyncio
import time
import uuid
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from letta_client import (
    CreateBlock,
    AsyncLetta,
    LettaMessageUnion,
    LettaResponse,
    EmbeddingConfig, RequiredBeforeExitToolRule,
)
from leaderboard.benchmark import Benchmark
from datasets import load_dataset
from leaderboard.evaluate import EvaluationResult
from leaderboard.utils import (
    Dotdict,
    grade_sample,
    UsageStatistics,
)


class LettaFileBenchmark(Benchmark):
    def __init__(self):
        # Load dataset from file benchmark questions
        self.required_tool_ids = None
        self.source_id = None
        raw = load_dataset(
            "json",
            data_files="leaderboard/letta_file_bench/data/llm_generated_questions.jsonl",
        )["train"]
        self.raw_datasets = raw
        
        # Track file-specific agent interactions
        self.agent_file_messages: Dict[str, List[LettaMessageUnion]] = {}
        self.agent_datum_mapping: Dict[str, Dotdict] = {}
        
        self.dataset = self._build_dataset()
        self.benchmark_type = "feature"
        self.source_name = "file_benchmark_data"

    def _build_dataset(self) -> List[Dotdict]:
        """
        Build dataset from raw file benchmark data
        Each item contains a question, answer, required files, and reasoning steps
        """
        data: List[Dotdict] = []

        for raw_datum in self.raw_datasets:
            data.append(
                Dotdict(
                    {
                        "message": raw_datum["question"],
                        "message_list": [raw_datum["question"]],
                        "answer": raw_datum["answer"],
                        "difficulty": raw_datum["difficulty"],
                        "question_type": raw_datum["question_type"],
                        "required_files": raw_datum["required_files"],
                        "reasoning_steps": raw_datum["reasoning_steps"],
                    }
                )
            )

        return data

    async def setup_required_tools(self, client: AsyncLetta):
        # Get required tool_ids
        required_tool_names = {"send_message", "open_files", "grep_files"}
        self.required_tool_ids = []
        for tool_name in required_tool_names:
            tools = await client.tools.list(name=tool_name)
            self.required_tool_ids.append(tools[0].id)

    async def setup_sources(self, client: AsyncLetta, embedding_config: EmbeddingConfig, force_refresh=False):
        try:
            self.source_id = await client.sources.retrieve_by_name(self.source_name)
        except Exception:
            self.source_id = None

        if force_refresh or not self.source_id:
            if self.source_id:
                print(f"[LettaFileBenchmark] Deleting existing source: {self.source_name}")
                await client.sources.delete(source_id=self.source_id)

            # Create source
            print(f"[LettaFileBenchmark] Creating new source: {self.source_name}")
            source = await client.sources.create(
                name=self.source_name,
                embedding_chunk_size=512,
                embedding_config=embedding_config,
                description="Structured personal data repository in JSONL format containing information about people, vehicles, pets, bank accounts, credit cards, medical records, internet accounts, insurance policies, employment records, and addresses.",
                instructions="Use this data to answer questions about individuals and their associated records. When searching, use person names or IDs to find related information across different files. Cross-reference between files using person_id when needed.",
            )
            self.source_id = source.id

            # Load all .jsonl files from the data directory (excluding questions file)
            data_dir = Path("leaderboard/letta_file_bench/data")
            jsonl_files = [f for f in data_dir.glob("*.jsonl") if f.name != "llm_generated_questions.jsonl"]
            
            await self._upload_files_concurrent(client, jsonl_files)
        else:
            print(f"[LettaFileBenchmark] Using existing source: {self.source_id}")

    async def _upload_file_and_wait(self, client: AsyncLetta, file_path: Path, max_wait: int = 30):
        """Upload a single file and wait for processing to complete"""
        with open(file_path, "rb") as f:
            file_metadata = await client.sources.files.upload(
                source_id=self.source_id,
                file=f
            )

        # Wait for the file to be processed
        start_time = time.time()
        while file_metadata.processing_status not in ["completed", "error"]:
            if time.time() - start_time > max_wait:
                raise TimeoutError(f"File {file_path.name} processing timed out after {max_wait} seconds")
            
            await asyncio.sleep(1)
            file_metadata = await client.sources.get_file_metadata(
                source_id=self.source_id, 
                file_id=file_metadata.id
            )

        if file_metadata.processing_status == "error":
            raise RuntimeError(f"File {file_path.name} processing failed: {file_metadata.error_message}")

        return file_metadata

    async def _upload_files_concurrent(self, client: AsyncLetta, file_paths: List[Path]):
        """Upload multiple files concurrently with progress tracking"""
        # Create tasks for all files
        tasks = [self._upload_file_and_wait(client, file_path) for file_path in file_paths]
        
        # Execute with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Uploading and processing files") as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    results.append(result)
                    pbar.set_postfix({"completed": result.file_name})
                except Exception as e:
                    print(f"Error uploading file: {e}")
                    results.append(None)
                finally:
                    pbar.update(1)
        
        successful_uploads = [r for r in results if r is not None]
        print(f"[LettaFileBenchmark] Successfully uploaded {len(successful_uploads)}/{len(file_paths)} files")
        return results

    async def setup_agent(self, datum: Dotdict, client: AsyncLetta, agent_id: str):
        """
        Set up agent with file system access
        TODO: This will need to be implemented to provide file access to the agent
        """
        pass

    async def create_agent_fun(
        self,
        client: AsyncLetta,
        datum: Dotdict,
        agent_config: dict,
    ) -> str:
        """
        Create agent with file system capabilities
        """
        # Ensure agent_config contains required keys
        assert "llm_config" in agent_config, "agent_config must contain 'llm_config'"
        assert "embedding_config" in agent_config, "agent_config must contain 'embedding_config'"
        assert self.source_id, "Did you forget to setup sources?"

        # Create detailed memory block with comprehensive file schemas
        file_system_block = CreateBlock(
            label="Personal Data Repository Schema",
            description="Schema reference for a personal data repository with 10 JSONL files containing structured information about people and their associated records (vehicles, financial data, pets, medical records, etc.). Each file contains JSON objects linked by person_id/owner_id. Use open_files tool to read files based on question requirements.",
            value=(
                "## FILE SCHEMAS\n\n"
                
                "### people.jsonl - Master Person Directory\n"
                "Schema: {\"full_name\": str, \"person_id\": str, \"dob\": str, \"email\": str, \"phone\": str}\n"
                "Example: {\"full_name\":\"Robert Smith\",\"person_id\":\"pers-0001\",\"dob\":\"1945-04-05\",\"email\":\"user@example.net\",\"phone\":\"(825)416-4829\"}\n\n"
                
                "### vehicles.jsonl - Vehicle Registration Records\n"
                "Schema: {\"vehicle_id\": str, \"owner_id\": str, \"make\": str, \"model\": str, \"year\": int, \"license_plate\": str}\n"
                "Linkage: owner_id field contains person_id\n\n"
                
                "### bank_accounts.jsonl - Banking Information\n"
                "Schema: {\"account_id\": str, \"owner_id\": str, \"bank_name\": str, \"routing\": str, \"account_no\": str, \"balance\": float, \"currency\": str}\n"
                "Linkage: owner_id field contains person_id\n\n"
                
                "### credit_cards.jsonl - Credit Card Records\n"
                "Schema: {\"card_id\": str, \"owner_id\": str, \"provider\": str, \"number\": str, \"expire\": str, \"cvc\": str}\n"
                "Linkage: owner_id field contains person_id\n\n"
                
                "### pets.jsonl - Pet Ownership Records\n"
                "Schema: {\"pet_id\": str, \"owner_id\": str, \"name\": str, \"species\": str, \"breed\": str}\n"
                "Linkage: owner_id field contains person_id\n\n"
                
                "### medical_records.jsonl - Health & Medical Data\n"
                "Schema: {\"record_id\": str, \"owner_id\": str, \"ssn\": str, \"blood_type\": str, \"condition\": str}\n"
                "Linkage: owner_id field contains person_id\n\n"
                
                "### addresses.jsonl - Address Directory\n"
                "Schema: {\"address_id\": str, \"owner_id\": str, \"street\": str, \"city\": str, \"state\": str, \"postal_code\": str, \"country\": str}\n"
                "Linkage: owner_id field contains person_id\n"
                "Note: Individuals may have multiple addresses\n\n"
                
                "### employments.jsonl - Employment History\n"
                "Schema: {\"employment_id\": str, \"owner_id\": str, \"employer\": str, \"job_title\": str, \"start_date\": str, \"salary\": float, \"currency\": str}\n"
                "Linkage: owner_id field contains person_id\n\n"
                
                "### insurance_policies.jsonl - Insurance Coverage\n"
                "Schema: {\"policy_id\": str, \"owner_id\": str, \"insurer\": str, \"policy_type\": str, \"policy_number\": str, \"expires\": str}\n"
                "Linkage: owner_id field contains person_id\n\n"
                
                "### internet_accounts.jsonl - Online Account Information\n"
                "Schema: {\"net_id\": str, \"owner_id\": str, \"username\": str, \"email\": str, \"url\": str, \"password\": str}\n"
                "Linkage: owner_id field contains person_id\n\n"
                
                "Use the open_files tool to read these JSONL files when answering questions. Each line in a JSONL file is a separate JSON object. You may need to read multiple files to find all the information needed."
            )
        )

        # Create unique agent name with question preview and UUID
        # Clean question text to only contain alphanumeric and underscores
        question_preview = "".join(c if c.isalnum() else "_" for c in datum.message[:30])
        agent_name = f"file_bench_{question_preview}_{str(uuid.uuid4())[:8]}"

        agent = await client.agents.create(
            name=agent_name,
            memory_blocks=[file_system_block],
            source_ids=[self.source_id],
            **agent_config,
            include_base_tools=False,
            tool_rules=[RequiredBeforeExitToolRule(tool_name="send_message")]
        )
        return agent.id

    async def get_response(
        self, client: AsyncLetta, agent_id: str, datum: Dotdict
    ) -> LettaResponse:
        return await super().get_response(client, agent_id, datum)

    async def metric(
        self, predicted: str, true: str, datum: Dotdict, agent_id: str
    ) -> float:
        """
        Grade file benchmark responses using LLM judge
        """
        result = await grade_sample(datum.message, true, predicted)
        return 1.0 if result == "A" else 0.0


class FileOpenBenchmark(LettaFileBenchmark):
    """
    Benchmark for testing file opening and reading capabilities
    Tests the agent's ability to use the open_files tool to read file contents
    and answer questions based on the information found in those files
    """

    async def setup_agent(self, datum: Dotdict, client: AsyncLetta, agent_id: str):
        """
        Set up agent with access to the required files for this specific question
        """
        # Store the datum mapping for usage statistics
        self.agent_datum_mapping[agent_id] = datum

        # Attach only tools
        await client.agents.modify(agent_id=agent_id, tool_ids=self.required_tool_ids)


    async def get_usage_statistics(
        self,
        client: AsyncLetta,
        agent_ids: List[str],
        evaluation_result: EvaluationResult,
    ) -> UsageStatistics:
        """
        Track file opening and reading usage statistics
        TODO: This should track which files were opened, how many times, etc.
        """
        return UsageStatistics(
            {
                # TODO: Track file-specific usage statistics
                # Examples:
                # "files_opened": number of unique files opened
                # "total_file_operations": total open_files calls
                # "average_files_per_question": avg files opened per question
            },
            {
                agent_id: {
                    "required_files": self.agent_datum_mapping[agent_id].required_files,
                    "question_type": self.agent_datum_mapping[agent_id].question_type,
                    "difficulty": self.agent_datum_mapping[agent_id].difficulty,
                }
                for agent_id in agent_ids 
                if agent_id in self.agent_datum_mapping
            },
        )


# Benchmark instance for evaluation system
file_open_benchmark = FileOpenBenchmark()