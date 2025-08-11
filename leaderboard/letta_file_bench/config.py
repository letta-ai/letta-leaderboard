"""Configuration loader for Letta File Benchmark."""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AgentConfig:
    max_files_open: int
    per_file_view_window_char_limit: int
    embedding_chunk_size: int

@dataclass
class UploadConfig:
    timeout_seconds: int

@dataclass
class QuestionTypes:
    single_hop_pct: float
    multi_hop_pct: float
    comparison_pct: float

@dataclass
class GenerationConfig:
    temperature: float
    max_concurrent: int
    seed: int
    question_types: QuestionTypes

@dataclass
class MaxPerPerson:
    addresses: int
    accounts: int
    employments: int
    credit_cards: int
    vehicles: int
    pets: int
    net_accounts: int
    insurances: int
    medical_records: int

@dataclass
class AgeRange:
    min: int
    max: int

@dataclass
class DataGenerationConfig:
    num_people: int
    max_per_person: MaxPerPerson
    age_range: AgeRange
    password_length: int
    seed: int

@dataclass
class UtilsConfig:
    max_uniqueness_attempts: int
    json_indent: int

@dataclass
class Config:
    agent: AgentConfig
    upload: UploadConfig
    generation: GenerationConfig
    data_generation: DataGenerationConfig
    utils: UtilsConfig

def load_config() -> Config:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return Config(
        agent=AgentConfig(**data['agent']),
        upload=UploadConfig(**data['upload']),
        generation=GenerationConfig(
            temperature=data['generation']['temperature'],
            max_concurrent=data['generation']['max_concurrent'],
            seed=data['generation']['seed'],
            question_types=QuestionTypes(**data['generation']['question_types'])
        ),
        data_generation=DataGenerationConfig(
            num_people=data['data_generation']['num_people'],
            max_per_person=MaxPerPerson(**data['data_generation']['max_per_person']),
            age_range=AgeRange(**data['data_generation']['age_range']),
            password_length=data['data_generation']['password_length'],
            seed=data['data_generation']['seed']
        ),
        utils=UtilsConfig(**data['utils'])
    )

# Load config once at module import
CONFIG = load_config()