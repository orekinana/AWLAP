from dataclasses import field, dataclass
from typing import Optional
from transformers import TrainingArguments


@dataclass
class DataArguments:
    t_region_path: str = field(default=None)
    train_data_path: str = field(default=None)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    num_heads: int = field(default=1)
    scl_alpha: float = field(default=1.0)
    

@dataclass
class FineTuneTrainingArguments(TrainingArguments):
    local_rank: int = field(default=0)

