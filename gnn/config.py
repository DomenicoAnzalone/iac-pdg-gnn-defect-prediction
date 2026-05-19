from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    seed: int = 42
    data_root: str = "input/repositories"
    ansible_csv: str = "input/ansible.csv"
    output_root: str = "output/gnn"
    torch_device: Optional[str] = None  # e.g., "cuda:0" or "cpu"
    balance_strategy: str = "none"  # none | undersample | oversample
    max_retries: int = 1
    node_label_hash_dim: int = 16
    node_type_vocab_file: str = "output/gnn/manifests/node_type_vocab.json"


DEFAULT_CONFIG = Config()
