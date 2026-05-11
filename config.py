
from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    experiment: int = 4
    batch_size: int = 64
    test_batch_size: int = 256
    epochs: int = 1
    seed: int = 42

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    data_root: str = "./data"

    base_lr: float = 0.01
    momentum: float = 0.9
