"""
Configuration for Resonant Interference Network

Centralized configuration using dataclasses for clean management.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ModelConfig:
    """Configuration for the RIN model architecture."""
    
    # Vocabulary and embeddings
    vocab_size: int = 50257  # GPT-2 vocabulary size
    embed_dim: int = 256
    
    # Resonant layers
    hidden_dim: int = 512
    num_layers: int = 1
    num_heads: int = 4
    neurons_per_head: int = 128
    use_multi_head: bool = True
    
    # Sequence handling
    max_seq_len: int = 512
    
    # Regularization
    dropout: float = 0.0
    
    # LUT configuration
    lut_resolution: int = 512
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.embed_dim > 0, "embed_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.lut_resolution >= 64, "lut_resolution should be at least 64"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Basic training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 10
    max_steps: Optional[int] = None  # If set, overrides epochs
    
    # Learning rate schedule
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    min_lr: float = 1e-5
    
    # Gradient handling
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging and saving
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None


@dataclass 
class DataConfig:
    """Configuration for data loading."""
    
    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    
    # Tokenization
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    
    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Train/val split
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"


@dataclass
class RINConfig:
    """Complete configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Experiment
    experiment_name: str = "rin_lm"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "RINConfig":
        """Create config from dictionary."""
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        train_cfg = TrainingConfig(**config_dict.get("training", {}))
        data_cfg = DataConfig(**config_dict.get("data", {}))
        
        return cls(
            model=model_cfg,
            training=train_cfg,
            data=data_cfg,
            device=config_dict.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            seed=config_dict.get("seed", 42),
            experiment_name=config_dict.get("experiment_name", "rin_lm"),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "device": self.device,
            "seed": self.seed,
            "experiment_name": self.experiment_name,
        }


# Preset configurations for quick experiments
TINY_CONFIG = RINConfig(
    model=ModelConfig(
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        num_heads=2,
        neurons_per_head=64,
        max_seq_len=256,
    ),
    training=TrainingConfig(
        batch_size=16,
        learning_rate=1e-3,
        max_epochs=5,
    ),
)

SMALL_CONFIG = RINConfig(
    model=ModelConfig(
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        num_heads=4,
        neurons_per_head=128,
        max_seq_len=512,
    ),
    training=TrainingConfig(
        batch_size=32,
        learning_rate=5e-4,
        max_epochs=10,
    ),
)

BASE_CONFIG = RINConfig(
    model=ModelConfig(
        embed_dim=512,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
        neurons_per_head=128,
        max_seq_len=1024,
    ),
    training=TrainingConfig(
        batch_size=64,
        learning_rate=3e-4,
        max_epochs=20,
    ),
)
