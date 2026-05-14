from dataclasses import dataclass

@dataclass
class EarlyStopConfig:
    """Configuration for training convergence and termination criteria."""
    min_iters: int = 4000
    plant_acc: float = 0.88
    bg_acc: float = 0.05
    psnr_min: float = 22.0
    patience: int = 600