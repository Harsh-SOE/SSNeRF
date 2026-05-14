import torch
from dataclasses import dataclass, field

from camera import CameraConfig
from turn_table import TurnTableConfig
from training import TrainingConfig
from semantic import SemanticConfig

@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    rotation: TurnTableConfig = field(default_factory=TurnTableConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)

    sam_checkpoint: str = '/content/sam_vit_h_4b8939.pth'

    density_threshold: float = 12
    plant_bound: float = 0.8
    grid_res: int = 512

    device: torch.device = field(
        default_factory=lambda: torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    )