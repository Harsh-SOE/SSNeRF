from typing import Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class SemanticConfig:
    num_classes: int = 5

    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: 'background',
        1: 'leaf',
        2: 'stem',
        3: 'petiole',
        4: 'apex'
    })

    class_colors_bgr: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: {
        0: (0, 0, 0),
        1: (0, 200, 0),
        2: (0, 140, 255),
        3: (0, 255, 255),
        4: (0, 0, 200)
    })