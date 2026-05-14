from dataclasses import dataclass

@dataclass
class SAMConfig:
  positive_points: int = 8
  negative_points: int = 12

  hsv_lower: tuple[int, int, int] = (25, 25, 25)
  hsv_upper: tuple[int, int, int] = (95, 255, 255)

  save_debug: bool = True