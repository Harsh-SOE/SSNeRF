from dataclasses import dataclass

@dataclass
class TurnTableConfig:
    angle_period: int = 18

    use_top_view: bool = False

    @property
    def rotation_angles(self):
        return list(range(0, 360, self.angle_period))

    @property
    def num_images(self):
        return len(self.rotation_angles)