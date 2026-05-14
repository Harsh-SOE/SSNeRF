import math
from typing import Dict, Any 
from dataclasses import dataclass

@dataclass
class CameraConfig:
    resized_w: int = 400
    resized_h: int = 533

    focal_length_mm: float = 5.58
    sensor_width_mm: float = 5.52 

    camera_radius = 2.5

    @property
    def focal_length(self) -> float:
        """
        Calculates focal length in pixels using physical camera specs.
        Formula: f_px = f_mm * (image_width_px / sensor_width_mm)
        """
        return self.focal_length_mm * (self.resized_w / self.sensor_width_mm)

    @property
    def fov_x_deg(self) -> float:
        """Horizontal Field of View in degrees"""
        return 2 * math.degrees(math.atan(self.resized_w / 2 / self.focal_length))

    @property
    def fov_y_deg(self) -> float:
        """Vertical Field of View in degrees"""
        return 2 * math.degrees(math.atan(self.resized_h / 2 / self.focal_length))

    @property
    def nerf_intrinsics(self) -> Dict[str, Any]:
        """
        Outputs the exact parameters needed for standard NeRF transforms.json files.
        (Omitted camera angles as fl_x and fl_y are sufficient and mathematically superior).
        """
        f = round(self.focal_length, 2)
        return {
            "fl_x": f,
            "fl_y": f,
            "cx": self.resized_w / 2.0,
            "cy": self.resized_h / 2.0,
            "w": self.resized_w,
            "h": self.resized_h
        }