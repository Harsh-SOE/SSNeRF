import torch
import math
import torch.nn as nn
import tinycudann as tcnn


class ProgressiveTCNNHashEncoding(nn.Module):
    """
    tiny-cuda-nn HashGrid wrapper with your same progressive level schedule.

    Input/output contract matches ProgressiveHashEncoding:
      input:  positions normalized to approximately [-1, 1]
      output: [N, n_levels * n_features]
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features: int = 2,
        log2_table: int = 18,
        base_res: int = 8,
        max_res: int = 512,
        start_level: int = 4,
        warmup_start: int = 500,
        warmup_end: int = 6000,
        interpolation: str = "Linear",
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.start_level = start_level
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end
        self.per_level_scale = math.exp(math.log(max_res / base_res) / (n_levels - 1))


        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": n_levels,
            "n_features_per_level": n_features,
            "log2_hashmap_size": log2_table,
            "base_resolution": base_res,
            "per_level_scale": self.per_level_scale,
            "interpolation": interpolation,
        }

        self.encoding = tcnn.Encoding(3, encoding_config, dtype=None)
        self._out_dim = int(self.encoding.n_output_dims)
        if self._out_dim != n_levels * n_features:
            raise RuntimeError(
                f"Unexpected tiny-cuda-nn output dim {self._out_dim}; "
                f"expected {n_levels * n_features}."
            )

        self.register_buffer("level_weights", torch.zeros(n_levels, dtype=torch.float32))
        self.register_buffer("feature_mask", torch.zeros(self._out_dim, dtype=torch.float32))
        self.update_step(0)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @torch.no_grad()
    def update_step(self, step: int) -> None:

        weights = torch.zeros_like(self.level_weights)
        if step < self.warmup_start:
            weights[: self.start_level] = 1.0
        else:
            t = (step - self.warmup_start) / max(1, self.warmup_end - self.warmup_start)
            t = float(max(0.0, min(1.0, t)))
            active_float = self.start_level + t * (self.n_levels - self.start_level)
            for lvl in range(self.n_levels):
                raw_w = max(0.0, min(1.0, active_float - lvl))
                weights[lvl] = raw_w * raw_w * (3.0 - 2.0 * raw_w)

        self.level_weights.copy_(weights)
        self.feature_mask.copy_(weights.repeat_interleave(self.n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x01 = ((x + 1.0) * 0.5).clamp(0.0, 1.0 - 1e-6).contiguous()
        y = self.encoding(x01.float())
        return y * self.feature_mask.to(device=y.device, dtype=y.dtype)
    
class SmallDirEnc(nn.Module):
    """Same direction encoding as your current code."""

    def __init__(self, num_freqs: int = 4) -> None:
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs, dtype=torch.float32))

    @property
    def out_dim(self) -> int:
        return 3 + 3 * 2 * len(self.freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for f in self.freqs:
            out += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(out, dim=-1)
