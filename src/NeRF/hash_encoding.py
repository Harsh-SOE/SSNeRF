import torch
import math
import torch.nn as nn


class ProgressiveHashEncoding(nn.Module):
    def __init__(
        self,
        n_levels=16,
        n_features=2,
        log2_table=18,
        base_res=8,
        max_res=512,
        start_level=4,
        warmup_start=500,
        warmup_end=6000,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features = n_features
        self.table_size = 2 ** log2_table

        self.start_level = start_level
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end

        self.embeddings = nn.ModuleList([
            nn.Embedding(self.table_size, n_features)
            for _ in range(n_levels)
        ])

        b = math.exp(math.log(max_res / base_res) / (n_levels - 1))
        self.resolutions = [int(base_res * (b ** i)) for i in range(n_levels)]

        corners = torch.tensor(
            [[dx, dy, dz] for dx in [0, 1]
                         for dy in [0, 1]
                         for dz in [0, 1]],
            dtype=torch.long
        )
        self.register_buffer("corners", corners)

        self.register_buffer(
            "level_weights",
            torch.zeros(n_levels, dtype=torch.float32)
        )

        self.update_step(0)

        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)

    @property
    def out_dim(self):
        return self.n_levels * self.n_features

    @torch.no_grad()
    def update_step(self, step: int):
        weights = torch.zeros_like(self.level_weights)

        if step < self.warmup_start:
            weights[:self.start_level] = 1.0
        else:
            t = (step - self.warmup_start) / max(1, self.warmup_end - self.warmup_start)
            t = float(max(0.0, min(1.0, t)))

            active_float = self.start_level + t * (self.n_levels - self.start_level)

            for lvl in range(self.n_levels):
                raw_w = active_float - lvl
                raw_w = max(0.0, min(1.0, raw_w))

                # Smoothstep
                weights[lvl] = raw_w * raw_w * (3.0 - 2.0 * raw_w)

        self.level_weights.copy_(weights)

    def _hash(self, coords):
        primes = [1, 2654435761, 805459861]

        h = torch.zeros(
            coords.shape[0],
            dtype=torch.long,
            device=coords.device
        )

        for i in range(3):
            h = h ^ (coords[:, i] * primes[i])

        return torch.remainder(h, self.table_size)

    def forward(self, x):
        """
        x expected in approximately [-1, 1].
        """

        # Convert [-1, 1] to [0, 1]
        x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0 - 1e-6)

        out = []

        for lvl, res in enumerate(self.resolutions):
            scaled = x * res

            xi = torch.floor(scaled).long()
            frac = (scaled - xi.float()).clamp(0.0, 1.0)

            c = self.corners

            wx = torch.stack([1.0 - frac[:, 0], frac[:, 0]], dim=1)
            wy = torch.stack([1.0 - frac[:, 1], frac[:, 1]], dim=1)
            wz = torch.stack([1.0 - frac[:, 2], frac[:, 2]], dim=1)

            w = wx[:, c[:, 0]] * wy[:, c[:, 1]] * wz[:, c[:, 2]]

            cc = (xi.unsqueeze(1) + c.unsqueeze(0)).reshape(-1, 3)
            idx = self._hash(cc)

            feat = self.embeddings[lvl](idx).reshape(
                x.shape[0],
                8,
                self.n_features
            )

            feat = (w.unsqueeze(-1) * feat).sum(dim=1)

            # Progressive level activation
            feat = feat * self.level_weights[lvl]

            out.append(feat)

        return torch.cat(out, dim=-1)
    
class SmallDirEnc(nn.Module):
    def __init__(self, num_freqs=4):
        super().__init__()
        self.register_buffer('freqs', 2.**torch.arange(num_freqs, dtype=torch.float32))
    @property
    def out_dim(self): return 3 + 3*2*len(self.freqs)
    def forward(self, x):
        out = [x]
        for f in self.freqs: out += [torch.sin(f*x), torch.cos(f*x)]
        return torch.cat(out, -1)