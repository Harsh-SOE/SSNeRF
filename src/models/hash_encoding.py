import torch
import math
import torch.nn as nn

class HashEncoding(nn.Module):
    def __init__(self, n_levels=16, n_features=4, log2_table=19,
                 base_res=16, max_res=1024):
        super().__init__()
        self.n_levels   = n_levels
        self.n_features = n_features
        self.table_size = 2 ** log2_table
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.table_size, n_features) for _ in range(n_levels)
        ])
        b = math.exp(math.log(max_res/base_res)/(n_levels-1))
        self.resolutions = [int(base_res*(b**i)) for i in range(n_levels)]
        # Precompute all 8 corner offsets — avoids creating tensors in loop
        corners = torch.tensor(
            [[dx,dy,dz] for dx in [0,1] for dy in [0,1] for dz in [0,1]],
            dtype=torch.long)
        self.register_buffer('corners', corners)
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)

    @property
    def out_dim(self): return self.n_levels * self.n_features

    def _hash(self, coords):
        primes = [1, 2654435761, 805459861]
        h = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        for i in range(3): h = h ^ (coords[:,i] * primes[i])
        return h.abs() % self.table_size

    def forward(self, x):
        x = x.clamp(-4.0, 4.0)  
        out = []
        for lvl, res in enumerate(self.resolutions):
            scaled = x * res
            xi     = torch.floor(scaled).long()          
            frac   = (scaled - xi.float()).clamp(0., 1.) 

            c  = self.corners                             
            wx = torch.stack([1.-frac[:,0], frac[:,0]], 1)
            wy = torch.stack([1.-frac[:,1], frac[:,1]], 1)
            wz = torch.stack([1.-frac[:,2], frac[:,2]], 1)
            w  = wx[:,c[:,0]] * wy[:,c[:,1]] * wz[:,c[:,2]]

            cc   = (xi.unsqueeze(1) + c.unsqueeze(0)).reshape(-1, 3)
            idx  = self._hash(cc)                                    
            feat = self.embeddings[lvl](idx).reshape(x.shape[0], 8, self.n_features)  
            out.append((w.unsqueeze(-1) * feat).sum(1))                 
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