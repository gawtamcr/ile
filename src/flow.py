import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    def __init__(self, dim=2, hidden_dim=64, mask_idx=0):
        super().__init__()
        self.mask_idx     = mask_idx
        self.non_mask_idx = 1 - mask_idx

        # Softplus ensures C² differentiability required by latent gradient flow
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 2),  # outputs: scale s and translation t
        )

    def forward(self, x):
        x_masked     = x[:, self.mask_idx].unsqueeze(1)
        x_non_masked = x[:, self.non_mask_idx].unsqueeze(1)

        st = self.net(x_masked)
        s, t = st[:, 0:1], st[:, 1:2]

        y = x.clone()
        y[:, self.non_mask_idx] = (x_non_masked * torch.exp(s) + t).squeeze(1)
        return y

    def inverse(self, y):
        y_masked     = y[:, self.mask_idx].unsqueeze(1)
        y_non_masked = y[:, self.non_mask_idx].unsqueeze(1)

        st = self.net(y_masked)
        s, t = st[:, 0:1], st[:, 1:2]

        x = y.clone()
        x[:, self.non_mask_idx] = ((y_non_masked - t) * torch.exp(-s)).squeeze(1)
        return x

class DiffeomorphicFlow(nn.Module):
    def __init__(self, layers=4, hidden_dim=64):
        super().__init__()
        self.transforms = nn.ModuleList(
            [CouplingLayer(hidden_dim=hidden_dim, mask_idx=i % 2) for i in range(layers)]
        )

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse(self, z):
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z