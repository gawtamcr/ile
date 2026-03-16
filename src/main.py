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
    def __init__(self, layers=4):
        super().__init__()
        self.transforms = nn.ModuleList(
            [CouplingLayer(mask_idx=i % 2) for i in range(layers)]
        )

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse(self, z):
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z


def generate_trajectory(flow_model, x_start, loss_function, steps=100, alpha=0.05):
    """
    Generates a trajectory via latent gradient flow on a given loss function.

    Runs gradient descent on U(z) = loss_function(z) in latent space Z.
    Each iterate is pulled back to physical space X via the exact inverse φ⁻¹.
    """
    flow_model.eval()

    x_current = torch.tensor(x_start,  dtype=torch.float32).unsqueeze(0)
    z_current = flow_model(x_current).detach()

    latent_trajectory   = [z_current.squeeze().numpy()]
    physical_trajectory = [x_current.squeeze().numpy()]

    for _ in range(steps):
        z_current.requires_grad_(True)

        loss   = loss_function(z_current)
        grad_z = torch.autograd.grad(loss, z_current)[0]

        with torch.no_grad():
            z_current = z_current - alpha * grad_z
            x_next    = flow_model.inverse(z_current)

        latent_trajectory.append(z_current.squeeze().numpy())
        physical_trajectory.append(x_next.squeeze().numpy())

    return latent_trajectory, physical_trajectory


if __name__ == "__main__":
    torch.manual_seed(42)
    phi = DiffeomorphicFlow(layers=4)

    start_state = [-4.0, 4.0]
    z_target    = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    loss_fn     = lambda z: torch.norm(z - z_target) ** 2

    z_traj, x_traj = generate_trajectory(phi, start_state, loss_fn)
    print(f"Final latent state:   {z_traj[-1]}")
    print(f"Final physical state: {x_traj[-1]}")
