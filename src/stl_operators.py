import torch


def smooth_max(u_a, u_b, beta=10.0):
    """
    C² differentiable approximation of max(u_a, u_b) via log-sum-exp.

    Encodes logical AND: minimising smooth_max requires both predicates
    to be small simultaneously.

    Numerically stable via torch.logaddexp (avoids exp overflow).
    """
    return (1.0 / beta) * torch.logaddexp(beta * u_a, beta * u_b)


def smooth_min(u_a, u_b, beta=10.0):
    """
    C² differentiable approximation of min(u_a, u_b).

    Encodes logical OR: minimising smooth_min requires at least one
    predicate to be small.

    Numerically stable via torch.logaddexp (avoids exp overflow).
    """
    return -(1.0 / beta) * torch.logaddexp(-beta * u_a, -beta * u_b)


if __name__ == "__main__":
    from main import DiffeomorphicFlow, generate_trajectory

    torch.manual_seed(42)
    phi = DiffeomorphicFlow(layers=4)

    start_state = [-4.0,  4.0]
    z_A         = torch.tensor([[ 2.0,  2.0]], dtype=torch.float32)
    z_B         = torch.tensor([[-2.0, -2.0]], dtype=torch.float32)
    beta        = 10.0

    # Define loss functions for OR and AND semantics
    loss_or = lambda z: smooth_min(torch.norm(z - z_A) ** 2, torch.norm(z - z_B) ** 2, beta)
    loss_and = lambda z: smooth_max(torch.norm(z - z_A) ** 2, torch.norm(z - z_B) ** 2, beta)

    z_traj_or, x_traj_or = generate_trajectory(
        phi, start_state, loss_or, steps=150, alpha=0.05
    )
    print(f"OR  -- final latent: {z_traj_or[-1]},  physical: {x_traj_or[-1]}")

    z_traj_and, x_traj_and = generate_trajectory(
        phi, start_state, loss_and, steps=150, alpha=0.05
    )
    print(f"AND -- final latent: {z_traj_and[-1]},  physical: {x_traj_and[-1]}")
