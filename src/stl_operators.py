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


def generate_compositional_trajectory(
    flow_model, x_start, z_A, z_B, operator="OR", steps=150, alpha=0.05, beta=10.0
):
    """
    Implements compositional STL operators (OR / AND) via latent gradient flow.

    Atomic predicates:  u_A(z) = ||z - z_A||²,  u_B(z) = ||z - z_B||²

      OR  -> loss = smooth_min(u_A, u_B)  -- converge to the nearer target
      AND -> loss = smooth_max(u_A, u_B)  -- minimise the larger residual
    """
    flow_model.eval()

    x_current = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0)
    z_current = flow_model(x_current).detach()
    z_A       = torch.tensor(z_A, dtype=torch.float32).unsqueeze(0)
    z_B       = torch.tensor(z_B, dtype=torch.float32).unsqueeze(0)

    latent_trajectory   = [z_current.squeeze().numpy()]
    physical_trajectory = [x_current.squeeze().numpy()]

    for _ in range(steps):
        z_current.requires_grad_(True)

        u_A = torch.norm(z_current - z_A) ** 2
        u_B = torch.norm(z_current - z_B) ** 2

        if operator == "OR":
            loss = smooth_min(u_A, u_B, beta)
        elif operator == "AND":
            loss = smooth_max(u_A, u_B, beta)
        else:
            raise ValueError(f"Unknown operator '{operator}'. Expected 'OR' or 'AND'.")

        grad_z = torch.autograd.grad(loss, z_current)[0]

        with torch.no_grad():
            z_current = z_current - alpha * grad_z
            x_next    = flow_model.inverse(z_current)

            latent_trajectory.append(z_current.squeeze().numpy())
            physical_trajectory.append(x_next.squeeze().numpy())

    return latent_trajectory, physical_trajectory


if __name__ == "__main__":
    from main import DiffeomorphicFlow

    torch.manual_seed(42)
    phi = DiffeomorphicFlow(layers=4)

    target_A    = [ 2.0,  2.0]
    target_B    = [-2.0, -2.0]
    start_state = [-4.0,  4.0]

    z_traj_or, x_traj_or = generate_compositional_trajectory(
        phi, start_state, target_A, target_B, operator="OR"
    )
    print(f"OR  -- final latent: {z_traj_or[-1]},  physical: {x_traj_or[-1]}")

    z_traj_and, x_traj_and = generate_compositional_trajectory(
        phi, start_state, target_A, target_B, operator="AND"
    )
    print(f"AND -- final latent: {z_traj_and[-1]},  physical: {x_traj_and[-1]}")
