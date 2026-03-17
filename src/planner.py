import torch
import numpy as np

def generate_trajectory(flow_model, x_start, formula, t_span, steps=150, alpha_lr=0.05, cbf_gamma=0.1):
    """
    Generates a trajectory via latent gradient flow.
    Enforces 'Eventually' via loss gradients and 'Always' via CBF penalties.
    """
    flow_model.eval()
    
    x_current = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0)
    z_current = flow_model(x_current).detach()
    
    latent_trajectory = []
    physical_trajectory = []
    
    # 3. Time vector literal to the input
    t_start, t_end = t_span
    time_steps = torch.linspace(t_start, t_end, steps)

    for i in range(steps):
        t_current = time_steps[i]
        z_current.requires_grad_(True)
        
        # Compute reaching loss L
        L = formula.compute_loss(z_current, t_current)
        
        # Compute CBF constraint h(z) >= 0. Softplus ensures C^2 differentiability.
        h = formula.compute_h(z_current, t_current)
        barrier_penalty = torch.nn.functional.softplus(-cbf_gamma * h)
        
        total_loss = L + barrier_penalty
        grad_z = torch.autograd.grad(total_loss, z_current)[0]
        
        with torch.no_grad():
            z_current = z_current - alpha_lr * grad_z
            x_next = flow_model.inverse(z_current)
            
            latent_trajectory.append(z_current.squeeze().numpy())
            physical_trajectory.append(x_next.squeeze().numpy())
            
    return np.array(latent_trajectory), np.array(physical_trajectory), time_steps.numpy()