import torch

def smooth_temporal_window(t, a, b, k=5.0):
    """
    Computes a C-infinity differentiable temporal window for interval [a, b].
    """
    # Using PyTorch's built-in sigmoid for numerical stability
    rise = torch.sigmoid(k * (t - a))
    fall = torch.sigmoid(k * (b - t))
    return rise * fall

def generate_temporal_logical_trajectory(flow_model, x_start, z_target, a, b, steps=150, alpha_lr=0.05):
    """
    Executes the bounded 'Eventually' operator: F_[a, b] target
    """
    flow_model.eval()
    
    x_current = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0)
    z_current = flow_model(x_current).detach()
    z_target = torch.tensor(z_target, dtype=torch.float32).unsqueeze(0)
    
    latent_trajectory = []
    physical_trajectory = []
    
    # Simulate a normalized time horizon t in [0, 1]
    time_steps = torch.linspace(0.0, 1.0, steps)

    for i in range(steps):
        t_current = time_steps[i]
        z_current.requires_grad_(True)
        
        # 1. Compute the static spatial potential
        spatial_loss = torch.norm(z_current - z_target)**2
        
        # 2. Modulate with the temporal window
        temporal_weight = smooth_temporal_window(t_current, a, b)
        time_varying_loss = temporal_weight * spatial_loss
        
        # 3. Compute gradient of the time-varying field
        grad_z = torch.autograd.grad(time_varying_loss, z_current)[0]
        
        with torch.no_grad():
            # Trajectory only updates if the temporal gradient is non-zero
            z_current = z_current - alpha_lr * grad_z
            x_next = flow_model.inverse(z_current)
            
            latent_trajectory.append(z_current.squeeze().numpy())
            physical_trajectory.append(x_next.squeeze().numpy())
            
    return latent_trajectory, physical_trajectory

# Example Execution: Target must be reached between t=0.6 and t=0.8
# The trajectory will remain relatively stationary until t approaches 0.6
# target_A and start_state defined previously
# z_traj_time, x_traj_time = generate_temporal_logical_trajectory(phi, start_state, target_A, a=0.6, b=0.8)