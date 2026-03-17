import torch
import numpy as np

def generate_guided_trajectory(flow_model, x_start, formula, t_span, steps=300, gamma=5.0):
    """
    Integrates a generative prior vector field, applying closed-form algebraic 
    projection to strictly satisfy the continuous CBF logic manifolds.
    """
    flow_model.eval()
    
    t_start, t_end = t_span
    time_steps = torch.linspace(t_start, t_end, steps)
    dt = (t_end - t_start) / steps
    
    x_current = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0)
    z_current = flow_model(x_current).detach()
    
    latent_trajectory = [z_current.squeeze().numpy()]
    physical_trajectory = [x_current.squeeze().numpy()]
    
    for i in range(steps - 1):
        # Tensors must track gradients for algebraic projection
        t_current = torch.tensor(time_steps[i], dtype=torch.float32, requires_grad=True)
        z_current.requires_grad_(True)
        
        # 1. Compile the global topology
        h_val = formula.compute_h(z_current, t_current).squeeze()
        
        # 2. Extract spatial and temporal boundaries
        grad_z, grad_t = torch.autograd.grad(
            outputs=h_val, 
            inputs=[z_current, t_current], 
            create_graph=False
        )
        # 3. Define the synthetic generative prior (Flowing towards latent origin)
        v_prior = -0.05 * z_current 
        
        # 4. Evaluate the strict Safety Condition: dh/dt + grad_z * v + gamma * h >= 0
        b_term = grad_t + gamma * h_val
        safety_margin = torch.sum(grad_z * v_prior) + b_term
        
        # 5. Closed-Form Algebraic Projection
        with torch.no_grad():
            if safety_margin < 0:
                # Prior violates logic: project orthogonally onto the safe half-space
                correction = -safety_margin * grad_z / (torch.norm(grad_z)**2 + 1e-6)
                v_guided = v_prior + correction
            else:
                # Prior naturally satisfies logic: remain unperturbed
                v_guided = v_prior
                
            # Execute Euler integration step
            z_current = z_current + v_guided * dt
            x_next = flow_model.inverse(z_current)
            
            latent_trajectory.append(z_current.squeeze().numpy())
            physical_trajectory.append(x_next.squeeze().numpy())
            
    return np.array(latent_trajectory), np.array(physical_trajectory), time_steps.numpy()