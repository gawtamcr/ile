import torch

class STLNode:
    def compute_h(self, z, t):
        """Returns the topological barrier function h(z, t). Valid space is h >= 0."""
        raise NotImplementedError

class Avoid(STLNode):
    def __init__(self, target, margin=1.0):
        self.target = torch.tensor(target, dtype=torch.float32)
        self.margin = margin
        
    def compute_h(self, z, t):
        # Manifold is the exterior of the hypersphere
        return torch.norm(z - self.target, dim=-1)**2 - self.margin

class Eventually(STLNode):
    def __init__(self, target, t_end, initial_radius=20.0, final_radius=0.5):
        self.target = torch.tensor(target, dtype=torch.float32)
        self.t_end = t_end
        self.r_start = initial_radius
        self.r_end = final_radius
        
    def compute_h(self, z, t):
        # Smoothly contracting radius R(t) using a sigmoid decay
        # Ensures C^2 differentiability for dh/dt
        decay_factor = torch.sigmoid(10.0 * (t / self.t_end - 0.8))
        R_t = self.r_start * (1.0 - decay_factor) + self.r_end * decay_factor
        
        # Enforce spatial norm across the last dimension
        return R_t**2 - torch.norm(z - self.target, dim=-1)**2

class And(STLNode):
    def __init__(self, *children, beta=20.0):
        self.children = children
        self.beta = beta
        
    def compute_h(self, z, t):
        # Geometric Intersection via smooth minimum of barriers
        hs = torch.stack([child.compute_h(z, t) for child in self.children])
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * hs, dim=0)

class Or(STLNode):
    def __init__(self, *children, beta=20.0):
        self.children = children
        self.beta = beta
        
    def compute_h(self, z, t):
        # Geometric Union via smooth maximum of barriers
        hs = torch.stack([child.compute_h(z, t) for child in self.children])
        return (1.0 / self.beta) * torch.logsumexp(self.beta * hs, dim=0)