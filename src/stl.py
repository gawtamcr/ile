import torch

def smooth_temporal_window(t, a, b, k=5.0):
    """C-infinity differentiable temporal window."""
    rise = torch.sigmoid(k * (t - a))
    fall = torch.sigmoid(k * (b - t))
    return rise * fall

class STLNode:
    def compute_loss(self, z, t):
        """Returns the attractive potential to minimize (L)."""
        return torch.tensor(0.0)
        
    def compute_h(self, z, t):
        """Returns the Control Barrier Function h(z, t). Safety requires h >= 0."""
        return torch.tensor(1.0)

class Reach(STLNode):
    def __init__(self, target, margin=0.0):
        self.target = torch.tensor(target, dtype=torch.float32)
        self.margin = margin
        
    def compute_loss(self, z, t):
        return torch.norm(z - self.target)**2
        
    def compute_h(self, z, t):
        # Safe region is strictly inside the target margin
        return self.margin - torch.norm(z - self.target)**2

class Not(STLNode):
    def __init__(self, child):
        self.child = child
        
    def compute_loss(self, z, t):
        return -self.child.compute_loss(z, t)
        
    def compute_h(self, z, t):
        # Negates the safety constraint: inside becomes outside
        return -self.child.compute_h(z, t)

class Always(STLNode):
    def __init__(self, child, interval, k=5.0):
        self.child = child
        self.a, self.b = interval
        self.k = k
        
    def compute_loss(self, z, t):
        # Always doesn't drive a target loss, it enforces a barrier
        return torch.tensor(0.0)
        
    def compute_h(self, z, t):
        h_val = self.child.compute_h(z, t)
        window = smooth_temporal_window(t, self.a, self.b, k=self.k)
        # Smoothly relax the barrier when outside the active time window
        return h_val / (window + 1e-6) - (1.0 - window) * 1e4

def Avoid(target, margin=1.0, interval=None, k=5.0):
    """Avoid B is logically equivalent to Always(Not(Reach(B)))."""
    avoid_node = Not(Reach(target, margin=margin))
    if interval:
        return Always(avoid_node, interval, k=k)
    return avoid_node

class Eventually(STLNode):
    def __init__(self, child, interval, k=5.0):
        self.child = child
        self.a, self.b = interval
        self.k = k
        
    def compute_loss(self, z, t):
        window = smooth_temporal_window(t, self.a, self.b, k=self.k)
        return window * self.child.compute_loss(z, t)
        
    def compute_h(self, z, t):
        return torch.tensor(1.0)

class And(STLNode):
    def __init__(self, left, right, beta=10.0):
        self.left = left
        self.right = right
        self.beta = beta
        
    def compute_loss(self, z, t):
        return (1.0 / self.beta) * torch.logaddexp(self.beta * self.left.compute_loss(z, t), self.beta * self.right.compute_loss(z, t))
        
    def compute_h(self, z, t):
        # Both h1 >= 0 and h2 >= 0 must hold -> use smooth_min for CBF intersection
        return -(1.0 / self.beta) * torch.logaddexp(-self.beta * self.left.compute_h(z, t), -self.beta * self.right.compute_h(z, t))