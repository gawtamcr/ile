import torch
import yaml
from flow import DiffeomorphicFlow
from stl import Reach, Avoid, Eventually, And
from planner import generate_trajectory
from viz import plot_full_analysis

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])
    
    # 1. Initialize Flow Model
    phi = DiffeomorphicFlow(
        layers=config["flow"]["layers"],
        hidden_dim=config["flow"]["hidden_dim"]
    )

    # Start and Target Configurations
    start_state = config["scenario"]["start_state"]
    target_A = config["scenario"]["target_A"]
    target_B = config["scenario"]["target_B"]

    # 2. Define STL Formula Programmatically: 
    formula = And(
        Eventually(Reach(target_A, margin=config["stl"]["reach_margin"]), interval=config["stl"]["reach_interval"], k=config["stl"]["temporal_k"]),
        Avoid(target_B, margin=config["stl"]["avoid_margin"], interval=config["stl"]["avoid_interval"], k=config["stl"]["temporal_k"]),
        beta=config["stl"]["and_beta"]
    )
    
    # 3. Plan Trajectory using literal simulation time
    z_traj, x_traj, time_steps = generate_trajectory(
        flow_model=phi, 
        x_start=start_state, 
        formula=formula, 
        t_span=tuple(config["planner"]["t_span"]), 
        steps=config["planner"]["steps"],
        alpha_lr=config["planner"]["alpha_lr"],
        cbf_gamma=config["planner"]["cbf_gamma"]
    )
    
    print(f"Time span tracked: {time_steps[0]:.1f} to {time_steps[-1]:.1f}")
    print(f"Final Latent State: {z_traj[-1]}")
    print(f"Final Physical State: {x_traj[-1]}")
    
    # 4. Visualize the diffeomorphism, temporal convergence, and trajectories
    plot_full_analysis(
        phi, z_traj, x_traj, time_steps, start_state, target_A, target_B,
        avoid_margin=config["stl"]["avoid_margin"],
        figsize=tuple(config["viz"]["figsize"]),
        grid_n=config["viz"]["grid_n"],
        grid_limits=tuple(config["viz"]["grid_limits"])
    )
