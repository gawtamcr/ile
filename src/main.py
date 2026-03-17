import torch
import yaml
from flow import DiffeomorphicFlow
from stl import Avoid, Eventually, And
from planner import generate_guided_trajectory
from viz import plot_full_analysis

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])
    
    # Initialize the structural diffeomorphism
    phi = DiffeomorphicFlow(
        layers=config["flow"]["layers"],
        hidden_dim=config["flow"]["hidden_dim"]
    )

    start_state = config["scenario"]["start_state"]
    reach_tgts = config["scenario"]["reach_targets"]
    avoid_tgts = config["scenario"]["avoid_targets"]
    t_span = tuple(config["planner"]["t_span"])

    # Map physical targets to latent coordinates to define the logic regions
    with torch.no_grad():
        z_A1 = phi(torch.tensor([reach_tgts["A1"]], dtype=torch.float32)).squeeze().numpy()
        z_A2 = phi(torch.tensor([reach_tgts["A2"]], dtype=torch.float32)).squeeze().numpy()
        z_B1 = phi(torch.tensor([avoid_tgts["B1"]], dtype=torch.float32)).squeeze().numpy()
        z_B2 = phi(torch.tensor([avoid_tgts["B2"]], dtype=torch.float32)).squeeze().numpy()

    # Compile the strict Geometric Syntax Tree
    # F(A1) AND F(A2) AND G(NOT B1) AND G(NOT B2)
    formula = And(
        Eventually(target=z_A1, t_end=config["stl"]["intervals"]["A1"][1]),
        Eventually(target=z_A2, t_end=config["stl"]["intervals"]["A2"][1]),
        Avoid(target=z_B1, margin=config["stl"]["avoid_margin"]),
        Avoid(target=z_B2, margin=config["stl"]["avoid_margin"]),
        beta=config["stl"]["and_beta"]
    )
    
    # Execute the Latent Compiler
    print("Compiling geometry and executing guided prior...")
    z_traj, x_traj, time_steps = generate_guided_trajectory(
        flow_model=phi, 
        x_start=start_state, 
        formula=formula, 
        t_span=t_span, 
        steps=config["planner"]["steps"],
        gamma=config["planner"]["cbf_gamma"]
    )
    
    print(f"Final Latent State: {z_traj[-1]}")
    
    # Generate the publication figure
    plot_full_analysis(
        phi=phi, 
        z_traj=z_traj, 
        x_traj=x_traj, 
        time_steps=time_steps, 
        start_state=start_state, 
        reach_targets=reach_tgts, 
        avoid_targets=avoid_tgts,
        formula=formula, # REQUIRED FOR CONTOURS
        avoid_margin=config["stl"]["avoid_margin"],
        figsize=tuple(config["viz"]["figsize"]),
        grid_n=config["viz"]["grid_n"],
        grid_limits=tuple(config["viz"]["grid_limits"])
    )