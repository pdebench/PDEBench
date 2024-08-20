#! /bin/bash
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho1e0_Nu1e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho1e0_Nu2e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho1e0_Nu5e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho2e0_Nu1e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho2e0_Nu2e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho2e0_Nu5e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho5e0_Nu1e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho5e0_Nu2e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho5e0_Nu5e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho1e1_Nu1e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho1e1_Nu2e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho1e1_Nu5e0.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho2e0_Nu5e-1.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho5e0_Nu5e-1.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho1e0_Nu5e-1.yaml
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho1e1_Nu5e-1.yaml
