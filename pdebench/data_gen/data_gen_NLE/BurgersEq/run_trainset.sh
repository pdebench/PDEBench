#!/bin/sh
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=1e0.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=1e-1.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=1e-2.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=1e-3.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=2e0.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=2e-1.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=2e-2.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=2e-3.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=4e0.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=4e-1.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=4e-2.yaml
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=4e-3.yaml
