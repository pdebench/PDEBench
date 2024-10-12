## Data Generation

#### Data generation for DarcyFlow Equation:

- Run the shell script:

```bash
bash data_gen/data_gen_NLE/ReactionDiffusionEq/run_DarcyFlow2D.sh
```

which will in turn run the python script
`data_gen/data_gen_NLE/ReactionDiffusionEq/reaction_diffusion_2D_multi_soluion_Hydra.py`

- Update `data_gen/data_gen_NLE/config/config.yaml` to:

```yaml
type: "ReacDiff" # 'advection'/'ReacDiff'/'burgers'/'CFD'
dim: 2
```

- Finally, run the data merge script:

```bash
python data_gen/data_gen_NLE/Data_Merge.py
```

---

#### Data generation for 1D Advection Equation:

```
# generate data and save as .npy array
cd PDEBench/pdebench/data_gen/data_gen_NLE/AdvectionEq

# Either generate a single file
CUDA_VISIBLE_DEVICES='2,3' python3 advection_multi_solution_Hydra.py +multi=beta1e0.yaml

# Or generate all files
bash run_trainset.sh
```

- Update `data_gen/data_gen_NLE/config/config.yaml` to:

```yaml
type: "advection" # 'advection'/'ReacDiff'/'burgers'/'CFD'
dim: 1
savedir: "./save/advection"
```

```
# serialize to hdf5 by transforming npy file
cd ..
python Data_Merge.py
```

---

#### Data generation for 1D Burgers' Equation:

```
# generate data and save as .npy array
cd PDEBench/pdebench/data_gen/data_gen_NLE/BurgersEq/

# Either generate a single file
CUDA_VISIBLE_DEVICES='0,2' python3 burgers_multi_solution_Hydra.py +multi=1e-1.yaml

# Or generate all files
bash run_trainset.sh
```

- Update `data_gen/data_gen_NLE/config/config.yaml` to:

```yaml
type: "burgers" # 'advection'/'ReacDiff'/'burgers'/'CFD'
dim: 1
savedir: "./save/burgers"
```

```
# serialize to hdf5 by transforming npy file
cd ..
python Data_Merge.py
```

---

#### Data generation for 1D Reaction Diffusion Equation:

```
# generate data and save as .npy array
cd PDEBench/pdebench/data_gen/data_gen_NLE/ReactionDiffusionEq/

# Either generate a single file
CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho2e0_Nu5e0.yaml

# Or generate all files
bash run_trainset.sh
```

- Update `data_gen/data_gen_NLE/config/config.yaml` to:

```yaml
type: "ReacDiff" # 'advection'/'ReacDiff'/'burgers'/'CFD'
dim: 1
savedir: "./save/ReacDiff"
```

```
# serialize to hdf5 by transforming npy file
cd ..
python Data_Merge.py
```
