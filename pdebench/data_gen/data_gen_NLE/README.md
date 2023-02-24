## Data Generation

#### Data generation for DarcyFlow Equation:

- Run the shell script:

```bash
bash data_gen/data_gen_NLE/ReactionDiffusionEq/run_DarcyFlow2D.sh
```

which will in turn run the python script `data_gen/data_gen_NLE/ReactionDiffusionEq/reaction_diffusion_2D_multi_soluion_Hydra.py`

- Update `data_gen/data_gen_NLE/config/config.yaml` to:

```yaml
type: 'ReacDiff'  # 'advection'/'ReacDiff'/'burgers'/'CFD'
dim: 2
```

- Finally, run the data merge script:

```bash
python data_gen/data_gen_NLE/Data_Merge.py
```

