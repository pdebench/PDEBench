## Package Requirements

We provide an `mml` file to recreate the mamba environment. To do so, do:
```shell
$ mamba env create --file orca_pdebench.mml
```



## Benchmarking on [PDEBench datasets](https://github.com/pdebench/PDEBench)

#### Downloading data and pretrained weights

1. Download the precomputed language features [text_xs.py](https://www.dropbox.com/s/yhlf25n8rzmdrtp/text_xs.npy?dl=0) and [text_ys.py](https://www.dropbox.com/s/16lj1vprg1pzckt/text_ys.npy?dl=0) (if you are using [RoBERTa models](https://huggingface.co/docs/transformers/model_doc/roberta)) to this directory.
1. Download [PDEBench datasets](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) to this directory.

```
# advection (1D)
# get data: 1D_Advection_Sols_beta0.4.hdf5
$ wget -c https://darus.uni-stuttgart.de/api/access/datafile/133110
$ mv 133110 1D_Advection_Sols_beta0.4.hdf5

# burgers' (1D)
# get data: 1D_Burgers_Sols_Nu1.0.hdf5
$ wget -c https://darus.uni-stuttgart.de/api/access/datafile/133142
$ mv 133142 1D_Burgers_Sols_Nu1.0.hdf5

# diffusion-reaction (1D)
# get data: ReacDiff_Nu0.5_Rho1.0.hdf5
$ wget -c https://darus.uni-stuttgart.de/api/access/datafile/133177
$ mv 133177 ReacDiff_Nu0.5_Rho1.0.hdf5

# diffusion-sorption (1D)
# get data: 1D_diff-sorp_NA_NA.h5
$ wget -c https://darus.uni-stuttgart.de/api/access/datafile/133020
$ mv 133020 1D_diff-sorp_NA_NA.h5

# Navier Stokes (1DCFD)
# get data: 1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5
$ wget -c https://darus.uni-stuttgart.de/api/access/datafile/164668
$ mv 164668 1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5


# Darcy Flow (2D)
# get data: 2D_DarcyFlow_beta0.1_Train.hdf5
$ wget -c https://darus.uni-stuttgart.de/api/access/datafile/133218
$ mv 133218 2D_DarcyFlow_beta0.1_Train.hdf5

# Diffusion-Reaction (2D)
# get data: 2D_diff-react_NA_NA.h5
$ wget -c https://darus.uni-stuttgart.de/api/access/datafile/133017
$ mv 133017 2D_diff-react_NA_NA.h5

# Navier Stokes (2DCFD)
# get data: 2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5
$ wget -c https://darus.uni-stuttgart.de/api/access/datafile/164691
$ mv 164691 2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5

# Shallow Water Eqn (2D)
# get data: 2D_rdb_NA_NA.h5
$ wget -c https://darus.uni-stuttgart.de/api/access/datafile/133021
$ mv 133021 2D_rdb_NA_NA.h5

```



<hr/>

#### Running experiments on [PDEBench datasets](https://github.com/pdebench/PDEBench)

<hr/>

| Dimension | PDE                    | Command                                         |
| --------- | ---------------------- | ----------------------------------------------- |
| 1D        | Advection              | `python main.py --config configs/PDEADV.yaml`   |
| 1D        | Burgers'               | `python main.py --config configs/PDEBG.yaml`    |
| 1D        | Diffusion-Reaction     | `python main.py --config configs/PDERD.yaml`    |
| 1D        | Diffusion-Sorption     | `python main.py --config configs/PDEDS.yaml`    |
| 1D        | Navier-Stokes (CFD)    | `python main.py --config configs/PDE1DCFD.yaml` |
|           |                        |                                                 |
| 2D        | Darcy Flow             | `python main.py --config configs/PDEDC.yaml`    |
| 2D        | Diffusion-Reaction     | `python main.py --config configs/PDERD2D.yaml`  |
| 2D        | Navier-Stokes (CFD)    | `python main.py --config configs/PDE2DCFD.yaml` |
| 2D        | Shallow Water Equation | `python main.py --config configs/PDESW.yaml`    |



### Citations

<hr/>

PDEBench Dataset:

```bibtex
@data{darus-2986_2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
publisher = {DaRUS},
title = {{PDEBench Datasets}},
year = {2022},
doi = {10.18419/darus-2986},
url = {https://doi.org/10.18419/darus-2986}
}
```
