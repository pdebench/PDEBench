
# Downloading PDEBench Datasets :earth_asia:

Here we enumerate the list of all available PDEs in PDEBench and the commands to download them.

| PDEs        | Dataset Download                                             | Dataset Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| advection   | ```python download_direct.py --root_folder $proj_home/data --pde_name advection``` | 47 GB        |
| burgers     | ```python download_direct.py --root_folder $proj_home/data --pde_name burgers``` | 93 GB        |
| 1d_cfd      | ```python download_direct.py --root_folder $proj_home/data --pde_name 1d_cfd``` | 88 GB        |
| diff_sorp   | ```python download_direct.py --root_folder $proj_home/data --pde_name diff_sorp``` | 4 GB         |
| 1d_reacdiff | ```python download_direct.py --root_folder $proj_home/data --pde_name 1d_reacdiff``` | 62 GB        |
| 2d_reacdiff | ```python download_direct.py --root_folder $proj_home/data --pde_name 2d_reacdiff``` | 13 GB        |
| 2d_cfd      | ```python download_direct.py --root_folder $proj_home/data --pde_name 2d_cfd``` | 551 GB       |
| 3d_cfd      | ```python download_direct.py --root_folder $proj_home/data --pde_name 3d_cfd``` | 285 GB       |
| darcy       | ```python download_direct.py --root_folder $proj_home/data --pde_name darcy``` | 6.2 GB       |
| ns_incom    | ```python download_direct.py --root_folder $proj_home/data --pde_name ns_incom``` | 2.3 TB       |
| swe         | ```python download_direct.py --root_folder $proj_home/data --pde_name swe``` | 6.2 GB       |

--------

# Visualizing PDEs :ocean:

Below are some illustrations for how to visualize a certain PDE. It is assumed that you first download the data shard you'd like to visualize for a desired PDE. Then you can use the `visualize_pde.py` script to generate an animation (i.e., `.gif`).

###### 1D Diffusion Sorption Eqn

```
# get data: 1D_diff-sorp_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133020

# visualize
python visualize_pdes.py --pde_name "diff_sorp" --data_path "./"
```

----------

###### 1D Diffusion Reaction Eqn

```
# get data: ReacDiff_Nu1.0_Rho1.0.hdf5
https://darus.uni-stuttgart.de/api/access/datafile/133181

# visualize
python visualize_pdes.py --pde_name "1d_reacdiff"
```

----------

###### 1D Advection Eqn

```
# get data: 1D_Advection_Sols_beta0.4.hdf5
https://darus.uni-stuttgart.de/api/access/datafile/133110

# visualize
python visualize_pdes.py --pde_name "advection"
```

-----------

###### 1D Burgers Eqn

```
# get data: 1D_Burgers_Sols_Nu0.01.hdf5
https://darus.uni-stuttgart.de/api/access/datafile/133136

# visualize
python visualize_pdes.py --pde_name "burgers"
```

--------------------

###### 1D CFD Eqn

```
# get data: 1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5
https://darus.uni-stuttgart.de/api/access/datafile/135485

# visualize
python visualize_pdes.py --pde_name "1d_cfd"
```

-------------

###### 2D Diffusion Reaction Eqn

```
# get data: 2D_diff-react_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133017

# visualize
python visualize_pdes.py --pde_name "2d_reacdiff"
```

-------------

###### 2D Darcy Flow Eqn

```
# get data: 2D_DarcyFlow_beta1.0_Train.hdf5
https://darus.uni-stuttgart.de/api/access/datafile/133219

# visualize
python visualize_pdes.py --pde_name "darcy"
```

------------------

###### 2D Shallow Water Eqn

```
# get data: 2D_rdb_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133021

# visualize
python visualize_pdes.py --pde_name "swe" --data_path "./"
```

