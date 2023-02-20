<hr/>

#### Reproducing results on [PDEBench datasets](https://github.com/pdebench/PDEBench)

<hr/>

| Dimension | PDE                    | ORCA <br />(nRMSE) | Reproduced <br />(nRMSE) | Training Time <br />(RTX 4K) |
| --------- | ---------------------- | ------------------ | ------------------------ | ---------------------------- |
| 1D        | Advection              | 9.8E-3             | 1.2E-2                   | 18.5h                        |
| 1D        | Burgers'               | 1.2E-2             |                          |                              |
| 1D        | Diffusion-Reaction     | 3.0E-3             |                          |                              |
| 1D        | Diffusion-Sorption     | 1.6E-3             | 1.8E-3                   |                              |
| 1D        | Navier-Stokes (CFD)    | 6.2E-2             | 4.4E-2                   |                              |
|           |                        |                    |                          |                              |
| 2D        | Darcy Flow             | 8.1E-2             | 7.8E-2                   | 8.75h                        |
| 2D        | Diffusion-Reaction     | 8.2E-1             | 8.2E-1                   |                              |
| 2D        | Navier-Stokes (CFD)    | -                  | -                        |                              |
| 2D        | Shallow Water Equation | 6.0E-3             | 5.8E-3                   | 2h                           |



### Citations

<hr/>

PDEBench Dataset:

```
@data{darus-2986_2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
publisher = {DaRUS},
title = {{PDEBench Datasets}},
year = {2022},
doi = {10.18419/darus-2986},
url = {https://doi.org/10.18419/darus-2986}
}
```
