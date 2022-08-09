# PDEBench

By : Makoto Takamoto `<makoto.takamoto@neclab.eu, takamtmk@gmail.com>`, Timothy Praditia `<timothy.praditia@iws.uni-stuttgart.de>`, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pflüger and Mathias Niepert

This repository contains the code for the paper:
PDEBench: An Extensive Benchmark for Scientific Machine Learning

In this work, we provide a diverse and comprehensive benchmark for scientific machine learning, including a variety of challenging and representative range of physical problems.
This repository consists of the codes used to generate the datasets, upload and download the datasets from the data repository, train and evaluate different machine learning models as baseline.
PDEBench features a much wider range of PDEs than existing approaches including realistic and difficult problems (both forward and inverse), larger ready-to-use datasets comprising various initial or boundary conditions and model parameters, and extensible source codes.

## Dataset

PDEBench Dataset
https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986

PDEBench Pre-Trained Models
https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987


DOIs 

[![DOI:10.18419/darus-2986](https://img.shields.io/badge/DOI-doi%3A10.18419%2Fdarus--2986-red)](https://doi.org/10.18419/darus-2986)
[![DOI:10.18419/darus-2987](https://img.shields.io/badge/DOI-doi%3A10.18419%2Fdarus--2987-red)](https://doi.org/10.18419/darus-2987)

## Installation

    pip install .

## Requirements

```bash
python3 -m venv ./venv --prompt pde_benchmark --system-site-packages 
. ./venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

The minimum required packages to train and run the baseline ML models are listed in [requirements.txt](./requirements.txt)
To run the data generation scripts, the complete package requirements are listed in [requirements_datagen.txt](./requirements_datagen.txt)

For GPU support there are additional platform-specific instructions:

For pytorch,

```bash
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

For jax, which is approximately 6 times faster for simulations than pytorch in our tests,

```bash
pip install "jax[cuda11_cudnn82]>=0.3.0" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Data Generation
The data generation codes are contained in [data_gen](./pdebench/data_gen):
- `gen_diff_react.py` to generate the 2D diffusion-reaction data.
- `gen_diff_sorp.py` to generate the 1D diffusion-sorption data.
- `gen_radial_dam_break.py` to generate the 2D shallow-water data.
- `gen_radial_dam_break.py` to generate the 2D shallow-water data.
- `gen_ns_incomp.py` to generate the 2D incompressible inhomogenous Navier-Stokes data.
- `plot.py` to plot the generated data.
- `uploader.py` to upload the generated data to the data repository.
- `.env` is the environment data to store Dataverse URL and API token to upload the generated data. Note that the filename should be strictly `.env` (i.e. remove the  `example` from the filename)
- `configs` directory contains the yaml files storing the configuration for the simulation. Arguments for the simulation are problem-specific and detailed explanation can be found in the simulation scripts.
- `src` directory contains the simulation scripts for different problems: `sim_diff_react-py` for 2D diffusion-reaction, `sim_diff_sorp.py` for 1D diffusion-sorption, and `swe` for the shallow-water equation.

### Data Generation for 1D Advection/Burgers/Reaction-Diffusion/2D DarcyFlow/Compressible Navier-Stokes Equations
The data generation codes are contained in [data_gen_NLE](./pdebench/data_gen/data_gen_NLE/):
- `utils.py` util file for data generation, mainly boundary conditions and initial conditions.
- `AdvectionEq` directory with the source codes to generate 1D Advection equation training samples
- `BurgersEq` directory with the source codes to generate 1D Burgers equation training samples
- `CompressibleFluid` directory with the source codes to generate compressible Navier-Stokes equations training samples
- `ReactionDiffusionEq` directory with the source codes to generate 1D Reaction-Diffusion equation training samples
- `save` directory saving the generated training samples

A typical example to generate training samples (1D Advection Equation):
(in `data_gen/data_gen_NLE/AdvectionEq/`)
```bash
python3 advection_multi_solution_Hydra.py +multi=beta1e0.yaml
```
which is assumed to be performed in each directory.

Examples for generating other PDEs are provided in `run_trainset.sh` in each PDE's directories.
The config files for Hydra are stored in `config` directory in each PDE's directory. 

#### Data Transformaion and Merge into HDF5 format
1D Advection/Burgers/Reaction-Diffusion/2D DarcyFlow/Compressible Navier-Stokes Equations save data as a numpy array. 
So, to read those data via our dataloaders, the data transformation/merge should be performed. 
This can be done using `data_gen_NLE/Data_Merge.py` whose config file is located at: `data_gen/data_gen_NLE/config/config.yaml`. 
After properly set parameters in the config file (type: name of PDEs, dim: number of spatial-dimension, bd: boundary condition), 
the corresponding HDF5 file could be obtained as: 
```bash
python3 Data_Merge.py
```


## Configuration

You can set the default values for data locations for this project by putting config vars like this in the `.env` file:

```
WORKING_DIR=~/Data/Working
ARCHIVE_DATA_DIR=~/Data/Archive
```

There is an example in `example.env`.


## Data Download
The data download codes are contained in [data_download](./pdebench/data_download):
- `download.py` to download the data.
- `.env` is the environment data to store Dataverse URL and API token to download the generated data. Note that the filename should be strictly `.env` (i.e. remove the  `example` from the filename)
- `config` directory contains the yaml files storing the configuration for the data downloader. The argument 'filename' should match the filename in the data repository.


## Baseline Models
In this work, we provide three different ML models to be trained and evaluated against the benchmark datasets, namely [FNO](https://arxiv.org/pdf/2010.08895.pdf), [U-Net](https://www.sciencedirect.com/science/article/abs/pii/S0010482519301520?via%3Dihub), and [PINN](https://www.sciencedirect.com/science/article/pii/S0021999118307125).
The codes for the baseline model implementations are contained in [models](./pdebench/models):
- `train_models_forward.py` is the main script to train and evaluate the model. It will call on model-specific script based on the input argument.
- `train_models_inverse.py` is the main script to train and evaluate the model for inverse problems. It will call on model-specific script based on the input argument.
- `metrics.py` is the script to evaluate the trained models based on various evaluation metrics described in our paper. Additionally, it also plots the prediction and target data.
- `analyse_result_forward.py` is the script to convert the saved pickle file from the metrics calculation script into pandas dataframe format and save it as a CSV file. Additionally it also plots a bar chart to compare the results between different models.
- `analyse_result_inverse.py` is the script to convert the saved pickle file from the metrics calculation script into pandas dataframe format and save it as a CSV file. This script is used for the inverse problems. Additionally it also plots a bar chart to compare the results between different models.
- `fno` contains the scripts of FNO implementation. These are partly adapted from the [FNO repository](https://github.com/zongyi-li/fourier_neural_operator).
- `unet` contains the scripts of U-Net implementation. These are partly adapted from the [U-Net repository](https://github.com/mateuszbuda/brain-segmentation-pytorch).
- `pinn` contains the scripts of PINN implementation. These utilize the [DeepXDE library](https://github.com/lululxvi/deepxde).
- `inverse` contains the model for inverse model based on gradient.
- `config` contains the yaml files for the model training input. The default templates for different equations are provided in the [args](./pdebench/models/config/args) directory. User just needs to copy and paste them to the args keyword in the [config.yaml](./pdebench/models/config/config.yaml) file.

An example to run the forward model training can be found in [run_forward_1D.sh](./pdebench/models/run_forward_1D.sh), and an example to run the inverse model training can be found in [run_inverse.sh](./pdebench/models/run_inverse.sh).


### Short explanations on the config args
- model_name: string, containing the baseline model name, either 'FNO', 'Unet', or 'PINN'.
- if_training: bool, set True for training, or False for evaluation.
- continue_training: bool, set True to continute training from a checkpoint.
- num_workers: int, number of workers for the PyTorch dataloader.
- batch_size: int, training batch size.
- initial_step: int, number of time steps used as input for FNO and U-Net.
- t_train: int, number of the last time step used for training (for extrapolation testing, set this to be < Nt).
- model_update: int, number of epochs to save model.
- filename: str, has to match the dataset filename.
- single_file: bool, set False for 2D diffusion-reaction, 1D diffusion-sorption, and the radial dam break scenarios, and set True otherwise.
- reduced_resolution: int, factor to downsample spatial resolution.
- reduced_resolution_t: int, factor to downsample temporal resolution.
- reduced_batch: int, factor to downsample sample size used for training.
- epochs: int, total epochs used for training.
- learning_rate: float, learning rate of the optimizer.
- scheduler_step: int, number of epochs to update the learning rate scheduler.
- scheduler_gamma: float, decay rate of the learning rate.

#### U-Net specific args:
- in_channels: int, number of input channels
- out_channels: int, number of output channels
- ar_mode: bool, set True for fully autoregressive or pushforward training.
- pushforward: bool, set True for pushforward training, False otherwise (ar_mode also has to be set True).
- unroll_step: int, number of time steps to backpropagate in the pushforward training.

#### FNO specific args:
- num_channels: int, number of channels (variables).
- modes: int, number of Fourier modes to multiply.
- width: int, number of channels for the Fourier layer.

#### INVERSE specific args:
 -  base_path: string, location of the data directory 
 -  training_type: string, type of training,  autoregressive, single
 -  mcmc_num_samples: int, number of generated samples
 -  mcmc_warmup_steps: 10
 -  mcmc_num_chains: 1
 -  num_samples_max: 1000
 -  in_channels_hid: 64
 -  inverse_model_type: striung, type of inverse inference model, ProbRasterLatent, InitialConditionInterp    
 -  inverse_epochs: int, number of epochs for the  gradint based method
 -  inverse_learning_rate: float, learning rate for the gradint based method
 -  inverse_verbose_flag: bool, some printing

#### Plotting specific args:
- plot: bool, set True to activate plotting.
- channel_plot: int, determines which channel/variable to plot.
- x_min: float, left spatial domain.
- x_max: float, right spatial domain.
- y_min: float, lower spatial domain.
- y_max: float, upper spatial domain.
- t_min: float, start of temporal domain.
- t_max: float, end of temporal domain.

## Datasets and pretrained models
We provide the benchmark datasets we used in the paper through our [data repository](https://darus.uni-stuttgart.de/privateurl.xhtml?token=1be27526-348a-40ed-9fd0-c62f588efc01).
The data generation configuration can be found in the paper.
Additionally, the pretrained models are also available to be downloaded [here](https://darus.uni-stuttgart.de/privateurl.xhtml?token=cd862f8c-8e1b-49d2-b4da-b35f8df5ac85). To use the pretrained models, users can specify the argument `continue_training: True` in the [config file](./pdebench/models/config/config.yaml).


## Citations

```
@online{PDEBenchDataset,
  author = {Makoto Takamoto and  Timothy Pradita and Raphael Leiteritz and Dan MacKinlay and Francesco Alesiani and Dirk Pflüger and Mathias Niepert},
  title = {{PDEBench}: A Diverse and Comprehensive Benchmark for Scientific Machine Learning},
  year = 2022,
  doi = {doi:10.18419/darus-2986},
  urldoi = {http://dx.doi.org/10.18419/darus-2986},
  url = {https://darus.uni-stuttgart.de/privateurl.xhtml?token=1be27526-348a-40ed-9fd0-c62f588efc01},
  urldate = {2022-06-06}
}


@data{darus-2986_2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
publisher = {DaRUS},
title = {{PDEBench Datasets}},
year = {2022},
doi = {10.18419/darus-2986},
url = {https://doi.org/10.18419/darus-2986}
}

@data{darus-2987_2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
publisher = {DaRUS},
title = {{PDEBench Pretrained Models}},
year = {2022},
doi = {10.18419/darus-2987},
url = {https://doi.org/10.18419/darus-2987}
}


@misc{li2020fourier,
  title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
  author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
  year={2020},
  eprint={2010.08895},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@article{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm},
  author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
  journal={Computers in Biology and Medicine},
  volume={109},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2019.05.002}
}

@article{raissi2019pinn,
title = {Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
journal = {Journal of Computational Physics},
volume = {378},
pages = {686-707},
year = {2019},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2018.10.045},
author = {M. Raissi and P. Perdikaris and G.E. Karniadakis}
}
```

## Code contributors

* [Makato Takamoto](https://github.com/mtakamoto-D)
* [Timothy Praditia](https://github.com/timothypraditia)
* [Raphael Leiteritz](https://github.com/leiterrl)
* [Francesco Alesiani](https://github.com/falesiani)
* [Dan MacKinlay](https://danmackinlay.name/)
* [John Kim](https://github.com/johnmjkim)
* [Gefei Shan](https://github.com/davecatmeow)
* [Yizhou Yang](https://github.com/verdantwynnd)
* [Ran Zhang](https://github.com/maphyca)
* [Simon Brown](https://github.com/SimonSyBrown)


## License 
MIT for solver code and baseline code, and NLE Academic License for selected code 
