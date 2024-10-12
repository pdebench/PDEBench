# PDEBench

The code repository for the NeurIPS 2022 paper
[PDEBench: An Extensive Benchmark for Scientific Machine Learning](https://arxiv.org/abs/2210.07182)

:tada:
[**SimTech Best Paper Award 2023**](https://www.simtech.uni-stuttgart.de/press/SimTech-Best-Paper-Award-2023-Benchmark-for-ML-for-scientific-simulations)
:confetti_ball:

PDEBench provides a diverse and comprehensive set of benchmarks for scientific
machine learning, including challenging and realistic physical problems. This
repository consists of the code used to generate the datasets, to upload and
download the datasets from the data repository, as well as to train and evaluate
different machine learning models as baselines. PDEBench features a much wider
range of PDEs than existing benchmarks and includes realistic and difficult
problems (both forward and inverse), larger ready-to-use datasets comprising
various initial and boundary conditions, and PDE parameters. Moreover, PDEBench
was created to make the source code extensible and we invite active
participation from the SciML community to improve and extend the benchmark.

![Visualizations of some PDE problems covered by the benchmark.](https://github.com/pdebench/PDEBench/blob/main/pdebench_examples.PNG)

Created and maintained by Makoto Takamoto
`<makoto.takamoto@neclab.eu, takamtmk@gmail.com>`, Timothy Praditia
`<timothy.praditia@iws.uni-stuttgart.de>`, Raphael Leiteritz, Dan MacKinlay,
Francesco Alesiani, Dirk Pflüger, and Mathias Niepert.

---

## Datasets and Pretrained Models

We also provide datasets and pretrained machine learning models.

PDEBench Datasets:
https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986

PDEBench Pre-Trained Models:
https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987

DOIs

[![DOI:10.18419/darus-2986](https://img.shields.io/badge/DOI-doi%3A10.18419%2Fdarus--2986-red)](https://doi.org/10.18419/darus-2986)
[![DOI:10.18419/darus-2987](https://img.shields.io/badge/DOI-doi%3A10.18419%2Fdarus--2987-red)](https://doi.org/10.18419/darus-2987)

## Installation

### Using pip

Locally:

```bash
pip install --upgrade pip wheel
pip install .
```

From PyPI:

```bash
pip install pdebench
```

To include dependencies for data generation:

```bash
pip install "pdebench[datagen310]"
pip install ".[datagen310]" # locally
```

or

```bash
pip install "pdebench[datagen39]"
pip install ".[datagen39]" # locally
```

### GPU Support

For GPU support there are additional platform-specific instructions:

For PyTorch, the latest version we support is v1.13.1
[see previous-versions/#linux - CUDA 11.7](https://pytorch.org/get-started/previous-versions/#linux-and-windows-2).

For JAX, which is approximately 6 times faster for simulations than PyTorch in
our tests,
[see jax#pip-installation-gpu-cuda-installed-via-pip](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier)

## Installation using conda:

If you like you can also install dependencies using anaconda, we suggest to use
[mambaforge](https://github.com/conda-forge/miniforge#mambaforge) as a
distribution. Otherwise you may have to **enable the conda-forge** channel for
the following commands.

Starting from a fresh environment:

```
conda create -n myenv python=3.9
conda activate myenv
```

Install dependencies for model training:

```
conda install deepxde hydra-core h5py -c conda-forge
```

According to your hardware availability, either install PyTorch with CUDA
support:

- [see previous-versions/#linux - CUDA 11.7](https://pytorch.org/get-started/previous-versions/#linux-and-windows-2).

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- [or CPU only binaries](https://pytorch.org/get-started/previous-versions/#linux-and-windows-2).

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```

Optional dependencies for data generation:

```
conda install clawpack jax jaxlib python-dotenv
```

## Configuring DeepXDE

In our tests we used PyTorch as backend for DeepXDE. Please
[follow the documentation](https://deepxde.readthedocs.io/en/latest/user/installation.html#working-with-different-backends)
to enable this.

## Data Generation

The data generation codes are contained in [data_gen](./pdebench/data_gen):

- `gen_diff_react.py` to generate the 2D diffusion-reaction data.
- `gen_diff_sorp.py` to generate the 1D diffusion-sorption data.
- `gen_radial_dam_break.py` to generate the 2D shallow-water data.
- `gen_ns_incomp.py` to generate the 2D incompressible inhomogeneous
  Navier-Stokes data.
- `plot.py` to plot the generated data.
- `uploader.py` to upload the generated data to the data repository.
- `.env` is the environment data to store Dataverse URL and API token to upload
  the generated data. Note that the filename should be strictly `.env` (i.e.
  remove the `example` from the filename)
- `configs` directory contains the yaml files storing the configuration for the
  simulation. Arguments for the simulation are problem-specific and detailed
  explanation can be found in the simulation scripts.
- `src` directory contains the simulation scripts for different problems:
  `sim_diff_react-py` for 2D diffusion-reaction, `sim_diff_sorp.py` for 1D
  diffusion-sorption, and `swe` for the shallow-water equation.

### Data Generation for 1D Advection/Burgers/Reaction-Diffusion/2D DarcyFlow/Compressible Navier-Stokes Equations

The data generation codes are contained in
[data_gen_NLE](./pdebench/data_gen/data_gen_NLE/):

- `utils.py` util file for data generation, mainly boundary conditions and
  initial conditions.
- `AdvectionEq` directory with the source codes to generate 1D Advection
  equation training samples
- `BurgersEq` directory with the source codes to generate 1D Burgers equation
  training samples
- `CompressibleFluid` directory with the source codes to generate compressible
  Navier-Stokes equations training samples

  - `ReactionDiffusionEq` directory with the source codes to generate 1D
    Reaction-Diffusion equation training samples (**Note:
    [DarcyFlow data can be generated by run_DarcyFlow2D.sh](pdebench/data_gen/data_gen_NLE/README.md)
    in this folder.**)

- `save` directory saving the generated training samples

A typical example to generate training samples (1D Advection Equation): (in
`data_gen/data_gen_NLE/AdvectionEq/`)

```bash
python3 advection_multi_solution_Hydra.py +multi=beta1e0.yaml
```

which is assumed to be performed in each directory.

Examples for generating other PDEs are provided in `run_trainset.sh` in each
PDE's directories. The config files for Hydra are stored in `config` directory
in each PDE's directory.

#### Data Transformaion and Merge into HDF5 format

1D Advection/Burgers/Reaction-Diffusion/2D DarcyFlow/Compressible Navier-Stokes
Equations save data as a numpy array. So, to read those data via our
dataloaders, the data transformation/merge should be performed. This can be done
using `data_gen_NLE/Data_Merge.py` whose config file is located at:
`data_gen/data_gen_NLE/config/config.yaml`. After properly setting the
parameters in the config file (type: name of PDEs, dim: number of
spatial-dimension, bd: boundary condition), the corresponding HDF5 file could be
obtained as:

```bash
python3 Data_Merge.py
```

## Configuration

You can set the default values for data locations for this project by putting
config vars like this in the `.env` file:

```
WORKING_DIR=~/Data/Working
ARCHIVE_DATA_DIR=~/Data/Archive
```

There is an example in `example.env`.

## Data Download

The download scripts are provided in [data_download](./pdebench/data_download).
There are two options to download data.

1. Using `download_direct.py` (**recommended**)
   - Retrieves data shards directly using URLs. Sample command for each PDE is
     given in the README file in the [data_download](./pdebench/data_download)
     directory.
2. Using `download_easydataverse.py` (might be slow and you could encounter
   errors/issues; hence, not recommended!)
   - Use the config files from the `config` directory that contains the yaml
     files storing the configuration. Any files in the dataset matching
     `args.filename` will be downloaded into `args.data_folder`.

## Baseline Models

In this work, we provide three different ML models to be trained and evaluated
against the benchmark datasets, namely
[FNO](https://arxiv.org/pdf/2010.08895.pdf),
[U-Net](https://www.sciencedirect.com/science/article/abs/pii/S0010482519301520?via%3Dihub),
and [PINN](https://www.sciencedirect.com/science/article/pii/S0021999118307125).
The codes for the baseline model implementations are contained in
[models](./pdebench/models):

- `train_models_forward.py` is the main script to train and evaluate the model.
  It will call on model-specific script based on the input argument.
- `train_models_inverse.py` is the main script to train and evaluate the model
  for inverse problems. It will call on model-specific script based on the input
  argument.
- `metrics.py` is the script to evaluate the trained models based on various
  evaluation metrics described in our paper. Additionally, it also plots the
  prediction and target data.
- `analyse_result_forward.py` is the script to convert the saved pickle file
  from the metrics calculation script into pandas dataframe format and save it
  as a CSV file. Additionally it also plots a bar chart to compare the results
  between different models.
- `analyse_result_inverse.py` is the script to convert the saved pickle file
  from the metrics calculation script into pandas dataframe format and save it
  as a CSV file. This script is used for the inverse problems. Additionally it
  also plots a bar chart to compare the results between different models.
- `fno` contains the scripts of FNO implementation. These are partly adapted
  from the
  [FNO repository](https://github.com/zongyi-li/fourier_neural_operator).
- `unet` contains the scripts of U-Net implementation. These are partly adapted
  from the
  [U-Net repository](https://github.com/mateuszbuda/brain-segmentation-pytorch).
- `pinn` contains the scripts of PINN implementation. These utilize the
  [DeepXDE library](https://github.com/lululxvi/deepxde).
- `inverse` contains the model for inverse model based on gradient.
- `config` contains the yaml files for the model training input. The default
  templates for different equations are provided in the
  [args](./pdebench/models/config/args) directory. User just needs to copy and
  paste them to the args keyword in the
  [config.yaml](./pdebench/models/config/config.yaml) file.

An example to run the forward model training can be found in
[run_forward_1D.sh](./pdebench/models/run_forward_1D.sh), and an example to run
the inverse model training can be found in
[run_inverse.sh](./pdebench/models/run_inverse.sh).

### Short explanations on the config args

- model_name: string, containing the baseline model name, either 'FNO', 'Unet',
  or 'PINN'.
- if_training: bool, set True for training, or False for evaluation.
- continue_training: bool, set True to continue training from a checkpoint.
- num_workers: int, number of workers for the PyTorch dataloader.
- batch_size: int, training batch size.
- initial_step: int, number of time steps used as input for FNO and U-Net.
- t_train: int, number of the last time step used for training (for
  extrapolation testing, set this to be < Nt).
- model_update: int, number of epochs to save model.
- filename: str, has to match the dataset filename.
- single_file: bool, set False for 2D diffusion-reaction, 1D diffusion-sorption,
  and the radial dam break scenarios, and set True otherwise.
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
- pushforward: bool, set True for pushforward training, False otherwise (ar_mode
  also has to be set True).
- unroll_step: int, number of time steps to backpropagate in the pushforward
  training.

#### FNO specific args:

- num_channels: int, number of channels (variables).
- modes: int, number of Fourier modes to multiply.
- width: int, number of channels for the Fourier layer.

#### INVERSE specific args:

- base_path: string, location of the data directory
- training_type: string, type of training, autoregressive, single
- mcmc_num_samples: int, number of generated samples
- mcmc_warmup_steps: 10
- mcmc_num_chains: 1
- num_samples_max: 1000
- in_channels_hid: 64
- inverse_model_type: string, type of inverse inference model, ProbRasterLatent,
  InitialConditionInterp
- inverse_epochs: int, number of epochs for the gradient based method
- inverse_learning_rate: float, learning rate for the gradient based method
- inverse_verbose_flag: bool, some printing

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

We provide the benchmark datasets we used in the paper through our
[DaRUS data repository](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986).
The data generation configuration can be found in the paper. Additionally, the
pretrained models are also available to be downloaded from
[PDEBench Pretrained Models](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987)
DaRus repository. To use the pretrained models, users can specify the argument
`continue_training: True` in the
[config file](./pdebench/models/config/config.yaml).

---

## Directory Tour

Below is an illustration of the directory structure of PDEBench.

```
📂 pdebench
|_📁 models
  |_📁 pinn    # Model: Physics-Informed Neural Network
    |_📄 train.py
    |_📄 utils.py
    |_📄 pde_definitions.py
  |_📁 fno     # Model: Fourier Neural Operator
    |_📄 train.py
    |_📄 utils.py
    |_📄 fno.py
  |_📁 unet    # Model: U-Net
    |_📄 train.py
    |_📄 utils.py
    |_📄 unet.py
  |_📁 inverse # Model: Gradient-Based Inverse Method
    |_📄 train.py
    |_📄 utils.py
    |_📄 inverse.py
  |_📁 config  # Config: All config files reside here
  |_📄 train_models_inverse.py
  |_📄 run_forward_1D.sh
  |_📄 analyse_result_inverse.py
  |_📄 train_models_forward.py
  |_📄 run_inverse.sh
  |_📄 metrics.py
  |_📄 analyse_result_forward.py
|_📁 data_download  # Data: Scripts to download data from DaRUS
  |_📁 config
  |_📄 download_direct.py
  |_📄 download_easydataverse.py
  |_📄 visualize_pdes.py
  |_📄 README.md
  |_📄 download_metadata.csv
|_📁 data_gen   # Data: Scripts to generate data
  |_📁 configs
  |_📁 data_gen_NLE
  |_📁 src
  |_📁 notebooks
  |_📄 gen_diff_sorp.py
  |_📄 plot.py
  |_📄 example.env
  |_📄 gen_ns_incomp.py
  |_📄 gen_diff_react.py
  |_📄 uploader.py
  |_📄 gen_radial_dam_break.py
|_📄 __init__.py
```

---

## Publications & Citations

Please cite the following papers if you use PDEBench datasets and/or source code
in your research.

<details>
<summary>
    <a href="https://arxiv.org/abs/2210.07182">PDEBench: An Extensive Benchmark for Scientific Machine Learning - NeurIPS'2022 </a>
</summary>
<br/>

```
@inproceedings{PDEBench2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
title = {{PDEBench: An Extensive Benchmark for Scientific Machine Learning}},
year = {2022},
booktitle = {36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
url = {https://arxiv.org/abs/2210.07182}
}
```

</details>

<details>
<summary>
    <a href="https://doi.org/10.18419/darus-2986">PDEBench Datasets - NeurIPS'2022 </a>
</summary>
<br/>

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

</details>

<details>
<summary>
    <a href="https://arxiv.org/abs/2304.14118">Learning Neural PDE Solvers with Parameter-Guided Channel Attention - ICML'2023 </a>
</summary>
<br/>

```
@article{cape-takamoto:2023,
     author   = {Makoto Takamoto and
                 Francesco Alesiani and
                 Mathias Niepert},
 title        = {Learning Neural {PDE} Solvers with Parameter-Guided Channel Attention},
 journal      = {CoRR},
 volume       = {abs/2304.14118},
 year         = {2023},
 url          = {https://doi.org/10.48550/arXiv.2304.14118},
 doi          = {10.48550/arXiv.2304.14118},
 eprinttype    = {arXiv},
 eprint       = {2304.14118},
 }
```

</details>

<details>
<summary>
    <a href="https://openreview.net/forum?id=I4WlXAA9Gd"> Vectorized Conditional Neural Fields: A Framework for Solving Time-dependent Parametric Partial Differential Equations - ICLR-W'2024 & ICML'2024 </a>
</summary>
<br/>

```
@inproceedings{vcnef-vectorized-conditional-neural-fields-hagnberger:2024,
author = {Hagnberger, Jan and Kalimuthu, Marimuthu and Musekamp, Daniel and Niepert, Mathias},
title = {{Vectorized Conditional Neural Fields: A Framework for Solving Time-dependent Parametric Partial Differential Equations}},
year = {2024},
booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML 2024)}
}
```

</details>

<details>
<summary>
    <a href="https://arxiv.org/abs/2408.01536"> Active Learning for Neural PDE Solvers - NeurIPS-W'2024 </a>
</summary>
<br/>

```
@article{active-learn-neuralpde-benchmark-musekamp:2024,
 author       = {Daniel Musekamp and
                 Marimuthu Kalimuthu and
                 David Holzm{\"{u}}ller and
                 Makoto Takamoto and
                 Mathias Niepert},
 title        = {Active Learning for Neural {PDE} Solvers},
 journal      = {CoRR},
 volume       = {abs/2408.01536},
 year         = {2024},
 url          = {https://doi.org/10.48550/arXiv.2408.01536},
 doi          = {10.48550/ARXIV.2408.01536},
 eprinttype    = {arXiv},
 eprint       = {2408.01536},
}
```

</details>

---

## Code contributors

- [Makato Takamoto](https://github.com/mtakamoto-D)
  ([NEC laboratories Europe](https://www.neclab.eu/))
- [Timothy Praditia](https://github.com/timothypraditia)
  ([Stuttgart Center for Simulation Science | University of Stuttgart](https://www.simtech.uni-stuttgart.de/))
- [Raphael Leiteritz](https://github.com/leiterrl)
  ([Stuttgart Center for Simulation Science | University of Stuttgart](https://www.simtech.uni-stuttgart.de/))
- [Francesco Alesiani](https://github.com/falesiani)
  ([NEC laboratories Europe](https://www.neclab.eu/))
- [Dan MacKinlay](https://danmackinlay.name/)
  ([CSIRO’s Data61](https://data61.csiro.au/))
- [Marimuthu Kalimuthu](https://github.com/kmario23)
  ([Stuttgart Center for Simulation Science | University of Stuttgart](https://www.simtech.uni-stuttgart.de/))
- [John Kim](https://github.com/johnmjkim)
  ([ANU TechLauncher](https://comp.anu.edu.au/TechLauncher/)/[CSIRO’s Data61](https://data61.csiro.au/))
- [Gefei Shan](https://github.com/davecatmeow)
  ([ANU TechLauncher](https://comp.anu.edu.au/TechLauncher/)/[CSIRO’s Data61](https://data61.csiro.au/))
- [Yizhou Yang](https://github.com/verdantwynnd)
  ([ANU TechLauncher](https://comp.anu.edu.au/TechLauncher/)/[CSIRO’s Data61](https://data61.csiro.au/))
- [Ran Zhang](https://github.com/maphyca)
  ([ANU TechLauncher](https://comp.anu.edu.au/TechLauncher/)/[CSIRO’s Data61](https://data61.csiro.au/))
- [Simon Brown](https://github.com/SimonSyBrown)
  ([ANU TechLauncher](https://comp.anu.edu.au/TechLauncher/)/[CSIRO’s Data61](https://data61.csiro.au/))

## License

MIT licensed, except where otherwise stated. See `LICENSE.txt` file.
