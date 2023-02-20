# Optimal Transport Dataset Distance (OTDD)

Codebase accompanying the papers:
* [Geometric Dataset Distances via Optimal Transport](https://papers.nips.cc/paper/2020/file/f52a7b2610fb4d3f74b4106fb80b233d-Paper.pdf).
* [Dataset Dynamics via Gradient Flows in Probability Space](http://proceedings.mlr.press/v139/alvarez-melis21a/alvarez-melis21a.pdf).

See the papers for technical details, or the [MSR Blog Post](https://www.microsoft.com/en-us/research/blog/measuring-dataset-similarity-using-optimal-transport/) for a high-level introduction.

## Getting Started

### Installation

**Note**: It is highly recommended that the following be done inside a virtual environment


#### Via Conda (recommended)

If you use [ana|mini]conda , you can simply do:

```
conda env create -f environment.yaml python=3.8
conda activate otdd
conda install .
```

(you might need to install pytorch separately if you need a custom install)

#### Via pip

First install dependencies. Start by install pytorch with desired configuration using the instructions provided in the [pytorch website](https://pytorch.org/get-started/locally/). Then do:
```
pip install -r requirements.txt
```
Finally, install this package:
```
pip install .
```

## Usage Examples

A vanilla example for OTDD:

```python
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance


# Load datasets
loaders_src = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=2000)[0]
loaders_tgt = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=2000)[0]

# Instantiate distance
dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device='cpu')

d = dist.distance(maxsamples = 1000)
print(f'OTDD(src,tgt)={d}')

```

## Advanced Usage

### Using a custom feature distance

By default, OTDD uses the (squared) Euclidean distance between features. To use a custom distance in domains where it makes sense to use one (e.g., images), one can pass a callable to OTDD using the `feature_cost` arg. Example:

```python

import torch
from torchvision.models import resnet18

from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance, FeatureCost

# Load MNIST/CIFAR in 3channels (needed by torchvision models)
loaders_src = load_torchvision_data('CIFAR10', resize=28, maxsize=2000)[0]
loaders_tgt = load_torchvision_data('MNIST', resize=28, to3channels=True, maxsize=2000)[0]

# Embed using a pretrained (+frozen) resnet
embedder = resnet18(pretrained=True).eval()
embedder.fc = torch.nn.Identity()
for p in embedder.parameters():
    p.requires_grad = False

# Here we use same embedder for both datasets
feature_cost = FeatureCost(src_embedding = embedder,
                           src_dim = (3,28,28),
                           tgt_embedding = embedder,
                           tgt_dim = (3,28,28),
                           p = 2,
                           device='cpu')

dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          feature_cost = feature_cost,
                          sqrt_method = 'spectral',
                          sqrt_niters=10,
                          precision='single',
                          p = 2, entreg = 1e-1,
                          device='cpu')

d = dist.distance(maxsamples = 10000)

```


### Gradient Flows

```python

import os
import matplotlib
%matplotlib inline # Comment out if not on notebook

from otdd.pytorch.flows import OTDD_Gradient_Flow
from otdd.pytorch.flows import CallbackList, ImageGridCallback, TrajectoryDump

# Load datasets
loaders_src = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=2000)[0]
loaders_tgt = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=2000)[0]


outdir =  os.path.join('out', 'flows')
callbacks = CallbackList([
  ImageGridCallback(display_freq=2, animate=False, save_path = outdir + '/grid'),
])

flow = OTDD_Gradient_Flow(loaders_src['train'], loaders_tgt['train'],
                          ### Gradient Flow Args
                          method = 'xonly-attached',                          
                          use_torchoptim=True,
                          optim='adam',
                          steps=10,
                          step_size=1,
                          callback=callbacks,              
                          clustering_method='kmeans',                                      
                          ### OTDD Args                          
                          online_stats=True,
                          diagonal_cov = False,
                          device='cpu'
                          )
d,out = flow.flow()

```



## Acknowledgements

This repo relies on the [geomloss](https://www.kernel-operations.io/geomloss/) and [POT](https://pythonot.github.io/) packages for internal EMD and Sinkhorn algorithm implementation. We are grateful to the authors and maintainers of those projects.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
