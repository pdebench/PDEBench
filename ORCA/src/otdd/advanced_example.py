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
print(f'Embedded OTDD(MNIST,USPS)={d:8.2f}')
