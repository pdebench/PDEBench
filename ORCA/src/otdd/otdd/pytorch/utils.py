import os
from itertools import zip_longest, product
from functools import partial
from os.path import dirname
import numpy as np
import scipy.sparse
from tqdm.autonotebook import tqdm
import torch
import random
import pdb
import string
import logging
from sklearn.cluster import k_means, DBSCAN

import matplotlib.pyplot as plt


from PIL import Image
import PIL.ImageOps

import torch.nn as nn
import torch.utils.data as torchdata
import torch.utils.data.dataloader as dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from munkres import Munkres

from .nets import BoWSentenceEmbedding
from .sqrtm import sqrtm, sqrtm_newton_schulz

DATASET_NORMALIZATION = {
    'MNIST': ((0.1307,), (0.3081,)),
    'USPS' : ((0.1307,), (0.3081,)),
    'FashionMNIST' : ((0.1307,), (0.3081,)),
    'QMNIST' : ((0.1307,), (0.3081,)),
    'EMNIST' : ((0.1307,), (0.3081,)),
    'KMNIST' : ((0.1307,), (0.3081,)),
    'ImageNet': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'CIFAR10': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'camelyonpatch': ((0.70038027, 0.53827554, 0.69125885), (0.23614734, 0.27760974, 0.21410067))
}

logger = logging.getLogger(__name__)

def inverse_normalize(tensor, mean, std):
    _tensor = tensor.clone()
    for ch in range(len(mean)):
        _tensor[:,ch,:,:].mul_(std[ch]).add_(mean[ch])
    return _tensor

def process_device_arg(device):
    " Convient function to abstract away processing of torch.device argument"
    if device is None: # Default to cuda:0 if possible, otherwise cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif type(device) is str:
        device = torch.device(device)
    else:
        pass
    return device


def interleave(*a):
    ## zip_longest filling values with as many NaNs as values in second axis
    l = *zip_longest(*a, fillvalue=[np.nan]*a[0].shape[1]),
    ## build a 2d array from the list
    out = np.concatenate(l)
    ## return non-NaN values
    return out[~np.isnan(out[:,0])]


def random_index_split(input, alpha=0.9, max_split_sizes=(None,None)):
    " Returns two np arrays of indices, such that the first one has size alpha*n"
    if type(input) is int:
        indices, n  = np.arange(input), input
    elif type(input) is list:
        indices, n  = np.array(input), len(input)
    elif type(input) is np.ndarray:
        indices, n = input, len(input)
    np.random.shuffle(indices) # inplace
    split = int(np.floor(alpha * n))
    idxs1, idxs2 = np.array(indices[:split]), np.array(indices[split:])
    if max_split_sizes[0] is not None and (max_split_sizes[0] < len(idxs1)):
        idxs1 = np.sort(np.random.choice(idxs1, max_split_sizes[0], replace = False))
    if max_split_sizes[1] is not None and (max_split_sizes[1] < len(idxs2)):
        idxs2 = np.sort(np.random.choice(idxs2, max_split_sizes[1], replace = False))
    return idxs1, idxs2


def extract_dataset_targets(d):
    """ Extracts targets from dataset.

    Extracts labels, classes and effective indices from a object of type
    torch.util.data.dataset.**.

    Arguments:
        d (torch Dataset): dataset to extract targets from

    Returns:
        targets (tensor): tensor with integer targets
        classes (tensor): tensor with class labels (might or might not be integer)
        indices (tensor): indices of examples

    Note:
        Indices can differ from range(len(d)) if, for example, this is a Subset dataset.

    """
    assert isinstance(d, torch.utils.data.dataset.Dataset)
    if isinstance(d, torch.utils.data.dataset.Subset):
        dataset = d.dataset
        indices = d.indices
    elif isinstance(d, torch.utils.data.dataset.Dataset): # should be last option, since all above satisfy it
        dataset = d
        indices = d.indices if hasattr(d, 'indices') else None # this should always return None. Check.

    if hasattr(dataset, 'targets'): # most torchivision datasets
        targets = dataset.targets
    elif hasattr(dataset, '_data'): # some torchtext datasets
        targets = torch.LongTensor([e[0] for e in dataset._data])
    elif hasattr(dataset, 'tensors') and len(dataset.tensors) == 2: # TensorDatasets
        targets = dataset.tensors[1]
    elif hasattr(dataset, 'tensors') and len(dataset.tensors) == 1:
        logger.warning('Dataset seems to be unlabeled - this modality is in beta mode!')
        targets = None
    else:
        raise ValueError("Could not find targets in dataset.")

    classes = dataset.classes if hasattr(dataset, 'classes') else torch.sort(torch.unique(targets)).values

    if (indices is None) and (targets is not None):
        indices = np.arange(len(targets))
    elif indices is None:
        indices = np.arange(len(dataset))
    else:
        indices = np.sort(indices)
    # targets = torch.LongTensor(targets)
    # indices = indices.astype(int)

    return targets, classes, indices


def extract_dataloader_targets(dl):
    """ Extracts targets from dataloader.

    Extracts labels, classes and effective indices from a object of type
    torch.util.data.dataset.**.

    Arguments:
        d (torch DataLoader): dataloader to extract targets from

    Returns:
        targets (tensor): tensor with integer targets
        classes (tensor): tensor with class labels (might or might not be integer)
        indices (tensor): indices of examples

    Note:
        Indices can differ from range(len(d)) if, for example, this is a Subset dataset.

    """
    assert isinstance(dl, torch.utils.data.dataloader.DataLoader)
    assert hasattr(dl, 'dataset'), "Dataloader does not have dataset attribute."

    ## Extract targets from underlying dataset
    targets, classes, indices = extract_dataset_targets(dl.dataset)

    ## Now need to check if loader does some subsampling
    if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'indices'):
        idxs_sampler = dl.sampler.indices
        if indices is not None and len(indices)!=len(targets) and idxs_sampler is not None:
            ## Sampler indices should be subset of datasetd indices
            if set(idxs_sampler).issubset(set(indices)):
                indices = idxs_sampler
            else:
                print("STOPPING. Incosistent dataset and sampler indices.")
                pdb.set_trace()
        else:
            indices = idxs_sampler

    if indices is None:
        indices = np.arange(len(targets))
    else:
        indices = np.sort(indices)

    return targets, classes, indices


def extract_data_targets(d):
    """ Wrapper around extract_dataloader_targets and extract_dataset_targets,
    for convenience """
    if isinstance(d, torch.utils.data.dataloader.DataLoader):
        return extract_dataloader_targets(d)
    elif isinstance(d, torch.utils.data.dataset.Dataset):
        return extract_dataset_targets(d)
    else:
        raise  ValueError("Incompatible data object")


def load_full_dataset(data, targets=False, return_both_targets=False,
                      labels_keep=None, min_labelcount=None,
                      batch_size = 256,
                      maxsamples = None, device='cpu', dtype=torch.FloatTensor,
                      feature_embedding=None, labeling_function=None,
                      force_label_alignment = False,
                      reindex=False, reindex_start=0):
    """ Loads full dataset into memory.

    Arguments:
        targets (bool, or 'infer'): Whether to colleect and return targets (labels) too
        return_both_targets (bool): Only used when targets='infer'. Indicates whether
            the true targets should also be returned.
        labels_keep (list): If provided, will only keep examples with these labels
        reindex (bool): Whether/how to reindex labels. If True, will
                              reindex to {reindex_start,...,reindex_start+num_unique_labels}.

        maxsamples (int): Maximum number of examples to load. (this might not equal
                          actual size of return tensors, if label_keep also provided)

    Returns:
        X (tensor): tensor of dataset features, stacked along first dimension
        Y (tensor): tensor of dataset targets

    """
    device = process_device_arg(device)
    orig_idxs = None
    if type(data) == dataloader.DataLoader:
        loader = data
        if maxsamples:
            if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'indices'):
                if len(loader.sampler.indices) <= maxsamples:
                    logger.warning('Maxsamples is greater than number of effective examples in loader. Will not subsample.')
                else:
                    ## Resample from sampler indices.
                    orig_idxs = loader.sampler.indices
                    idxs = np.sort(np.random.choice(orig_idxs, maxsamples, replace=False))
                    loader.sampler.indices = idxs
            elif hasattr(loader, 'dataset'): # This probably means the sampler is not a subsampler. So len(dataset) is indeed true size.
                if len(loader.dataset) <= maxsamples:
                    logger.warning('Maxsamples is greater than number of examples in loader. Will not subsample.')
                else:
                    ## Create new sampler
                    idxs = np.sort(np.random.choice(len(loader.dataset), maxsamples, replace=False))
                    sampler = SubsetRandomSampler(idxs)
                    loader = dataloader.DataLoader(data, sampler=sampler, batch_size=batch_size)
            else:
                ## I don't think we'll ever be in this case.
                print('Warning: maxsamplers provided but loader doesnt have subsampler or dataset. Cannot subsample.')
    else:
        ## data is a dataset
        if maxsamples and len(data) > maxsamples:
            idxs = np.sort(np.random.choice(len(data), maxsamples, replace=False))
            sampler = SubsetRandomSampler(idxs)
            loader = dataloader.DataLoader(data, sampler=sampler, batch_size=batch_size)
        else:
            ## No subsampling
            loader = dataloader.DataLoader(data, batch_size=batch_size)

    X = []
    Y = []
    seen_targets = {}
    keeps = None
    collect_targets = targets and ((targets != 'infer') or return_both_targets)

    for batch in loader:#tqdm(loader, leave=False):
        x = batch[0]
        if (len(batch) == 2) and targets:
            y = batch[1]

        if feature_embedding is not None:
            ## if embedding is cuda, and device='cpu', want to map to device *after*
            ## embedding, to take advantage of CUDA forward pass.
            try:
                x = feature_embedding(x.type(dtype).cuda()).detach().to(device)
            except:
                x = feature_embedding(x.type(dtype).to(device)).detach()
        else:
            x = x.type(dtype).to(device)

        X.append(x.squeeze().view(x.shape[0],-1))
        if collect_targets: # = True or infer
            Y.append(y.to(device).squeeze())
    X = torch.cat(X)

    if collect_targets: Y = torch.cat(Y)

    if targets == 'infer':
        logger.warning('Performing clustering')
        # if Y is not None: # Save true targets before overwriting them with inferred
        #    Y_true = Y
        Y = labeling_function(X)
        Y_true = Y
        
        if force_label_alignment:
            K = torch.unique(Y_true).shape[0]
            M = [((Y == k) & (Y_true == l)).sum().item() for k,l in product(range(K),range(K))]
            M = np.array(M).reshape(K,K)
            idx_map = dict(Munkres().compute(1 - M/len(Y)))
            Y = torch.tensor([idx_map[int(y.item())] for y in Y])

    if min_labelcount is not None:
        assert not labels_keep, "Cannot specify both min_labelcount and labels_keep"
        vals, cts = torch.unique(Y, return_counts=True)
        labels_keep = torch.sort(vals[cts >= min_labelcount])[0]


    if labels_keep is not None: # Filter out examples with unwanted label
        keeps = np.isin(Y.cpu(), labels_keep)
        X = X[keeps,:]
        Y = Y[keeps]

    if orig_idxs is not None:
        loader.sampler.indices = orig_idxs
    if targets is False:
        return X
    else:
        if reindex:
            labels = sorted(torch.unique(Y).tolist())
            reindex_vals = range(reindex_start, reindex_start + len(labels))
            lmap = dict(zip(labels, reindex_vals))
            Y = torch.LongTensor([lmap[y.item()] for y in Y]).to(device)
        if not return_both_targets:
            return X, Y
        else:
            return X, Y, Y_true


def sample_kshot_task(dataset,k=10,valid=None):
    """ This is agnostic to the labels used, it will inferr them from dataset
        so it works equally well with remaped or non remap subsets.
    """
    inds_train = []
    Y = dataset.targets
    V = sorted(list(torch.unique(Y)))
    inds_valid = []
    for c in V:
        m = torch.where(Y == c)[0].squeeze()
        srt_ind = m[torch.randperm(len(m))]
        inds_train.append(srt_ind[:k])
        if valid:
            inds_valid.append(srt_ind[k:k+valid])
    inds_train = torch.sort(torch.cat(inds_train))[0]
    assert len(inds_train) == k*len(V)
    train = torch.utils.data.Subset(dataset,inds_train)
    tr_lbls = [train[i][1] for i in range(len(train))]
    tr_cnts = np.bincount(tr_lbls)
    assert np.all(tr_cnts == [k]*len(V))

    if valid:
        inds_valid = torch.sort(torch.cat(inds_valid))[0]
        valid = torch.utils.data.Subset(dataset,inds_valid)
        return train, valid
    else:
        return train


def load_trajectories(path, device='cpu'):
    Xt = torch.load(path + '/trajectories_X.pt')
    Yt = torch.load(path + '/trajectories_Y.pt')
    assert Xt.ndim == 3
    assert Yt.ndim == 2
    assert Xt.shape[0]  == Yt.shape[0]
    assert Xt.shape[-1] == Yt.shape[-1]
    n,d,t = Xt.shape
    logger.info(f'Trajectories: {n} points, {d} dim, {t} steps.')
    if device is not None:
        Xt = Xt.to(torch.device(device))
        Yt = Yt.to(torch.device(device))
    return Xt, Yt


def augmented_dataset(dataset, means, covs, maxn=1000):#, diagonal_cov=False):
    """ Generate moment-augmented dataset by concatenating features, means and
    covariances. This will only make sense when using Gaussians for target
    representation. Every instance in the augmented dataset will have form:

                    x̂_i = [x_i,mean(y_i),vec(cov(y_i))]

    Therefore:
        ||x̂_i - x̂_j||_p^p = ||x_i - x_j||_p^p +
                            ||mean(y_i)-mean(y_j)||_p^p +
                            ||sqrt(cov(y_i))-sqrt(cov(y_j))||_p^p

    """
    if type(dataset) is tuple and type(dataset[0]) is torch.Tensor:
        X, Y = dataset
    elif type(dataset) is torch.utils.data.dataset.Dataset:
        X, Y = load_full_dataset(dataset, targets=True)
    else:
        raise ValueError('Wrong Format')

    if maxn and maxn < X.shape[0]:
        idxs = sorted(np.random.choice(range(X.shape[0]),maxn, replace=False))
    else:
        idxs = range(X.shape[0])

    X = X[idxs,:]
    Y = Y[idxs]
    if Y.min() > 0: # We reindxed the labels, need to revert
        Y -= Y.min()
    M = means[Y[idxs],:]
    if covs[0].ndim == 1:
        ## Implies Covariance is diagonal
        sqrt_covs = torch.sqrt(covs)
    else:
        sqrt_covs = torch.stack([sqrtm(c) for c in torch.unbind(covs, 0)])

    C = sqrt_covs[Y[idxs],:]

    C = C.view(C.shape[0], -1)

    dim_before = X.shape[1]
    X_aug = torch.cat([X,M,C],1)
    logger.info('Augmented from dim {} to {}'.format(dim_before, X_aug.shape[1]))
    return X_aug


def extract_torchmeta_task(cs, class_ids):
    """ Extracts a single "episode" (ie, task) from a ClassSplitter object, in the
        form of a dataset, and appends variables needed by DatasetDistance computation.

        Arguments:
            cs (torchmeta.transforms.ClassSplitter): the ClassSplitter where to extract data from
            class_ids (tuple): indices of classes to be selected by Splitter

        Returns:
            ds_train (Dataset): train dataset
            ds_test (Dataset): test dataset

    """
    ds = cs[class_ids]
    ds_train, ds_test = ds['train'], ds['test']

    for ds in [ds_train, ds_test]:
        ds.targets = torch.tensor([ds[i][1] for i in range(len(ds))])
        ds.classes = [p[-1] for i,p in enumerate(cs.dataset._labels) if i in class_ids]
    return ds_train, ds_test


def save_image(tensor, fp, dataname, format='png', invert=True):
    """ Similar to torchvision's save_image, but corrects normalization """
    if dataname and dataname in DATASET_NORMALIZATION:
        ## Brings back to [0,1] range
        mean, std = (d[0] for d in DATASET_NORMALIZATION[dataname])
        tensor = tensor.mul(std).add_(mean)
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if invert:
        im = PIL.ImageOps.invert(im)
    im.save(fp, format=format)

def show_grid(tensor, dataname=None, invert=True, title=None,
             save_path=None, to_pil=False, ax = None,format='png'):
    """ Displays image grid. To be used after torchvision's make_grid """
    if dataname and dataname in DATASET_NORMALIZATION:
        ## Brings back to [0,1] range
        mean, std = (d[0] for d in DATASET_NORMALIZATION[dataname])
        tensor = tensor.mul(std).add_(mean)
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    ndarr = np.transpose(ndarr, (1,2,0))
    if to_pil:
        im = Image.fromarray(ndarr)
        if invert:
            im = PIL.ImageOps.invert(im)
        im.show(title=title)
        if save_path:
            im.save(save_path, format=format)
    else:
        if not ax: fig, ax  = plt.subplots()
        ax.imshow(ndarr, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if title: ax.set_title(title)

def coupling_to_csv(G, fp, thresh = 1e-14, sep=',', labels1=None,labels2=None):
    """ Dumps an OT coupling matrix to a csv file """
    sG = G.copy()
    if thresh is not None:
        sG[G<thresh] = 0
    sG = scipy.sparse.coo_matrix(sG)
    l1 = labels1 is not None
    l2 = labels2 is not None
    header = ['i', 'j', 'val']
    if l1: header.append('ci')
    if l2: header.append('cj')
    with open(fp, 'w') as f:
        f.write(sep.join(header) + '\n')
        for i,j,v in  zip(sG.row, sG.col, sG.data):
            row = [str(i),str(j),'{:.2e}'.format(v)]
            if l1: row.append(str(labels1[i]))
            if l2: row.append(str(labels2[j]))
            f.write(sep.join(row) + '\n')
    print('Done!')

def multiclass_hinge_loss(Y1, Y2, margin=1.0):
    """ Hinge-loss for multi-class classification settings """
    Y1 = torch.nn.functional.one_hot(Y1)
    Y2 = torch.nn.functional.one_hot(Y2)
    n,K = Y1.shape
    assert Y1.shape[1] == Y2.shape[1]
    m = Y2.shape[0]
    res = torch.zeros(n,m)
    for k in range(K):
        res += torch.relu(margin-torch.ger(Y1[:,k], Y2[:,k]))**2
    return res

### DEBUGGING TOOLS ###

def get_printer(msg):
    """ This function returns a printer function, that prints information about
    a tensor's gradient. Used by register_hook in the backward pass.
    """
    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(f"{msg} shape: {tensor.shape}"
                  f" max: {tensor.max():8.2f} min: {tensor.min():8.2f}"
                  f" mean: {tensor.mean():8.2f}")
    return printer


def register_gradient_hook(tensor, msg):
    """ Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))

### EIGEN-MANIPULATION TOOLS ###

def rot(v, theta):
    " Extends torch.rot90 to arbitrary degrees (works only for 2d data) "
    theta = np.pi*(theta/180)
    R = torch.Tensor([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return torch.matmul(R, v)

def rot_evecs(M, theta):
    " Rotate eigenvectors of matrix M "
    evals, evecs = torch.eig(M, eigenvectors = True)
    evecs_rot = rot(evecs, theta)
    return spectrally_prescribed_matrix(evals, evecs_rot)

def spectrally_prescribed_matrix(evals, evecs):
    """ Make a matrix with the desired eigenvaules and eigenvectors.
        Args:
            evals is tensor of size (n, )
            evecs is tensor of size (n,n), columns are eigenvectors
    """
    if type(evals) is list:
        evals = torch.Tensor(evals)
    elif evals.ndim == 2:
        " Probably evals comes from torch.eig, get rid of complex part"
        evals = evals[:,0]
    assert len(evals) == evecs.shape[0]
    assert evals.shape[0] == evecs.shape[1]
    S = torch.diag(evals)
    M = torch.matmul(evecs, torch.matmul(S, evecs.T))
    return M

#### MISC

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
