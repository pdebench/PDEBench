import os
import pdb
from functools import partial
import random
import logging
import string

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.utils.data as torchdata
import torch.utils.data.dataloader as dataloader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as dset

import torchtext
from torchtext.data.utils import get_tokenizer

import h5py

from .. import DATA_DIR

from .utils import interleave, process_device_arg, random_index_split, \
                   spectrally_prescribed_matrix, rot, rot_evecs

from .sqrtm import create_symm_matrix

logger = logging.getLogger(__name__)


DATASET_NCLASSES = {
    'MNIST': 10,
    'FashionMNIST': 10,
    'EMNIST': 26,
    'KMNIST': 10,
    'USPS': 10,
    'CIFAR10': 10,
    'SVHN': 10,
    'STL10': 10,
    'LSUN': 10,
    'tiny-ImageNet': 200
}

DATASET_SIZES = {
    'MNIST': (28,28),
    'FashionMNIST': (28,28),
    'EMNIST': (28,28),
    'QMNIST': (28,28),
    'KMNIST': (28,28),
    'USPS': (16,16),
    'SVHN': (32, 32),
    'CIFAR10': (32, 32),
    'STL10': (96, 96),
    'tiny-ImageNet': (64,64)
}

DATASET_NORMALIZATION = {
    'MNIST': ((0.1307,), (0.3081,)),
    'USPS' : ((0.1307,), (0.3081,)),
    'FashionMNIST' : ((0.1307,), (0.3081,)),
    'QMNIST' : ((0.1307,), (0.3081,)),
    'EMNIST' : ((0.1307,), (0.3081,)),
    'KMNIST' : ((0.1307,), (0.3081,)),
    'ImageNet': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'tiny-ImageNet': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'CIFAR10': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'STL10': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
}


def sort_by_label(X,Y):
    idxs = np.argsort(Y)
    return X[idxs,:], Y[idxs]


### Data Transforms
class DiscreteRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class SubsetSampler(torch.utils.data.Sampler):
    r"""Samples elements in order (not randomly) from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
        (this is identical to torch's SubsetRandomSampler except not random)
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None, target_transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

class SubsetFromLabels(torch.utils.data.dataset.Dataset):
    """ Subset of a dataset at specified indices.

    Adapted from torch.utils.data.dataset.Subset to allow for label re-mapping
    without having to copy whole dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        targets_map (dict, optional):  Dictionary to map targets with
    """
    def __init__(self, dataset, labels, remap=False):
        self.dataset = dataset
        self.labels  = labels
        self.classes = [dataset.classes[i] for i in labels]
        self.mask    = np.isin(dataset.targets, labels).squeeze()
        self.indices = np.where(self.mask)[0]
        self.remap   = remap
        targets = dataset.targets[self.indices]
        if remap:
            V = sorted(np.unique(targets))
            assert list(V) == list(labels)
            targets = torch.tensor(np.digitize(targets, self.labels, right=True))
            self.tmap = dict(zip(V,range(len(V))))
        self.targets = targets

    def __getitem__(self, idx):
        if self.remap is False:
            return self.dataset[self.indices[idx]]
        else:
            item =  self.dataset[self.indices[idx]]
            return (item[0], self.tmap[item[1]])

    def __len__(self):
        return len(self.indices)

def subdataset_from_labels(dataset, labels, remap=True):
    mask = np.isin(dataset.targets, labels).squeeze()
    idx  = np.where(mask)[0]
    subdataset = Subset(dataset,idx, remap_targets=True)
    return subdataset


def dataset_from_numpy(X, Y, classes = None):
    targets =  torch.LongTensor(list(Y))
    ds = TensorDataset(torch.from_numpy(X).type(torch.FloatTensor),targets)
    ds.targets =  targets
    ds.classes = classes if classes is not None else [i for i in range(len(np.unique(Y)))]
    return ds


gmm_configs = {
    'star': {
            'means': [torch.Tensor([0,0]),
                      torch.Tensor([0,-2]),
                      torch.Tensor([2,0]),
                      torch.Tensor([0,2]),
                      torch.Tensor([-2,0])],
            'covs':  [spectrally_prescribed_matrix([1,1], torch.eye(2)),
                      spectrally_prescribed_matrix([2.5,1], torch.eye(2)),
                      spectrally_prescribed_matrix([1,20], torch.eye(2)),
                      spectrally_prescribed_matrix([10,1], torch.eye(2)),
                      spectrally_prescribed_matrix([1,5], torch.eye(2))
                     ],
            'spread': 6,
    }

}

def make_gmm_dataset(config='random', classes=10,dim=2,samples=10,spread = 1,
                     shift=None, rotate=None, diagonal_cov=False, shuffle=True):
    """ Generate Gaussian Mixture Model datasets.

    Arguments:
        config (str): determines cluster locations, one of 'random' or 'star'
        classes (int): number of classes in dataset
        dim (int): feature dimension of dataset
        samples (int): number of samples in dataset
        spread (int): separation of clusters
        shift (bool): whether to add a shift to dataset
        rotate (bool): whether to rotate dataset
        diagonal_cov(bool): whether to use a diagonal covariance matrix
        shuffle (bool): whether to shuffle example indices

    Returns:
        X (tensor): tensor of size (samples, dim) with features
        Y (tensor): tensor of size (samples, 1) with labels
        distribs (torch.distributions): data-generating distributions of each class

    """
    means, covs, distribs = [], [], []
    _configd = None if config == 'random' else gmm_configs[config]
    spread = spread if (config == 'random' or not 'spread' in _configd) else _configd['spread']
    shift  = shift if (config == 'random' or not 'shift' in _configd) else _configd['shift']

    for i in range(classes):
        if config == 'random':
            mean = torch.randn(dim)
            cov  = create_symm_matrix(1, dim, verbose=False).squeeze()
        elif config == 'star':
            mean = gmm_configs['star']['means'][i]
            cov  = gmm_configs['star']['covs'][i]
        if rotate:
            mean = rot(mean, rotate)
            cov  = rot_evecs(cov, rotate)

        if diagonal_cov:
            cov.masked_fill_(~torch.eye(dim, dtype=bool), 0)

        means.append(spread*mean)
        covs.append(cov)
        distribs.append(MultivariateNormal(means[-1],covs[-1]))

    X = torch.cat([P.sample(sample_shape=torch.Size([samples])) for P in distribs])
    Y = torch.LongTensor([samples*[i] for i in range(classes)]).flatten()

    if shift:
        X += torch.tensor(shift)

    if shuffle:
        idxs = torch.randperm(Y.shape[0])
        X = X[idxs, :]
        Y = Y[idxs]
    return X, Y, distribs

def load_torchvision_data(dataname, valid_size=0.1, splits=None, shuffle=True,
                    stratified=False, random_seed=None, batch_size = 64,
                    resize=None, to3channels=False,
                    maxsize = None, maxsize_test=None, num_workers = 0, transform=None,
                    data=None, datadir=None, download=True, filt=False, print_stats = False):
    """ Load torchvision datasets.

        We return train and test for plots and post-training experiments
    """
    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = 'ImageNet'

        transform_list = []

        if dataname in ['MNIST', 'USPS'] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        transform_list.append(
            torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 2:
            train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == 'EMNIST':
            split = 'letters'
            train = DATASET(datadir, split=split, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, split=split, train=False, download=download, transform=valid_transform)
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(['C', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'])
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                'byclass': list(_all_classes),
                'bymerge': sorted(list(_all_classes - _merged_classes)),
                'balanced': sorted(list(_all_classes - _merged_classes)),
                'letters': list(string.ascii_lowercase),
                'digits': list(string.digits),
                'mnist': list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == 'letters':
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == 'STL10':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            train.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
            test.classes = train.classes
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(test.labels)
        elif dataname == 'SVHN':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        elif dataname == 'LSUN':
            pdb.set_trace()
            train = DATASET(datadir, classes='train', download=download, transform=train_transform)
        else:
            train = DATASET(datadir, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, train=False, download=download, transform=valid_transform)
    else:
        train, test = data


    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets  = torch.LongTensor(test.targets)

    if not hasattr(train, 'classes') or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes  = sorted(torch.unique(train.targets).tolist())


    ### Data splitting
    fold_idxs    = {}
    if splits is None and valid_size == 0:
        ## Only train
        fold_idxs['train'] = np.arange(len(train))
    elif splits is None and valid_size > 0:
        ## Train/Valid
        train_idx, valid_idx = random_index_split(len(train), 1-valid_size, (maxsize, None)) # No maxsize for validation
        fold_idxs['train'] = train_idx
        fold_idxs['valid'] = valid_idx
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ['split_{}'.format(i) for i in range(len(splits))]
            slens  = splits
        slens = np.array(slens)
        if any(slens < 0): # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, 'Can only deal with one split being -1'
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if 'train' in snames:
                slens[snames.index('train')] = len(train) - slens[np.array(snames) != 'train'].sum()

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [np.random.permutation(np.where(train.targets==c)).T for c in np.unique(train.targets)]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum() # Need to make cumulative for np.split
        split_idxs = [np.sort(s) for s in np.split(idxs, slens)[:-1]] # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i,v in enumerate(split_idxs)}


    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace = False))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k,idxs in fold_idxs.items()}


    ### Create DataLoaders
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)

    fold_loaders = {k:dataloader.DataLoader(train, sampler=sampler,**dataloader_args)
                    for k,sampler in fold_samplers.items()}

    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace = False))
        sampler_test = SubsetSampler(test_idxs) # For test don't want Random
        dataloader_args['sampler'] = sampler_test
    else:
        dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders['test'] = test_loader

    fnames, flens = zip(*[[k,len(v)] for k,v in fold_idxs.items()])
    fnames = '/'.join(list(fnames) + ['test'])
    flens  = '/'.join(map(str, list(flens) + [len(test)]))

    if hasattr(train, 'data'):
        logger.info('Input Dim: {}'.format(train.data.shape[1:]))
    logger.info('Classes: {} (effective: {})'.format(len(train.classes), len(torch.unique(train.targets))))
    print(f'Fold Sizes: {flens} ({fnames})')

    return fold_loaders, {'train': train, 'test':test}


def load_imagenet(datadir=None, resize=None, tiny=False, augmentations=False, **kwargs):
    """ Load ImageNet dataset """
    if datadir is None and (not tiny):
        datadir = os.path.join(DATA_DIR,'imagenet')
    elif datadir is None and tiny:
        datadir = os.path.join(DATA_DIR,'tiny-imagenet-200')

    traindir = os.path.join(datadir, "train")
    validdir = os.path.join(datadir, "val")
    if augmentations:
        train_transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
        ]
    else:
        train_transform_list = [
            transforms.Resize(224), # revert back to 256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
        ]

    valid_transform_list = [
        transforms.Resize(224),# revert back to 256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
    ]

    if resize is not None:
        train_transform_list.insert(3, transforms.Resize(
            (resize, resize)))
        valid_transform_list.insert(2, transforms.Resize(
            (resize, resize)))

    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose(
            train_transform_list
        ),
    )

    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose(
            valid_transform_list
        ),
    )
    fold_loaders, dsets = load_torchvision_data('Imagenet', transform=[],
                                                data=(train_data, valid_data),
                                                **kwargs)

    return fold_loaders, dsets


TEXTDATA_PATHS = {
    'AG_NEWS': 'ag_news_csv',
    'SogouNews': 'sogou_news_csv',
    'DBpedia': 'dbpedia_csv',
    'YelpReviewPolarity': 'yelp_review_polarity_csv',
    'YelpReviewFull': 'yelp_review_full_csv',
    'YahooAnswers': 'yahoo_answers_csv',
    'AmazonReviewPolarity': 'amazon_review_polarity_csv',
    'AmazonReviewFull': 'amazon_review_full_csv',
}

def load_textclassification_data(dataname, vecname='glove.42B.300d', shuffle=True,
            random_seed=None, num_workers = 0, preembed_sentences=True,
            loading_method='sentence_transformers', device='cpu',
            embedding_model=None,
            batch_size = 16, valid_size=0.1, maxsize=None, print_stats = False):
    """ Load torchtext datasets.

    Note: torchtext's TextClassification datasets are a bit different from the others:
        - they don't have split method.
        - no obvious creation of (nor access to) fields

    """



    def batch_processor_tt(batch, TEXT=None, sentemb=None, return_lengths=True, device=None):
        """ For torchtext data/models """
        labels, texts = zip(*batch)
        lens = [len(t) for t in texts]
        labels = torch.Tensor(labels)
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        texttensor = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_idx)
        if sentemb:
            texttensor = sentemb(texttensor)
        if return_lengths:
            return texttensor, labels, lens
        else:
            return texttensor, labels

    def batch_processor_st(batch, model, device=None):
        """ For sentence_transformers data/models """
        device = process_device_arg(device)
        with torch.no_grad():
            batch = model.smart_batching_collate(batch)
            ## Always run embedding model on gpu if available
            features, labels = st.util.batch_to_device(batch, device)
            emb = model(features[0])['sentence_embedding']
        return emb, labels


    if shuffle == True and random_seed:
        np.random.seed(random_seed)

    debug = False

    dataroot = '/tmp/' if debug else DATA_DIR #os.path.join(ROOT_DIR, 'data')
    veccache = os.path.join(dataroot,'.vector_cache')

    if loading_method == 'torchtext':
        ## TextClassification object datasets already do word to token mapping inside.
        DATASET = getattr(torchtext.datasets, dataname)
        train, test = DATASET(root=dataroot, ngrams=1)

        ## load_vectors reindexes embeddings so that they match the vocab's itos indices.
        train._vocab.load_vectors(vecname,cache=veccache,max_vectors = 50000)
        test._vocab.load_vectors(vecname,cache=veccache, max_vectors = 50000)

        ## Define Fields for Text and Labels
        text_field = torchtext.data.Field(sequential=True, lower=True,
                           tokenize=get_tokenizer("basic_english"),
                           batch_first=True,
                           include_lengths=True,
                           use_vocab=True)

        text_field.vocab = train._vocab

        if preembed_sentences:
            ## This will be used for distance computation
            vsize = len(text_field.vocab)
            edim  = text_field.vocab.vectors.shape[1]
            pidx  = text_field.vocab.stoi[text_field.pad_token]
            sentembedder = BoWSentenceEmbedding(vsize, edim, text_field.vocab.vectors, pidx)
            batch_processor = partial(batch_processor_tt,TEXT=text_field,sentemb=sentembedder,return_lengths=False)
        else:
            batch_processor = partial(batch_processor_tt,TEXT=text_field,return_lengths=True)
    elif loading_method == 'sentence_transformers':
        import sentence_transformers as st
        dpath  = os.path.join(dataroot,TEXTDATA_PATHS[dataname])
        reader = st.readers.LabelSentenceReader(dpath)
        if embedding_model is None:
            model  = st.SentenceTransformer('distilbert-base-nli-stsb-mean-tokens').eval()
        elif type(embedding_model) is str:
            model  = st.SentenceTransformer(embedding_model).eval()
        elif isinstance(embedding_model, st.SentenceTransformer):
            model = embedding_model.eval()
        else:
            raise ValueError('embedding model has wrong type')
        print('Reading and embedding {} train data...'.format(dataname))
        train  = st.SentencesDataset(reader.get_examples('train.tsv'), model=model)
        train.targets = train.labels
        print('Reading and embedding {} test data...'.format(dataname))
        test   = st.SentencesDataset(reader.get_examples('test.tsv'), model=model)
        test.targets = test.labels
        if preembed_sentences:
            batch_processor = partial(batch_processor_st, model=model, device=device)
        else:
            batch_processor = None

    ## Seems like torchtext alredy maps class ids to 0...n-1. Adapt class names to account for this.
    classes = torchtext.datasets.text_classification.LABELS[dataname]
    classes = [classes[k+1] for k in range(len(classes))]
    train.classes = classes
    test.classes  = classes

    train_idx, valid_idx = random_index_split(len(train), 1-valid_size, (maxsize, None)) # No maxsize for validation
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers,collate_fn=batch_processor)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler,**dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler,**dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader  = dataloader.DataLoader(test, **dataloader_args)

    if print_stats:
        print('Classes: {} (effective: {})'.format(len(train.classes), len(torch.unique(train.targets))))
        print('Fold Sizes: {}/{}/{} (train/valid/test)'.format(len(train_idx), len(valid_idx), len(test)))

    return train_loader, valid_loader, test_loader, train, test


class H5Dataset(torchdata.Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        super(H5Dataset, self).__init__()

        f = h5py.File(images_path, "r")
        self.data = f.get("x")

        g = h5py.File(labels_path, "r")
        self.targets = torch.from_numpy(g.get("y")[:].flatten())

        self.transform = transform
        self.classes = [0, 1]

    def __getitem__(self, index):
        if type(index) != slice:
            X = (
                torch.from_numpy(self.data[index, :, :, :]).permute(2, 0, 1).float()
                / 255
            )
        else:
            X = (
                torch.from_numpy(self.data[index, :, :, :]).permute(0, 3, 1, 2).float()
                / 255
            )

        y = int(self.targets[index])

        if self.transform:
            X = self.transform(torchvision.transforms.functional.to_pil_image(X))

        return X, y

    def __len__(self):
        return self.data.shape[0]


def combine_datasources(dset, dset_extra, valid_size=0, shuffle=True, random_seed=2019,
                        maxsize=None, device='cpu'):
    """ Combine two datasets.

    Extends dataloader with additional data from other dataset(s). Note that we
    add the examples in dset only to train (no validation)

    Arguments:
        dset (DataLoader): first dataloader
        dset_extra (DataLoader): additional dataloader
        valid_size (float): fraction of data use for validation fold
        shiffle (bool): whether to shuffle train data
        random_seed (int): random seed
        maxsize (int): maximum number of examples in either train or validation loader
        device (str): device for data loading

    Returns:
        train_loader_ext (DataLoader): train dataloader for combined data sources
        valid_loader_ext (DataLoader): validation dataloader for combined data sources

    """
    if shuffle == True and random_seed:
        np.random.seed(random_seed)

    ## Convert both to TensorDataset
    if isinstance(dset, torch.utils.data.DataLoader):
        dataloader_args = {k:getattr(dset, k) for k in ['batch_size', 'num_workers']}
        X, Y = load_full_dataset(dset, targets=True, device=device)
        d = int(np.sqrt(X.shape[1]))
        X = X.reshape(-1, 1, d, d)
        dset = torch.utils.data.TensorDataset(X, Y)
        logger.info(f'Main data size. X: {X.shape}, Y: {Y.shape}')
    elif isinstance(dst, torch.utils.data.Dataset):
        raise NotImplemented('Error: combine_datasources cant take Datasets yet.')

    merged_dset = torch.utils.data.ConcatDataset([dset, dset_extra])
    train_idx, valid_idx = random_index_split(len(dset), 1-valid_size, (maxsize, None)) # No maxsize for validation
    train_idx = np.concatenate([train_idx, np.arange(len(dset_extra)) + len(dset)])

    if shuffle:
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
    else:
        train_sampler = SubsetSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

    train_loader_ext  = dataloader.DataLoader(merged_dset, sampler =  train_sampler, **dataloader_args)
    valid_loader_ext  = dataloader.DataLoader(merged_dset, sampler =  valid_sampler, **dataloader_args)

    logger.info(f'Fold Sizes: {len(train_idx)}/{len(valid_idx)} (train/valid)')

    return train_loader_ext, valid_loader_ext
