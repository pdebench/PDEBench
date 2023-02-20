""" Main module for optimal transport dataset distance.

Throught this module, source and target are often used to refer to the two datasets
being compared. This notation is legacy from NLP, and does not carry other particular
meaning, e.g., the distance is nevertheless symmetric (though not always identical -
due to stochsticity in the computation) to the order of D1/D2. The reason for this
notation is that here X and Y are usually reserved to distinguish between features and labels.

Other important notation:
    X1, X2: feature tensors of the two datasets
    Y1, Y2: label tensors of the two datasets
    N1, N2 (or N,M): number of samples in datasets
    D1, D2: (feature) dimension of the datasets
    C1, C2: number of classes in the datasets
    π: transport coupling

"""
import os
import pdb
from time import time
import itertools
import numpy as np
from tqdm.autonotebook import tqdm
import torch
from functools import partial
import inspect
import logging
import geomloss
import ot
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import k_means, DBSCAN

## Local Imports
from ..plotting import heatmap, gaussian_density_plot, imshow_group_boundaries
from .utils import load_full_dataset, augmented_dataset, extract_data_targets
from .moments import compute_label_stats
from .wasserstein import efficient_pwdist_gauss, pwdist_exact, pwdist_upperbound, pwdist_means_only
from .utils import register_gradient_hook, process_device_arg, multiclass_hinge_loss, load_full_dataset


import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
    nodisplay = True
else:
    nodisplay = False


logger = logging.getLogger(__name__)

try:
    import ot.gpu
except:
    logger.warning('ot.gpu not found - coupling computation will be in cpu')


cost_routines = {
    1: (lambda x, y: geomloss.utils.distances(x, y)),
    2: (lambda x, y: geomloss.utils.squared_distances(x, y) / 2),
}


class DatasetDistance():
    """The main class for the Optimal Transport Dataset Distance.

    An object of this class is instantiated with two datasets (the source and
    target), which are stored in it, and various arguments determining how the
    OTDD is to be computed.


    Arguments:
        D1 (Dataset or Dataloader): the first (aka source) dataset.
        D2 (Dataset or Dataloader): the second (aka target) dataset.
        method (str): if set to 'augmentation', the covariance matrix will be
            approximated and appended to each point, if 'precomputed_labeldist',
            the label-to-label distance is computed exactly in advance.
        symmetric_tasks (bool): whether the two underlying datasets are the same.
            If true, will save some computation.
        feature_cost (str or callable): if not 'euclidean', must be a callable
            that implements a cost function between feature vectors.
        src_embedding (callable, optional): if provided, source data will be
            embedded using this function prior to distance computation.
        tgt_embedding (callable, optional): if provided, target data will be
            embedded using this function prior to distance computation.
        ignore_source_labels (bool): for unsupervised computation of distance
        ignore_target_labels (bool): for unsupervised computation of distance
        loss (str): loss type to be passed to samples_loss. only 'sinkhorn' is
            accepted for now.
        debiased_loss (bool): whether to use the debiased version of sinkhorn.
        p (int): the coefficient in the OT cost (i.e., the p in p-Wasserstein).
        entreg (float): the strength of entropy regularization for sinkhorn.
        λ_x (float): weight parameter for feature component of distance.
        λ_y (float): weight parameter for label component of distance.
        inner_ot_method (str): the method to compute the inner (instance-wise)
            OT problem. Must be one of 'gaussian_approx', 'exact', 'jdot', or
            'naive_upperbound'. If set to 'gaussian_approx', the label distributions
            are approximated as Gaussians, and thus their distance is computed as
            the Bures-Wasserstein distance. If set to 'exact', no approximation is
            used, and their distance is computed as an exact Wasserstein problem.
            If 'naive_upperbound', a simple upper bound on the exact distance is
            computed. If 'jdot', the label distance is computed using a classifi-
            cation loss (see JDOT paper for details).
        inner_ot_loss (str): loss type fo inner OT problem.
        inner_ot_debiased (bool): whether to use the debiased version of sinkhorn
            in the inner OT problem.
        inner_ot_p (int): the coefficient in the inner OT cost.
        inner_ot_entreg (float): the strength of entropy regularization for sinkhorn
            in the inner OT problem.
        diagonal_cov (bool): whether to use the diagonal approxiation to covariance.
        min_labelcount (int): classes with less than `min_labelcount` examples will
            be ignored in the computation of the distance.
        online_stats (bool): whether to compute the per-label means and covariance
            matrices online. If false, for every class, all examples are loaded
            into memory.
        coupling_method (str): If 'geomloss', the OT coupling is computed from
            the dual potentials obtained from geomloss (faster, less precise),
            if 'pot', it will recomputed using the POT library.
        sqrt_method (str): If 'spectral' or 'exact', it uses eigendecomposition
            to compute square root matrices (exact, slower). If 'approximate',
            it uses Newton-Schulz iterative algorithm (can be faster, though less exact).
        sqrt_niters (int): Only used if `sqrt_method` is 'approximate'. Determines
            the number of interations used for Newton-Schulz's approach to sqrtm.
        sqrt_pref (int): One of 0 or 1. Preference for cov sqrt used in cross-wass
            distance (only one of the two is needed, see efficient_pairwise_wd_gauss). Useful
            for differentiable settings, two avoid unecessary computational graph.
        nworkers_stats (int): number of parallel workers used in mean and
            covariance estimation.
        coupling_method (str): method to use for computing coupling matrix.
        nworkers_dists(int): number of parallel workers used in distance computation.
        eigen_correction (bool): whether to use eigen-correction on covariance
            matrices for additional numerical stability.
        device (str): Which device to use in pytorch convention (e.g. 'cuda:2')
        precision (str): one of 'single' or 'double'.
        verbose (str): level of verbosity.

    """

    def __init__(self, D1=None, D2=None,
                 ## General Arguments
                 method='precomputed_labeldist',
                 symmetric_tasks=False,
                 feature_cost='euclidean',
                 src_embedding=None,
                 tgt_embedding=None,
                 ignore_source_labels=False,
                 ignore_target_labels=False,
                 ## Outer OT (dataset to dataset) problem arguments
                 loss='sinkhorn', debiased_loss=True, p=2, entreg=0.1,
                 λ_x=1.0, λ_y=1.0,
                 ## Inner OT (label to label) problem arguments
                 inner_ot_method = 'gaussian_approx',
                 inner_ot_loss='sinkhorn',
                 inner_ot_debiased=False,
                 inner_ot_p=2,
                 inner_ot_entreg=0.1,
                 ## Gaussian Approximation Args
                 diagonal_cov=False,
                 min_labelcount=2,
                 max_labelcount = 50000,
                 online_stats=True,
                 sqrt_method='spectral',
                 sqrt_niters=20,
                 sqrt_pref=0,
                 nworkers_stats=0,
                 ## Misc
                 coupling_method='geomloss',
                 nworkers_dists=0,
                 eigen_correction=False,
                 device='cpu',
                 precision='single',
                 verbose=1,load_prev_dyy1=None, *args, **kwargs):

        self.method = method
        assert self.method in ['precomputed_labeldist', 'augmentation', 'jdot']
        self.symmetric_tasks = symmetric_tasks
        self.diagonal_cov = diagonal_cov
        ## For outer OT problem
        self.p = p
        self.entreg = entreg
        self.loss = loss
        self.debiased_loss = debiased_loss
        self.feature_cost = feature_cost
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.ignore_source_labels = ignore_source_labels
        self.ignore_target_labels = ignore_target_labels
        self.λ_x = λ_x
        self.λ_y = λ_y
        ## For inner (label) OT problem - only used if gaussian approx is False
        self.inner_ot_method = inner_ot_method
        self.inner_ot_p = inner_ot_p
        self.inner_ot_entreg = inner_ot_entreg
        self.inner_ot_loss = inner_ot_loss
        self.inner_ot_debiased = inner_ot_debiased
        self.online_stats = online_stats
        self.coupling_method = coupling_method
        self.min_labelcount = min_labelcount
        self.max_labelcount = max_labelcount
        self.nworkers_stats = nworkers_stats
        self.nworkers_dists = nworkers_dists
        self.sqrt_method = sqrt_method
        if self.sqrt_method == 'exact':
            self.sqrt_method = 'spectral'
        self.sqrt_niters = sqrt_niters
        assert sqrt_pref in [0,1], 'sqrt pref must be 0 or 1'
        self.sqrt_pref   = sqrt_pref
        self.device = device
        self.precision = precision
        self.eigen_correction = eigen_correction
        self.verbose = verbose

        if self.method == 'augmentation' and not self.diagonal_cov:
            logger.error('Method augmentation requires diagonal_cov = True')

        ## Placeholders
        self.Means = [None, None]
        self.Covs = [None, None]
        self.label_distances = None
        self.X1, self.X2 = None, None
        self.Y1, self.Y2 = None, None
        self._pwlabel_stats_1 = None if load_prev_dyy1 is None else load_prev_dyy1
        self._pwlabel_stats_2 = None

        self.D1 = D1
        if D2 is None:
            self.D2 = self.D1
            self.symmetric_tasks = True
        else:
            self.D2 = D2

        if self.D1 is not None and self.D2 is not None:
            self._init_data(self.D1, self.D2)
        else:
            logger.warning('DatasetDistance initialized with empty data')

        #if self.src_embedding is not None or self.tgt_embedding is not None:
        #    self.feature_cost = partial(FeatureCost,
        #                           src_emb = self.src_embedding,
        #                           src_dim = (3,28,28),
        #                           tgt_emb = self.tgt_embedding,
        #                           tgt_dim = (3,28,28),
        #                           p = self.p, device=self.device)

        #self.src_embedding = None
        #self.tgt_embedding = None


    def _load_infer_labels(self, D, classes=None, reindex=None, reindex_start=None):

        if classes is not None:
            k = len(classes)
            labeling_fun = lambda X: torch.LongTensor(k_means(X.clone().detach().cpu().numpy(), k)[1])
        else:
            labeling_fun = lambda X: torch.LongTensor(DBSCAN(eps=5, min_samples = 4).fit(X).labels_)

        X, Y_infer, Y_true = load_full_dataset(D, targets='infer',
                                 min_labelcount=self.min_labelcount,
                                 labeling_function = labeling_fun,
                                 return_both_targets=True,
                                 force_label_alignment=True,
                                 reindex=reindex, reindex_start=reindex_start)

        return X, Y_infer, Y_true


    def _init_data(self, D1, D2):
        """ Preprocessing of datasets. Extracts value and coding for effective
        (i.e., those actually present in sampled data) class labels.
        """

        targets1, classes1, idxs1 = extract_data_targets(D1)
        targets2, classes2, idxs2 = extract_data_targets(D2)

        ## Get effective dataset number of samples
        self.idxs1, self.idxs2 = idxs1, idxs2
        self.n1 = len(self.idxs1)
        self.n2 = len(self.idxs2)

        if (targets1 is None) or self.ignore_source_labels: # Unsupervised setting
            X, Y_infer, Y_true = self._load_infer_labels(D1, classes1, reindex=True, reindex_start=0)
            self.targets1 = targets1 = Y_infer
            self.X1, self.Y1 = X, Y_infer
            if Y_true is not None: self.Y1_true = Y_true # will not be used by OTDD, stored only for downstream eval
        else:
            self.targets1 = targets1

        ## Effective classes seen in data (idxs here needed to be filtered
        ## in case dataloader has subsampler)
        ## Indices of classes (might be different from class ids!)
        if (targets1 is None) or self.ignore_source_labels:
            vals1, cts1 = torch.unique(targets1, return_counts=True)
        else:
            vals1, cts1 = torch.unique(targets1[idxs1], return_counts=True)

        ## Ignore everything with a label occurring less than k times
        self.V1 = torch.sort(vals1[(cts1 >= self.min_labelcount) & (cts1 <= self.max_labelcount)])[0]

        if (targets2 is None) or self.ignore_target_labels:
            reindex_start = len(self.V1) if (self.loss == 'sinkhorn' and self.debiased_loss) else True
            X, Y_infer, Y_true = self._load_infer_labels(D2, classes2, reindex=True, reindex_start=reindex_start)
            self.targets2 = targets2 = Y_infer - reindex_start
            assert self.targets2.min() == 0
            self.X2, self.Y2 = X, Y_infer
            if Y_true is not None: self.Y2_true = Y_true
        else:
            self.targets2 = targets2

        ## Effective classes seen in data (idxs here needed to be filtered
        ## in case dataloader has subsampler)
        ## Indices of classes (might be different from class ids!)
        if (targets2 is None) or self.ignore_target_labels:
            vals2, cts2 = torch.unique(targets2, return_counts=True)
        else:
            vals2, cts2 = torch.unique(targets2[idxs2], return_counts=True)

        ## Ignore everything with a label occurring less than k times
        self.V2 = torch.sort(vals2[(cts2 >= self.min_labelcount) & (cts2 <= self.max_labelcount)])[0]

        self.classes1 = self.V1 #[classes1[i] for i in self.V1]
        self.classes2 = self.V2 #[classes2[i] for i in self.V2]

        if self.method == 'jdot': ## JDOT only works if same labels on both datasets
            assert torch.all(self.V1 == self.V2)

        ## Keep track of real classes vs indices (always 0 to n)(useful if e.g., missing classes):
        self.class_to_idx_1 = {i: c for i, c in enumerate(self.V1)}
        self.class_to_idx_2 = {i: c for i, c in enumerate(self.V2)}

    def copy(self, keep=[], drop=[]):
        """ Copy method for Dataset Distance object.

        Copies 'shell' of object only: configs, but no dataset or its derivatives.

        """
        dataattrs = ['D1', 'D2','X1', 'X2','Y1','Y2','V1','V2',
                    'targets1', 'targets2', 'classes1', 'classes2',
                    'idxs1', 'idxs2', 'class_to_idx_1', 'class_to_idx_2',
                    'Covs', 'Means', 'label_distances', '_label_mean_distances']

        initattrs = list(inspect.signature(DatasetDistance).parameters.keys())

        if not keep:
            ## By default, we keep all non-data attribs, drop all data-dependent ones
            keep = set(initattrs) - set(['D1','D2'])
        elif keep == 'all':
            keep = set(self.__dict__.keys()) # postattrs + initattrs
        else:
            keep = set(self.__dict__.keys()).difference(dataattrs).union(keep)

        kept_init_attrs = set(initattrs).intersection(set(keep))
        dobj = DatasetDistance(**{k:self.__dict__[k] for k in kept_init_attrs})

        ## Must also add attribs that are not taken by __init__ method (because they're generated after)
        kept_post_attrs = set(keep).difference(kept_init_attrs)
        dobj.__dict__.update({k:self.__dict__[k] for k in kept_post_attrs})
        return dobj

    def _load_datasets(self, maxsamples=None, device=None):
        """ Dataset loading, wrapper for `load_full_dataset` function.

        Loads full datasets into memory (into gpu if in CUDA mode).

        Arguments:
            maxsamples (int, optional): maximum number of samples to load.
            device (str, optional): if provided, will override class attribute device.
        """
        logger.info('Concatenating feature vectors...')

        ## We probably don't ever want to store the full datasets in GPU
        device = 'cpu'

        dtype = torch.DoubleTensor if self.precision == 'double' else torch.FloatTensor

        if self.loss == 'sinkhorn' and self.debiased_loss:
            ## We will need to relabel targets {0,...,n-1} and {n,...,n-m-1}
            reindex_start_d2 = len(self.V1)
        else:
            ## Suffices to relabel targets {0,...,n-1} and {0,...,m-1}
            reindex_start_d2 = 0

        if self.X1 is None or self.Y1 is None:
            assert not self.ignore_source_labels, 'Should not be here if igoring target labels'
            self.X1, self.Y1 = load_full_dataset(self.D1, targets=True,
                                                 labels_keep=self.V1,
                                                 maxsamples=maxsamples,
                                                 device=device,
                                                 dtype=dtype,
                                                 reindex=True,
                                                 reindex_start = 0)
        if self.X2 is None or self.Y2 is None:
            assert not self.ignore_target_labels, 'Should not be here if igoring target labels'
            if self.symmetric_tasks:
                self.X2, self.Y2 = self.X1, self.Y1
            else:
                self.X2, self.Y2 = load_full_dataset(self.D2, targets=True,
                                                     labels_keep=self.V2,
                                                     maxsamples=maxsamples,
                                                     device=device,
                                                     dtype=dtype,
                                                     reindex=True,
                                                     reindex_start = reindex_start_d2)


        logger.info("Full datasets sizes")
        logger.info(" * D1 = {} x {} ({} unique labels)".format(*
                                                          self.X1.shape, len(self.V1)))
        logger.info(" * D2 = {} x {} ({} unique labels)".format(*
                                                          self.X2.shape, len(self.V2)))

    def _get_label_stats(self, side='both'):
        """ Return per-label means and covariances.

        Computes means and covariances only once, then stores and retrieves in
        subsequent calls.

        """
        ## Check if already computed
        if (not None in self.Means) and (not None in self.Covs):
            return self.Means, self.Covs

        dtype = torch.DoubleTensor if self.precision == 'double' else torch.FloatTensor

        shared_args = {'to_tensor': True, 'nworkers': self.nworkers_stats,
                       'device': self.device, 'online': self.online_stats,
                       'dtype': dtype,
                       'diagonal_cov': self.diagonal_cov}


        if (side=='both' or side == 'src') and (self.Means[0] is None):
            logger.info("Computing per-class means and variances D1....")
            M1, C1 = compute_label_stats(self.D1, self.targets1, self.idxs1,
                                         self.classes1, embedding=self.src_embedding,
                                         **shared_args)
            self.Means[0] = M1.cpu() # No reason to keep this in GPU, convert on the fly
            self.Covs[0]  = C1.cpu()

        if (side == 'both' or side =='tgt') and (self.Means[1] is None):
            if self.symmetric_tasks:
                M2, C2 = self.Means[0], self.Covs[0]
            else:
                logger.info("Computing per-class means and variances D2....")
                M2, C2 = compute_label_stats(self.D2, self.targets2, self.idxs2,
                                     self.classes2, embedding=self.tgt_embedding,
                                     **shared_args)
            self.Means[1] = M2.cpu()
            self.Covs[1]  = C2.cpu()

        return self.Means, self.Covs

    def _get_label_distances(self):
        """ Precompute label-to-label distances.

        Returns tensor of size nclasses_1 x nclasses_2.

        Useful when computing multiple distances on same pair of datasets
        e.g. between subsets of each datasets. Will store them in memory.

        Only useful if method=='precomputed_labeldist', for now.

        Note that _get_label_stats not called for inner_ot_method = `exact`,
        since exact computation does not use Gaussian approximation, so means
        and covariances are not needed.

        Returns:
            label_distances (torch.tensor): tensor of size (C1, C2) with pairwise
                label-to-label distances across the two datasets.

        """
        ## Check if already computed
        if not self.label_distances is None:
            return self.label_distances

        ## If not, compute from scratch
        if self.inner_ot_method == 'gaussian_approx':
            ## Instantiate call to pairwise wasserstein distance
            pwdist = partial(efficient_pwdist_gauss,
                             symmetric=self.symmetric_tasks,
                             diagonal_cov=self.diagonal_cov,
                             sqrt_method=self.sqrt_method,
                             sqrt_niters=self.sqrt_niters,
                             sqrt_pref  =self.sqrt_pref,
                             cost_function = self.feature_cost,
                             device=self.device,
                             return_dmeans=True,
                             return_sqrts=True)

            Means, Covs = self._get_label_stats()

        elif self.inner_ot_method == 'exact':
            ## In this case, need to load data *before* computing label stats.
            if (self.X1 is None) or (self.X2 is None):
                self._load_datasets(maxsamples=None)  # for now, will use *all* data, to be equiv  to Gaussian

            pwdist = partial(pwdist_exact,
                             symmetric=self.symmetric_tasks,
                             p = self.inner_ot_p,
                             loss = self.inner_ot_loss,
                             debias=self.inner_ot_debiased,
                             entreg = self.inner_ot_entreg,
                             cost_function = self.feature_cost,
                             device=self.device)

        elif self.inner_ot_method == 'naive_upperbound':
            pwdist = partial(pwdist_upperbound,
                             symmetric=self.symmetric_tasks,
                             diagonal_cov=self.diagonal_cov,
                             device=self.device,
                             return_dmeans=True)

            Means, Covs = self._get_label_stats()

        elif self.inner_ot_method == 'means_only':
            pwdist = partial(pwdist_means_only,
                             symmetric=self.symmetric_tasks,
                             device=self.device)

            Means, Covs = self._get_label_stats()

        else:
            raise ValueError()

        if self.debiased_loss and not self.symmetric_tasks:
            ## Then we also need within-collection label distances
            if self._pwlabel_stats_1 is None:
                logger.info('Pre-computing pairwise label Wasserstein distances D1 <-> D1...')
                if self.inner_ot_method == 'gaussian_approx':
                    DYY1, DYY1_means, sqrtΣ1 = pwdist(Means[0], Covs[0])
                elif self.inner_ot_method == 'naive_upperbound':
                    DYY1, DYY1_means  = pwdist(Means[0], Covs[0])
                elif self.inner_ot_method == 'means_only':
                    DYY1 = pwdist(Means[0])
                    DYY1_means = DYY1
                else: # Exact
                    if not isinstance(self.feature_cost, str):
                        self.feature_cost.sides = 'xx'
                    DYY1 = pwdist(self.X1, self.Y1)
            else:
                if self.inner_ot_method == 'gaussian_approx':
                    DYY1, DYY1_means, sqrtΣ1 = [self._pwlabel_stats_1[k] for k in ['dlabs','dmeans','sqrtΣ']]
                elif self.inner_ot_method in ['naive_upperbound', 'means_only']:
                    DYY1, DYY1_means = [self._pwlabel_stats_1[k] for k in ['dlabs','dmeans']]
                else:
                    DYY1 = self._pwlabel_stats_1['dlabs']

            if self._pwlabel_stats_2 is None:
                logger.info('Pre-computing pairwise label Wasserstein distances D2 <-> D2...')
                if self.inner_ot_method == 'gaussian_approx':
                    DYY2, DYY2_means, sqrtΣ2 = pwdist(Means[1], Covs[1])
                elif self.inner_ot_method == 'naive_upperbound':
                    DYY2, DYY2_means  = pwdist(Means[1], Covs[1])
                elif self.inner_ot_method == 'means_only':
                    DYY2 = pwdist(Means[1])
                    DYY2_means = DYY2
                else: # Exact
                    if not isinstance(self.feature_cost, str):
                        self.feature_cost.sides = 'yy'
                    DYY2 = pwdist(self.X2, self.Y2)
            else:
                logger.info('Found pre-existing D2 label-label stats, will not recompute')
                if self.inner_ot_method == 'gaussian_approx':
                    DYY2, DYY2_means, sqrtΣ2 = [self._pwlabel_stats_2[k] for k in ['dlabs','dmeans','sqrtΣ']]
                elif self.inner_ot_method in ['naive_upperbound', 'means_only']:
                    DYY1, DYY1_means = [self._pwlabel_stats_2[k] for k in ['dlabs','dmeans']]
                else:
                    DYY2 = self._pwlabel_stats_2['dlabs']
        else:
            sqrtΣ1, sqrtΣ2 = None, None  # Will have to compute during cross
            DYY1 = DYY2 = None
            DYY1_means = DYY2_means = None

        ## Compute Cross-Distances
        logger.info('Pre-computing pairwise label Wasserstein distances D1 <-> D2...')
        if self.inner_ot_method == 'gaussian_approx':
            DYY12, DYY12_means, _ = pwdist(Means[0], Covs[0], Means[1], Covs[1], sqrtΣ1, sqrtΣ2)
        elif self.inner_ot_method == 'naive_upperbound':
            DYY12, DYY12_means    = pwdist(Means[0], Covs[0], Means[1], Covs[1])
        elif self.inner_ot_method == 'means_only':
            DYY12    = pwdist(Means[0], Means[1])
            DYY12_means = DYY12
        else:
            if not isinstance(self.feature_cost, str):
                self.feature_cost.sides = 'xy'
            DYY12 = pwdist(self.X1,self.Y1,self.X2, self.Y2)
            DYY12_means = None


        if self.debiased_loss and self.symmetric_tasks:
            ## In this case we can reuse DXY to get DYY1 and DYY
            DYY1, DYY2 = DYY12, DYY12
            if self.inner_ot_method in ['gaussian_approx', 'naive_upperbound', 'means_only']:
                DYY1_means, DYY2_means = DXY_means, DXY_means

        if self.debiased_loss:
            D = torch.cat([torch.cat([DYY1, DYY12], 1),
                           torch.cat([DYY12.t(), DYY2], 1)], 0)
            if self.inner_ot_method in ['gaussian_approx', 'naive_upperbound', 'means_only']:
                D_means = torch.cat([torch.cat([DYY1_means, DYY12_means], 1),
                                 torch.cat([DYY12_means.t(), DYY2_means], 1)], 0)
        else:
            D = DYY12
            if self.inner_ot_method in ['gaussian_approx', 'naive_upperbound', 'means_only']:
                D_means = DYY12_means

        ## Collect and save
        self.label_distances  = D
        if self.inner_ot_method == 'gaussian_approx':
            self._label_mean_distances = D_means
            self._pwlabel_stats_1 = {'dlabs':DYY1, 'dmeans':DYY1_means, 'sqrtΣ':sqrtΣ1}
            self._pwlabel_stats_2 = {'dlabs':DYY2, 'dmeans':DYY2_means, 'sqrtΣ':sqrtΣ2}
        elif self.inner_ot_method  in ['naive_upperbound', 'means_only']:
            self._label_mean_distances = D_means
            self._pwlabel_stats_1 = {'dlabs':DYY1, 'dmeans':DYY1_means}#, 'sqrtΣ':sqrtΣ1}
            self._pwlabel_stats_2 = {'dlabs':DYY2, 'dmeans':DYY2_means}#, 'sqrtΣ':sqrtΣ2}
        else:
            self._pwlabel_stats_1 = {'dlabs':DYY1}#
            self._pwlabel_stats_2 = {'dlabs':DYY2}#, 'dmeans':DYY2_means, 'sqrtΣ':sqrtΣ2}


        return self.label_distances

    def distance(self, maxsamples=10000, return_coupling=False):
        """ Compute dataset distance.

            Note:
                Currently both methods require fully loading dataset into memory,
                this can probably be avoided, e.g., via subsampling.

            Arguments:
                maxsamples (int): maximum number of samples used in outer-level
                    OT problem. Note that this is different (and usually smaller)
                    than the number of samples used when computing means and covs.
                return_coupling (bool): whether to return the optimal coupling.

            Returns:
                dist (float): the optimal transport dataset distance value.
                π (tensor, optional): the optimal transport coupling.

        """
        device_dists = self.device
        GPU_LIMIT = 10000
        if (self.n1 > GPU_LIMIT or self.n2 > GPU_LIMIT) and maxsamples > GPU_LIMIT and self.device != 'cpu':
            logger.warning('Warning: maxsamples = {} > 5000, and device = {}. Loaded data' \
                   ' might not fit in GPU. Computing distances on' \
                   ' CPU.'.format(maxsamples, self.device))
            device_dists = 'cpu'
        if self.X1 is None or self.X2 is None:
            if (not self.method == 'jdot') and (self.λ_y is not None) and (self.λ_y > 0):
                s = time()
                _ = self._get_label_distances()
                logger.info('/* Time to precompute label distances: {} */'.format(time() - s))

            self._load_datasets(maxsamples, device=device_dists)
        if self.method == 'augmentation':
            DA = (self.X1, self.Y1)
            DB = (self.X2, self.Y2)

            if self.λ_x != 1.0 or self.λ_y != 1.0:
                raise NotImplementedError('Unevenly weighted feature/label' \
                    'not available for method=augmentation yet')

            if not hasattr(self, 'XμΣ1') or self.XμΣ1 is None:
                XA = augmented_dataset(DA, self.Means[0], self.Covs[0], maxn=maxsamples)#, diagonal_cov=self.diagonal_cov)
                del DA
                XB = augmented_dataset(DB, self.Means[1], self.Covs[1], maxn=maxsamples)#, diagonal_cov=self.diagonal_cov)
                del DB
                self.XμΣ1 = XA
                self.XμΣ2 = XB
            else:
                XA, XB = self.XμΣ1, self.XμΣ2

            loss = geomloss.SamplesLoss(
                loss=self.loss, p=self.p,
                debias=self.debiased_loss,
                blur=self.entreg**(1 / self.p), # "blur" of geomloss is eps^(1/p).
                backend='tensorized',
                )
            ## By default, use constant weights = 1/number of samples
            dist = loss(XA, XB)
            del XA, XB
        elif self.method == 'jdot':
            loss = geomloss.SamplesLoss(
                loss=self.loss, p=self.p,
                cost=partial(batch_jdot_cost, alpha = self.λ_x),
                debias=self.debiased_loss,
                blur=self.entreg**(1 / self.p),
                backend='tensorized'
            )
            if maxsamples and self.X1.shape[0] > maxsamples:
                idxs_1 = sorted(np.random.choice(
                    self.X1.shape[0], maxsamples, replace=False))
            else:
                idxs_1 = np.s_[:]  # hack to get a full slice

            if maxsamples and self.X2.shape[0] > maxsamples:
                idxs_2 = sorted(np.random.choice(
                    self.X2.shape[0], maxsamples, replace=False))
            else:
                idxs_2 = np.s_[:]  # hack to get a full slice


            Z1 = torch.cat((self.X1[idxs_1],
                            self.Y1[idxs_1].type(self.X1.dtype).unsqueeze(1)), -1)
            Z2 = torch.cat((self.X2[idxs_2],
                            self.Y2[idxs_2].type(self.X2.dtype).unsqueeze(1)), -1)
            Z1 = Z1.to(device_dists)
            Z2 = Z2.to(device_dists)
            dist = loss(Z1,Z2)
        elif self.method == 'precomputed_labeldist':
            if self.λ_y is None or self.λ_y == 0:
                W = None
            else:
                W = self._get_label_distances().to(torch.device(device_dists))

            ## This one leverages precomputed pairwise label distances
            cost_geomloss = partial(
                batch_augmented_cost,
                W=W,
                λ_x=self.λ_x,
                λ_y=self.λ_y,
                feature_cost=self.feature_cost
            )

            loss = geomloss.SamplesLoss(
                loss=self.loss, p=self.p,
                cost=cost_geomloss,
                debias=self.debiased_loss,
                blur=self.entreg**(1 / self.p),
                backend='tensorized'
            )

            if maxsamples and self.X1.shape[0] > maxsamples:
                idxs_1 = sorted(np.random.choice(
                    self.X1.shape[0], maxsamples, replace=False))
            else:
                idxs_1 = np.s_[:]  # hack to get a full slice

            if maxsamples and self.X2.shape[0] > maxsamples:
                idxs_2 = sorted(np.random.choice(
                    self.X2.shape[0], maxsamples, replace=False))
            else:
                idxs_2 = np.s_[:]  # hack to get a full slice


            Z1 = torch.cat((self.X1[idxs_1],
                            self.Y1[idxs_1].type(self.X1.dtype).unsqueeze(1)), -1)
            Z2 = torch.cat((self.X2[idxs_2],
                            self.Y2[idxs_2].type(self.X2.dtype).unsqueeze(1)), -1)
            Z1 = Z1.to(device_dists)
            Z2 = Z2.to(device_dists)
            dist = loss(Z1,Z2)


            if return_coupling:
                with torch.no_grad():
                    C = cost_geomloss(Z1.unsqueeze(0), Z2.unsqueeze(0)).squeeze()
                    if self.coupling_method == 'geomloss':
                        loss.potentials = True
                        u, v = loss(Z1, Z2)
                        π = torch.exp(1 / self.entreg * (u.t() + v - C))  # * (pq)
                    elif self.coupling_method == 'pot':
                        C = C.cpu()
                        π = ot.sinkhorn(ot.unif(Z1.shape[0]), ot.unif(Z2.shape[0]),
                                        C / C.max(), self.entreg, numItermax=50,
                                        method='sinkhorn_epsilon_scaling', verbose=True)
                    else:
                        pass  # nonadimisslbe args already caught in argparse
                    del C
            del Z1, Z2


        ## Admittedly ugly but might be necessary to avoid memory clogs
        torch.cuda.empty_cache()

        if return_coupling:
            return dist, π
        else:
            return dist

    def compute_coupling(self, entreg=None, gpu=None, **kwargs):
        """ Compute the optimal transport coupling.

        Arguments:
            entreg (float): strength of entropy regularization.
            gpu (bool): whether to use gpu for coupling computation.
            **kwargs: arbitrary keyword args passed to ot.sinkhorn

        Returns:
            π (tensor): tensor of size (N1, N2) with optimal transport coupling.

        """
        if self.X1 is None or self.X2 is None:
            self._load_datasets()
        entreg = entreg if entreg else self.entreg
        Z1 = torch.cat((self.X1, self.Y1.type(self.X1.dtype).unsqueeze(1)), -1)
        Z2 = torch.cat((self.X2, self.Y2.type(self.X2.dtype).unsqueeze(1)), -1)
        ## Compute on device of Z1, Z1. If cuda is available but Z1,Z2 are in cpu
        ## it was decided in .distance() that they're too large for GPU.
        device = Z1.device
        a = ot.unif(Z1.shape[0])
        b = ot.unif(Z2.shape[0])
        W = self._get_label_distances().to(device)

        C = batch_augmented_cost(Z1.unsqueeze(0), Z2.unsqueeze(0),W=W).squeeze()
        C = C.cpu()
        if gpu is None:
            gpu = self.device != 'cpu'
        if 'method' in kwargs and kwargs['method'] == 'emd':
            π = ot.emd(a, b, C / C.max())
        elif not gpu:
            π = ot.sinkhorn(a, b, C / C.max(), entreg, **kwargs)
        else:
            kwargs['verbose'] = False
            π = ot.gpu.sinkhorn(a, b, C / C.max(), entreg, **kwargs)
        self.π = π
        return π

    def final_distance(self):
        """ Computes the outer-level OT distance between the datasets.
        FIXME: This function seems to be deprecated.
        """
        α = ot.utils.unif(self.n1)
        β = ot.utils.unif(self.n2)
        if normalize_dists == 'max':
            D_norm = D / D.max()
        else:
            D_norm = D
        π, log = ot.bregman.sinkhorn(
            α, β, D_norm, reg=reg, method='sinkhorn', log=True)
        d = (π * D).sum()
        return d, π

    def subgroup_distance(self, labels_a, labels_b, maxsamples=500):
        """
        Compute dataset distance between subsets of the two datasets, where
        the subsets are defined through subroups of labels.

        """
        mask_a = np.isin(self.Y1.cpu(), labels_a)
        idxs_a = mask_a.nonzero()[0].squeeze()
        mask_b = np.isin(self.Y2.cpu(), labels_b)
        idxs_b = mask_b.nonzero()[0].squeeze()

        if self.method == 'augmentation':
            DA = (self.X1[mask_a], self.Y1[mask_a])
            DB = (self.X2[mask_b], self.Y2[mask_b])
            XA = augmented_dataset(
                DA, self.Means[0], self.Covs[0], maxn=maxsamples)
            del DA
            XB = augmented_dataset(
                DB, self.Means[1], self.Covs[1], maxn=maxsamples)
            del DB
            pdb.set_trace()
            loss = geomloss.SamplesLoss(
                loss="sinkhorn", p=2, blur=self.entreg**(1 / self.p))
            ## By default, use constant weights = 1/number of samples
            dist = loss(XA, XB)
            del XA, XB
        elif self.method == 'precomputed_labeldist':
            ## This one leverages precomputed pairwise label distances
            cost_geomloss = partial(batch_augmented_cost,
                                    W=self._get_label_distances(),
                                    V1=len(self.V1),
                                    V2=len(self.V2))

            loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, cost=cost_geomloss,
                                        blur=self.entreg**(1 / self.p), backend='tensorized')

            if maxsamples and mask_a.sum() > maxsamples:
                idxs_a = idxs_a[sorted(np.random.choice(
                    len(idxs_a), maxsamples, replace=False))]
            if maxsamples and mask_b.sum() > maxsamples:
                idxs_b = idxs_b[sorted(np.random.choice(
                    len(idxs_b), maxsamples, replace=False))]
            ZA = torch.cat(
                (self.X1[idxs_a], self.Y1[idxs_a].float().unsqueeze(1)), -1)
            ZB = torch.cat(
                (self.X2[idxs_b], self.Y2[idxs_b].float().unsqueeze(1)), -1)
            dist = loss(ZA, ZB)
            del ZA, ZB

        torch.cuda.empty_cache()

        return dist

    def plot_label_distances(self, plot_means=False, ax=None, show=True, cbar=True,
                             cmap="YlGn", cbarlabel="default", save_path=None, xlabel=None, ylabel=None,
                             fontsize=10,
                             **kwargs):
        LD = self._get_label_distances().sqrt()
        LMD = self._label_mean_distances.sqrt()

        if LD.shape[0] > len(self.V1):
            ## Means we also have self, distance, don't want to plot those usually
            LD = LD[:len(self.V1), len(self.V1):]
            LMD = LMD[:len(self.V1), len(self.V1):]

        if not ax:
            ncol = 2 if plot_means else 1
            fig, ax = plt.subplots(1, ncol, figsize=(ncol * 5, 5))
        elif type(ax) is np.ndarray:
            assert len(ax) == 2
            ncol = 2
        else:
            assert not plot_means
            ncol = 1

        ax0 = ax[0] if plot_means else ax
        if cbarlabel == 'default': cbarlabel = r"Wasserstein Distance $d(y,y')$"
        ax0.set_title('Label-to-Label Distance',fontsize=fontsize)
        im, cbar = heatmap(LD.cpu(), self.classes1, self.classes2, ax=ax0,
                           cmap=cmap, cbar=cbar, cbarlabel=cbarlabel, **kwargs)
        if plot_means:
            ax[1].set_title('Label-to-label Distance')
            im, cbar = heatmap(LMD.cpu(), self.V1.tolist(), self.V2.tolist(), ax=ax[1],
                               cmap=cmap, cbar=cbar, cbarlabel=cbarlabel, **kwargs)
            
        if xlabel: ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel: ax.set_ylabel(ylabel, fontsize=fontsize)
        if save_path:
            fig = plt.gcf()
            fig.tight_layout()
            plt.savefig(save_path, format='pdf', dpi=300)
        if show:
            plt.show()

    def plot_label_stats(self, same_plot=False, show_target = True,
                         label_groups=True, label_means=True,
                         color_by='domain',
                         pad=0.1, ax=None, show=True, shift=(1,1)):
        ## Assert that both datasets are 2 dim
        Means, Covs = self._get_label_stats()
        if self.X1 is None or self.X2 is None:
            self._load_datasets()

        k1,k2 = len(self.classes1), len(self.classes2)

        if ax is None:
            ncol = 1 if same_plot else 2
            fig, ax = plt.subplots(1, ncol, figsize=(14, 7))
        elif type(ax) is np.ndarray and len(ax) == 2:
            assert not same_plot
            ncol = 2
        else:
            assert same_plot
            ncol = 1

        color_by = 'label'
        if color_by == 'domain':
            ## Only two colours, don't distinguish between label groups
            ## These are for the density plots
            colors = ['red', 'blue']
            cmaps  = [cm.Reds, cm.Blues]

            ## These are for the scatter plots
            scatter_colors = [self.Y1-self.Y1.min(),self.Y2-self.Y2.min()]
            scatter_cmaps = [
                mpl.colors.ListedColormap(cm.get_cmap('Reds', 2*k1)(np.linspace(0.4,0.8,k1))),
                mpl.colors.ListedColormap(cm.get_cmap('Blues', 2*k2)(np.linspace(0.4,0.8,k2)))
            ]
            markers = ['o', 'o']

        elif color_by == 'label':
            ## These are for the density plots
            colors = ['red', 'blue']
            cmaps = [cm.Reds, cm.Blues]

            ## These are for the scatter plots
            scatter_colors = [self.Y1-self.Y1.min(),self.Y2-self.Y2.min()]
            scatter_cmaps = [cm.get_cmap('tab10', k1),cm.get_cmap('tab10', k2)]
            markers = ['*', 'o']

        else:
            raise ValueError('Unrecognized value')

        ## Set plot limits
        lims = {'x': [None, None], 'y': [None, None]}
        padx1 = (self.X1[:, 0].max() - self.X1[:, 0].min()) * pad
        pady1 = (self.X1[:, 1].max() - self.X1[:, 1].min()) * pad
        padx2 = (self.X2[:, 0].max() - self.X2[:, 0].min()) * pad
        pady2 = (self.X2[:, 1].max() - self.X2[:, 1].min()) * pad

        lims['x'][0] = (self.X1[:, 0].min() - padx1,
                        self.X1[:, 0].max() + padx1)
        lims['y'][0] = (self.X1[:, 1].min() - pady1,
                        self.X1[:, 1].max() + pady1)
        lims['x'][1] = (self.X2[:, 0].min() - padx2,
                        self.X2[:, 0].max() + padx2)
        lims['y'][1] = (self.X2[:, 1].min() - pady2,
                        self.X2[:, 1].max() + pady2)

        ## Maybe repeated calls to single gaussian_distrib_plot, change colors
        for i in range(2):
            if i == 1 and not show_target: continue
            X, Y, c = (self.X1, self.Y1, self.classes1) if i == 0 else (
                self.X2, self.Y2, self.classes2)
            X = X.clone().detach() # In case we're in dynamic setting
            Y = Y.clone().detach()
            axi = ax[i] if ncol == 2 else ax
            if same_plot:
                axi.set_xlim(min(lims['x'][0][0], lims['x'][1][0]),
                             max(lims['x'][0][1], lims['x'][1][1]))
                axi.set_ylim(min(lims['y'][0][0], lims['y'][1][0]),
                             max(lims['y'][0][1], lims['y'][1][1]))
            else:
                axi.set_xlim(lims['x'][i])
                axi.set_ylim(lims['y'][i])
            for j, (μ, Σ) in enumerate(zip(Means[i], Covs[i])):
                μi, Σi = μ.clone().detach(), Σ.clone().detach()
                if Σ.ndim == 1: Σi = torch.diag(Σi)
                try:
                    P = MultivariateNormal(μi, Σi)
                    gaussian_density_plot(P, X=X[Y == j], method='exact',
                                      nsamples=100, label_means=label_means,
                                      color=colors[i], cmap=cmaps[i], ax=axi)
                except:
                    logger.warning('Gaussian density plot failed - probably singular covariance')
                axi.scatter(X[:, 0], X[:, 1], marker=markers[i], s=8,
                            c=scatter_colors[i], cmap=scatter_cmaps[i])#, normalize = )

                if label_groups:
                    axi.text(μ[0] + shift[0], μ[1] + shift[1], r"$y={}$".format(c[j]))
            axi.set_xlabel('')
            axi.set_ylabel('')

        if show:
            plt.show()

        return ax

    @staticmethod
    def plot_coupling(pi, Y1, Y2, ax=None, boundaries=None, sorting=False, title=None,
                      ticks=False, xlabel=None, ylabel=None, axlabel_fontsize=10,
                      show=True, save_path=None):
        Y1 -= Y1.min()
        Y2 -= Y2.min()
        if not ax:
            maxn = max(len(Y1), len(Y2))
            fig, ax = plt.subplots(
                figsize=(5 * len(Y1) / maxn, 5 * len(Y2) / maxn))

        if sorting:
            if type(sorting) is bool:
                _,idxs1 = zip(*sorted(zip(Y1.numpy(),range(len(Y1)))))
                _,idxs2 = zip(*sorted(zip(Y2.numpy(),range(len(Y2)))))
                pi = pi[idxs1, :][:, idxs2]
            elif type(sorting) is tuple:
                pi = pi[sorting[0], :][:, sorting[1]]

        ax.imshow(pi,  cmap='Reds', aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])

        if boundaries:
            src_group_sizes = torch.bincount(Y1)
            trg_group_sizes = torch.bincount(Y2)
            gnames = [range(10), range(10)]
            imshow_group_boundaries(ax, src_group_sizes, trg_group_sizes,
                                    group_names=gnames)

        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ytick_spacing = max(round(pi.shape[0] / 20), 1)
            xtick_spacing = max(round(pi.shape[1] / 20), 1)
            rownames = Y1
            colnames = Y2
            ax.set_xticks(range(len(colnames))[::xtick_spacing])
            ax.set_xticklabels([str(a) for a in colnames[::xtick_spacing]])
            ax.set_yticks(range(len(rownames))[::ytick_spacing])
            ax.set_yticklabels([str(a) for a in rownames[::ytick_spacing]])

        if xlabel:
            ax.set_xlabel(r'Dataset: {}'.format(xlabel), fontsize=axlabel_fontsize)
        if ylabel:
            ax.set_ylabel(r'Dataset: {}'.format(ylabel), fontsize=axlabel_fontsize)
        if title:
            ax.set_title(title, pad=5 + 20 * (boundaries))
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300)
        if show:
            plt.show()


class IncomparableDatasetDistance(DatasetDistance):
    """ Dataset Distance subclass for datasets that have different feature dimension.

    Note:
        Proceed with caution, this class is still experimental and in active
        development

    """
    def __init__(self, *args, **kwargs):
        super(IncomparableDatasetDistance, self).__init__(*args, **kwargs)
        if self.debiased_loss:
            raise ValueError('Debiased GWOTDD not implemented yet')

    def _get_label_distances(self):
        """
            TODO: We could instead modify method in parent class to allow for only
            within-domain label distance computation.
        """
        Means, Covs = self._get_label_stats()

        ## Instantiate call to pairwise wasserstein distance
        pwdist = partial(efficient_pwdist_gauss,
                         symmetric=self.symmetric_tasks,
                         diagonal_cov=self.diagonal_cov,
                         sqrt_method=self.sqrt_method,
                         sqrt_niters=self.sqrt_niters,
                         sqrt_pref  =self.sqrt_pref,
                         device=self.device,
                         return_dmeans=True,
                         return_sqrts=True)


        if not self._pwlabel_stats_1 is None:
            logger.info('Found pre-existing D1 label-label stats, will not recompute')
            DYY1, DYY1_means, sqrtΣ1 = [self._pwlabel_stats_1[k] for k in ['dlabs','dmeans','sqrtΣ']]
        else:
            logger.info('Pre-computing pairwise label Wasserstein distances D1 <-> D1...')
            DYY1, DYY1_means, sqrtΣ1 = pwdist(Means[0], Covs[0])

        if not self._pwlabel_stats_2 is None:
            logger.info('Found pre-existing D2 label-label stats, will not recompute')
            DYY2, DYY2_means, sqrtΣ2 = [self._pwlabel_stats_2[k] for k in ['dlabs','dmeans','sqrtΣ']]
        else:
            logger.info('Pre-computing pairwise label Wasserstein distances D1 <-> D1...')
            DYY2, DYY2_means, sqrtΣ2 = pwdist(Means[1], Covs[1])


        self._pwlabel_stats_1 = {'dlabs':DYY1, 'dmeans':DYY1_means, 'sqrtΣ':sqrtΣ1}
        self._pwlabel_stats_2 = {'dlabs':DYY2, 'dmeans':DYY2_means, 'sqrtΣ':sqrtΣ2}

        return DYY1, DYY2


    def _compute_intraspace_distances(self):
        if self.X1 is None or self.X2 is None:
            self._load_datasets()
        DYY1, DYY2 = self._get_label_distances()

        Z1 = torch.cat((self.X1, self.Y1.type(self.X1.dtype).unsqueeze(1)), -1)
        Z1 = Z1.to(self.device)
        C1 = batch_augmented_cost(Z1.unsqueeze(0), Z1.unsqueeze(0), W=DYY1).squeeze()

        Z2 = torch.cat((self.X2, self.Y2.type(self.X2.dtype).unsqueeze(1)), -1)
        Z2 = Z2.to(self.device)
        C2 = batch_augmented_cost(Z2.unsqueeze(0), Z2.unsqueeze(0), W=DYY2).squeeze()

        return C1, C2

    def distance(self, maxsamples=10000, return_coupling=False):
        C1, C2 = self._compute_intraspace_distances()
        a = torch.ones(self.X1.shape[0]).to(self.device)/self.X1.shape[0]
        b = torch.ones(self.X2.shape[0]).to(self.device)/self.X2.shape[0]

        ## Normalize distances
        C1 = (C1 - C1.min())/C1.max()
        C2 = (C2 - C2.min())/C2.max()

        π, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, a, b,
                                loss_fun = 'square_loss', epsilon=self.entreg,
                                log=True, verbose=True)
        dist = log['gw_dist']

        if return_coupling:
            return dist, π
        else:
            return dist


class FeatureCost():
    """ Class implementing a cost (or distance) between feature vectors.

    Arguments:
        p (int): the coefficient in the OT cost (i.e., the p in p-Wasserstein).
        src_embedding (callable, optional): if provided, source data will be
            embedded using this function prior to distance computation.
        tgt_embedding (callable, optional): if provided, target data will be
            embedded using this function prior to distance computation.

    """
    def __init__(self, src_embedding=None, tgt_embedding=None, src_dim=None,
                 tgt_dim=None, p=2, device='cpu'):
        assert (src_embedding is None) or (src_dim is not None)
        assert (tgt_embedding is None) or (tgt_dim is not None)
        self.src_emb = src_embedding
        self.tgt_emb = tgt_embedding
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.p = p
        self.device = device
        self.sides = None

    def _get_batch_shape(self, b):
        if b.ndim == 3: return b.shape
        elif b.ndim == 2: return (1,*b.shape)
        elif b.ndim == 1: return (1,1,b.shape[0])

    def _batchify_computation(self, X, side='x', slices=250):
        slices = len(X)
        if side == 'x':
            out = torch.cat([self.src_emb(b).to('cpu') for b in torch.chunk(X, slices, dim=0)])
        else:
            out = torch.cat([self.tgt_emb(b).to('cpu') for b in torch.chunk(X, slices, dim=0)])
        return out.to(self.device)

    def __call__(self, X1, X2):
        _orig_device = X1.device
        device = process_device_arg(self.device)
        if self.src_emb is not None:
            B1, N1, D1 = self._get_batch_shape(X1)
            dim = self.src_dim if self.sides[0]  == 'x' else self.tgt_dim
            emb = self.src_emb if self.sides[0]  == 'x' else self.tgt_emb
            try:
                X1 = emb(X1.view(-1,*dim).to(self.device)).reshape(B1, N1, -1)
            except: # Memory error?
                print('Batchifying feature distance computation')
                X1 = self._batchify_computation(X1.view(-1,*dim).to(self.device), side=self.sides[0]).reshape(B1, N1, -1)
        if self.tgt_emb is not None:
            B2, N2, D2 = self._get_batch_shape(X2)
            emb = self.src_emb if self.sides[1]  == 'x' else self.tgt_emb
            dim = self.src_dim if self.sides[1]  == 'x' else self.tgt_dim
            try:
                X2 = emb(X2.view(-1,*dim).to(self.device)).reshape(B2, N2, -1)
            except:
                print('Batchifying feature distance computation')
                X2 = self._batchify_computation(X2.view(-1,*dim).to(self.device), self.sides[1]).reshape(B2, N2, -1)
        if self.p == 1:
            c = geomloss.utils.distances(X1, X2)
        elif self.p == 2:
            c = geomloss.utils.squared_distances(X1, X2) / 2
        else:
            raise ValueError()
        return c.to(_orig_device)


def batch_jdot_cost(Z1, Z2, p=2, alpha=1.0, feature_cost=None):
    " https://papers.nips.cc/paper/6963-joint-distribution-optimal-transportation-for-domain-adaptation.pdf"
    B, N, D1 = Z1.shape
    B, M, D2 = Z2.shape
    assert (D1 == D2) or (feature_cost is not None)
    Y1 = Z1[:, :, -1].long()
    Y2 = Z2[:, :, -1].long()
    if feature_cost is None or feature_cost == 'euclidean': # default is euclidean
        C1 = cost_routines[p](Z1[:, :, :-1], Z2[:, :, :-1])
    else:
        C1 = feature_cost(Z1[:, :, :-1], Z2[:, :, :-1])
    ## hinge loss assumes classes and indices are same for both - shift back to [0,K]
    C2 = multiclass_hinge_loss(Y1.squeeze()-Y1.min(), Y2.squeeze()-Y2.min()).reshape(B, N, M)
    return alpha*C1 + C2


def batch_augmented_cost(Z1, Z2, W=None, Means=None, Covs=None, feature_cost=None,
                         p=2, λ_x=1.0, λ_y=1.0):
    """ Batch ground cost computation on augmented datasets.

    Defines a cost function on augmented feature-label samples to be passed to
    geomloss' samples_loss. Geomloss' expected inputs determine the requirtements
    below.

    Args:
        Z1 (torch.tensor): torch Tensor of size (B,N,D1), where last position in
            last dim corresponds to label Y.
        Z2 (torch.tensor): torch Tensor of size (B,M,D2), where last position in
            last dim corresponds to label Y.
        W (torch.tensor): torch Tensor of size (V1,V2) of precomputed pairwise
            label distances for all labels V1,V2 and returns a batched cost
            matrix as a (B,N,M) Tensor. W is expected to be congruent with p.
            I.e, if p=2, W[i,j] should be squared Wasserstein distance.
        Means (torch.tensor, optional): torch Tensor of size (C1, D1) with per-
            class mean vectors.
        Covs (torch.tensor, optional): torch Tehsor of size (C2, D2, D2) with
            per-class covariance matrices
        feature_cost (string or callable, optional): if None or 'euclidean', uses
            euclidean distances as feature metric, otherwise uses this function
            as metric.
        p (int): order of Wasserstein distance.
        λ_x (float): weight parameter for feature component of distance
        λ_y (float): weight parameter for label component of distance

    Returns:
        D (torch.tensor): torch Tensor of size (B,N,M)

    Raises:
        ValueError: If neither W nor (Means, Covs) are provided.

    """

    B, N, D1 = Z1.shape
    B, M, D2 = Z2.shape
    assert (D1 == D2) or (feature_cost is not None)

    Y1 = Z1[:, :, -1].long()
    Y2 = Z2[:, :, -1].long()

    ## Compute distances between features X in the usual way
    if λ_x is None or λ_x == 0:
        ## Features ignored in d(z,z'), C1 is dummy
        logger.info('no d_x')
        C1 = torch.zeros(B,N,M)
    elif feature_cost is None or feature_cost == 'euclidean': # default is euclidean
        C1 = cost_routines[p](Z1[:, :, :-1], Z2[:, :, :-1])  # Get from GeomLoss
    else:
        C1 = feature_cost(Z1[:, :, :-1], Z2[:, :, :-1])

    if λ_y is None or λ_y == 0:
        ## Labels ignored in d(z,z'), C2 is dummy
        logger.info('no d_y')
        C2 = torch.zeros_like(C1)
        λ_y = 0.0
    elif W is not None:
        ## Label-to-label distances have been precomputed and passed
        ## Stores flattened index corresponoding to label pairs
        M = W.shape[1] * Y1[:, :, None] + Y2[:, None, :]
        C2 = W.flatten()[M.flatten(start_dim=1)].reshape(-1,Y1.shape[1], Y2.shape[1])
    elif Means is not None and Covs is not None:
        ## We need to compate label distances too
        dmeans = cost_routines[p](Means[0][Y1.squeeze()], Means[1][Y2.squeeze()])
        dcovs  = torch.zeros_like(dmeans)
        pdb.set_trace("TODO: need to finish this. But will we ever use it?")
    else:
        raise ValueError("Must provide either label distances or Means+Covs")

    assert C1.shape == C2.shape

    ## NOTE: geomloss's cost_routines as defined above already divide by p. We do
    ## so here too for consistency. But as a consequence, need to divide C2 by p too.
    D = λ_x * C1  +  λ_y * (C2/p)

    return D
