"""
    Tools for moment (mean/cov) computation needed by OTTD and other routines.
"""

import logging
import pdb

import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data.sampler import SubsetRandomSampler

from .utils import process_device_arg, extract_data_targets

logger = logging.getLogger(__name__)


def cov(m, mean=None, rowvar=True, inplace=False):
    """ Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Arguments:
        m (tensor): A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar (bool): If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    if mean is None:
        mean = torch.mean(m, dim=1, keepdim=True)
    else:
        mean = mean.unsqueeze(1) # For broadcasting
    if inplace:
        m -= mean
    else:
        m = m - mean
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

class OnlineStatsRecorder:
    """ Online batch estimation of multivariate sample mean and covariance matrix.

    Alleviates numerical instability due to catastrophic cancellation that
    the naive estimation suffers from.

    Two pass approach first computes population mean, and then uses stable
    one pass algorithm on residuals x' = (x - μ). Uses the fact that Cov is
    translation invariant, and less cancellation happens if E[XX'] and
    E[X]E[X]' are far apart, which is the case for centered data.

    Ideas from:
        - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        - https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    """
    def __init__(self, data=None, twopass=True, centered_cov=False,
                diagonal_cov=False, embedding=None,
                device='cpu', dtype=torch.FloatTensor):
        """
        Arguments:
            data (torch tensor): batch of data of shape (nobservations, ndimensions)
            twopass  (bool): whether two use the two-pass approach (recommended)
            centered_cov (bool): whether covariance matrix is centered throughout
                                 the iterations. If false, centering happens once,
                                 at the end.
            diagonal_cov (bool): whether covariance matrix should be diagonal
                                 (i.e. ignore cross-correlation terms). In this
                                 case only diagonal (1xdim) tensor retrieved.
            embedding (callable): if provided, will map features using this
            device (str): device for storage of computed statistics
            dtype (torch data type): data type for computed statistics

        """
        self.device = device
        self.centered_cov = centered_cov
        self.diagonal_cov = diagonal_cov
        self.twopass      = twopass
        self.dtype = dtype
        self.embedding = embedding

        self._init_values()

    def _init_values(self):
        self.μ = None
        self.Σ = None
        self.n = 0

    def compute_from_loader(self, dataloader):
        """ Compute statistics from dataloader """
        device = process_device_arg(self.device)
        for x, _ in dataloader:
            x = x.type(self.dtype).to(device)
            x = self.embedding(x).detach() if self.embedding is not None else x
            self.update(x.view(x.shape[0], -1))
        μ, Σ = self.retrieve()
        if self.twopass:
            self._init_values()
            self.centered_cov = False
            for x, _ in dataloader:
                x = x.type(self.dtype).to(device)
                x = self.embedding(x).detach() if self.embedding is not None else x
                self.update(x.view(x.shape[0],-1)-μ) # We compute cov on residuals
            _, Σ = self.retrieve()
        return μ, Σ

    def update(self, batch):
        """ Update statistics using batch of data.

        Arguments:
            data (tensor): tensor of shape (nobservations, ndimensions)
        """
        if self.n == 0:
            self.n,self.d = batch.shape
            self.μ = batch.mean(axis=0)
            if self.diagonal_cov and self.centered_cov:
                self.Σ  = torch.var(batch, axis=0, unbiased=True)
                ## unbiased is default in pytorch, shown here just to be explicit
            elif self.diagonal_cov and not self.centered_cov:
                self.Σ  = batch.pow(2).sum(axis=0)/(1.0*self.n-1)
            elif self.centered_cov:
                self.Σ = ((batch-self.μ).T).matmul(batch-self.μ)/(1.0*self.n-1)
            else:
                self.Σ = (batch.T).matmul(batch)/(1.0*self.n-1)
                ## note that this not really covariance yet (not centered)
        else:
            if batch.shape[1] != self.d:
                raise ValueError("Data dims don't match prev observations.")

            ### Dimensions
            m = self.n * 1.0
            n = batch.shape[0] *1.0

            ### Mean Update
            self.μ =  self.μ + (batch-self.μ).sum(axis=0)/(m+n)   # Stable Algo

            ### Cov Update
            if self.diagonal_cov and self.centered_cov:
                self.Σ = ((m-1)*self.Σ + ((m-1)/(m+n-1))*((batch-self.μ).pow(2).sum(axis=0)))/(m+n-1)
            elif self.diagonal_cov and not self.centered_cov:
                self.Σ = (m-1)/(m+n-1)*self.Σ + 1/(m+n-1)*(batch.pow(2).sum(axis=0))
            elif self.centered_cov:
                self.Σ = ((m-1)*self.Σ + ((m-1)/(m+n-1))*((batch-self.μ).T).matmul(batch-self.μ))/(m+n-1)
            else:
                self.Σ = (m-1)/(m+n-1)*self.Σ + 1/(m+n-1)*(batch.T).matmul(batch)

            ### Update total number of examples seen
            self.n += n

    def retrieve(self, verbose=False):
        """ Retrieve current statistics """
        if verbose: print('Mean and Covariance computed on {} samples'.format(int(self.n)))
        if self.centered_cov:
            return self.μ, self.Σ
        elif self.diagonal_cov:
            Σ = self.Σ - self.μ.pow(2)*self.n/(self.n-1)
            Σ = torch.nn.functional.relu(Σ) # To avoid negative variances due to rounding
            return self.μ, Σ
        else:
            return self.μ, self.Σ - torch.ger(self.μ.T,self.μ)*self.n/(self.n-1)


def _single_label_stats(data, i, c, label_indices, M=None, S=None, batch_size=256,
                        embedding=None, online=True, diagonal_cov=False,
                        dtype=None, device=None):
    """ Computes mean/covariance of examples that have a given label. Note that
    classname c is only needed for vanity printing. Device info needed here since
    dataloaders are used inside.

    Arguments:
        data (pytorch Dataset or Dataloader): data to compute stats on
        i (int): index of label (a.k.a class) to filter
        c (int/str): value of label (a.k.a class) to filter

    Returns:
        μ (torch tensor): empirical mean of samples with given label
        Σ (torch tensor): empirical covariance of samples with given label
        n (int): number of samples with giben label

    """
    device = process_device_arg(device)
    if len(label_indices) < 2:
        logger.warning(" -- Class '{:10}' has too few examples ({})." \
              " Ignoring it.".format(c, len(label_indices)))
        if M is None:
            return None,None,len(label_indices)
    else:
        if type(data) == dataloader.DataLoader:
            ## We'll reuse the provided dataloader, just setting indices.
            ## If loader had indices before, we restore them when we're done
            filtered_loader = data
            if hasattr(data.sampler,'indices'):
                _orig_indices = data.sampler.indices
            else:
                _orig_indices = None
            filtered_loader.sampler.indices = label_indices

        else:
            ## Create our own loader
            filtered_loader = dataloader.DataLoader(data, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(label_indices))
            _orig_indices = None

        if online:
            ## Will compute online (i.e. without loading all the data at once)
            stats_rec = OnlineStatsRecorder(centered_cov=True, twopass=True,
                                            diagonal_cov=diagonal_cov, device=device,
                                            embedding=embedding,
                                            dtype=dtype)
            μ, Σ = stats_rec.compute_from_loader(filtered_loader)

            n = int(stats_rec.n)
        else:
            X = torch.cat([d[0].to(device) for d in filtered_loader]).squeeze()
            X = embedding(X) if embedding is not None else X
            μ = torch.mean(X, dim = 0).flatten()
            if diagonal_cov:
                Σ = torch.var(X, dim=0).flatten()
            else:
                Σ = cov(X.view(X.shape[0], -1).t())
            n = X.shape[0]
        logger.info(' -> class {:10} (id {:2}): {} examples'.format(c, i, n))

        if diagonal_cov:
            try:
                assert Σ.min() >= 0
            except:
                pdb.set_trace()

        ## Reinstante original indices in sampler
        if _orig_indices is not None: data.sampler.indices = _orig_indices

        if M is not None:
            M[i],S[i] = μ.cpu(),Σ.cpu() # To avoid GPU parallelism problems
        else:
            return μ,Σ,n


def compute_label_stats(data, targets=None,indices=None,classnames=None,
                        online=True, batch_size=100, to_tensor=True,
                        eigen_correction=False,
                        eigen_correction_scale=1.0,
                        nworkers=0, diagonal_cov = False,
                        embedding=None,
                        device=None, dtype = torch.FloatTensor):
    """
    Computes mean/covariance of examples grouped by label. Data can be passed as
    a pytorch dataset or a dataloader. Uses dataloader to avoid loading all
    classes at once.

    Arguments:
        data (pytorch Dataset or Dataloader): data to compute stats on
        targets (Tensor, optional): If provided, will use this target array to
            avoid re-extracting targets.
        indices (array-like, optional): If provided, filtering is based on these
            indices (useful if e.g. dataloader has subsampler)
        eigen_correction (bool, optional):  If ``True``, will shift the covariance
            matrix's diagonal by :attr:`eigen_correction_scale` to ensure PSD'ness.
        eigen_correction_scale (numeric, optional): Magnitude of eigenvalue
            correction (used only if :attr:`eigen_correction` is True)

    Returns:
        M (dict): Dictionary with sample means (Tensors) indexed by target class
        S (dict): Dictionary with sample covariances (Tensors) indexed by target class
    """

    device = process_device_arg(device)
    M = {} # Means
    S = {} # Covariances

    ## We need to get all targets in advance, in order to filter.
    ## Here we assume targets is the full dataset targets (ignoring subsets, etc)
    ## so we need to find effective targets.
    if targets is None:
        targets, classnames, indices = extract_data_targets(data)
    else:
        assert (indices is not None), "If targets are provided, so must be indices"
    if classnames is None:
        classnames = sorted([a.item() for a in torch.unique(targets)])

    effective_targets = targets[indices]

    if nworkers > 1:
        import torch.multiprocessing as mp # Ugly, sure. But useful.
        mp.set_start_method('spawn',force=True)
        M = mp.Manager().dict() # Alternatively, M = {}; M.share_memory
        S = mp.Manager().dict()
        processes = []
        for i,c in enumerate(classnames): # No. of processes
            label_indices = indices[effective_targets == i]
            p = mp.Process(target=_single_label_stats,
                           args=(data, i,c,label_indices,M,S),
                           kwargs={'device': device, 'online':online})
            p.start()
            processes.append(p)
        for p in processes: p.join()
    else:
        for i,c in enumerate(classnames):
            label_indices = indices[effective_targets == i]
            μ,Σ,n = _single_label_stats(data, i,c,label_indices, device=device,
                                        dtype=dtype, embedding=embedding,
                                        online=online, diagonal_cov=diagonal_cov)
            M[i],S[i] = μ, Σ

    if to_tensor:
        ## Warning: this assumes classes are *exactly* {0,...,n}, might break things
        ## downstream if data is missing some classes
        M = torch.stack([μ.to(device) for i,μ in sorted(M.items()) if μ is not None], dim=0)
        S = torch.stack([Σ.to(device) for i,Σ in sorted(S.items()) if Σ is not None], dim=0)

    ### Shift the Covariance matrix's diagonal to ensure PSD'ness
    if eigen_correction:
        logger.warning('Applying eigenvalue correction to Covariance Matrix')
        λ = eigen_correction_scale
        for i in range(S.shape[0]):
            if eigen_correction == 'constant':
                S[i] += torch.diag(λ*torch.ones(S.shape[1], device = device))
            elif eigen_correction == 'jitter':
                S[i] += torch.diag(λ*torch.ones(S.shape[1], device=device).uniform_(0.99, 1.01))
            elif eigen_correction == 'exact':
                s,v = torch.symeig(S[i])
                print(s.min())
                s,v = torch.lobpcg(S[i], largest=False)
                print(s.min())
                s = torch.eig(S[i], eigenvectors=False).eigenvalues
                print(s.min())
                pdb.set_trace()
                s_min = s.min()
                if s_min <= 1e-10:
                    S[i] += torch.diag(λ*torch.abs(s_min)*torch.ones(S.shape[1], device=device))
                raise NotImplemented()
    return M,S


def dimreduce_means_covs(Means, Covs, redtype='diagonal'):
    """ Methods to reduce the dimensionality of the Feature-Mean/Covariance
        representation of Labels.

    Arguments:
        Means (tensor or list of tensors):  original mean vectors
        Covs (tensor or list of tensors):  original covariances matrices
        redtype (str): dimensionality reduction methods, one of 'diagonal', 'mds'
            or 'distance_embedding'.

    Returns:
        Means (tensor or list of tensors): dimensionality-reduced mean vectors
        Covs (tensor or list of tensors): dimensionality-reduced covariance matrices

    """
    n1, d1 = Means[0].shape
    n2, d2 = Means[1].shape
    k = d1

    print(n1, d1, n2, d2)
    if redtype == 'diagonal':
        ## Leave Means As Is, Keep Only Diag of Covariance Matrices, Independent DR for Each Task
        Covs[0] = torch.stack([torch.diag(C) for C in Covs[0]])
        Covs[1] = torch.stack([torch.diag(C) for C in Covs[1]])
    elif redtype == 'mds':
        ## Leave Means As Is, Use MDS to DimRed Covariance Matrices, Independent DR for Each Task
        Covs[0] = mds(Covs[0].view(Covs[0].shape[0], -1), output_dim=k)
        Covs[1] = mds(Covs[1].view(Covs[1].shape[0], -1), output_dim=k)
    elif redtype == 'distance_embedding':
        ## Leaves Means As Is, Use Bipartitie MSE Embedding, Which Embeds the Pairwise Distance Matrix, Rather than the Cov Matrices Directly
        print('Will reduce dimension of Σs by embedding pairwise distance matrix...')
        D = torch.zeros(n1, n2)
        print('... computing pairwise bures distances ...')
        for (i, j) in itertools.product(range(n1), range(n2)):
            D[i, j] = bures_distance(Covs[0][i], Covs[1][j])
        print('... embedding distance matrix ...')
        U, V = bipartite_mse_embedding(D, k=k)
        Covs = [U, V]
        print("Done! Σ's Dimensions: {} (Task 1) and {} (Task 2)".format(
            list(U.shape), list(V.shape)))
    else:
        raise ValueError('Reduction type not recognized')
    return Means, Covs


def pairwise_distance_mse(U, V, D, reg=1):
    d_uv = torch.cdist(U, V)
    l = torch.norm(D - d_uv)**2 / D.numel() + reg * (torch.norm(U) **
                                                     2 / U.numel() + torch.norm(V)**2 / V.numel())  # MSE per entry
    return l


def bipartite_mse_embedding(D, k=100, niters=10000):
    n, m = D.shape
    U = torch.randn(n, k, requires_grad=True)
    V = torch.randn(m, k, requires_grad=True)
    optim = torch.optim.SGD([U, V], lr=1e-1)
    for i in range(niters):
        optim.zero_grad()
        loss = pairwise_distance_mse(U, V, D)
        loss.backward()
        if i % 100 == 0:
            print(i, loss.item())
        optim.step()
    loss = pairwise_distance_mse(U, V, D, reg=0)
    print(
        "Final distortion: ||D - D'||\u00b2/|D| = {:4.2f}".format(loss.item()))
    return U.detach(), V.detach()
