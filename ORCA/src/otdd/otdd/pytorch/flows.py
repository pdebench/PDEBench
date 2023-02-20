import os
import sys
import time
import pdb
import logging

import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
    nodisplay = True
else:
    nodisplay = False
import matplotlib.pyplot as plt


import numpy as np
from tqdm.autonotebook import tqdm

import torch
from torch.utils.data import TensorDataset
from torchvision.utils import make_grid


from celluloid import Camera

from .distance import DatasetDistance
from .moments import compute_label_stats
from .utils import  show_grid, register_gradient_hook, inverse_normalize
from ..plotting import gaussian_density_plot, plot2D_samples_mat

logger = logging.getLogger(__name__)

try:
    from tsnecuda import TSNE
    tsnelib = 'tsnecuda'
except:
    logger.warning("tsnecuda not found - will use (slower) TSNE from sklearn")
    from sklearn.manifold import TSNE
    tsnelib = 'sklearn'
    #
    from openTSNE import TSNE
    tsnelib = 'opentsne'


################################################################################
#####################   MAIN GRADIENT FLOW CLASSES  ############################
################################################################################

class GradientFlow():
    """ Parent class for Gradient flows. Subclasses should provide at the very
    least init, step, end and flow methods.

    """
    def __init__(src, tgt):
        self.Ds = src
        self.Dt = tgt
        self.Zt = [] # To store trajectories

    def _add_callback(self, cb):
        self.callback = cb
        self.compute_coupling = cb.compute_coupling if hasattr(cb,'compute_coupling') else False
        self.store_trajectories = cb.store_trajectories if hasattr(cb,'store_trajectories') else False

    def init(self):
        pass

    def end(self):
        pass

    def step(self):
        pass

    def flow(self):
        """ """
        f_val = self.init()
        self.callback.on_flow_begin(self.otdd, d)

        # pbar = tqdm(self.times, leave=False)
        for iter, t in self.times: #enumerate(pbar):
            # pbar.set_description(f'Flow Step {iter}/{len(self.times)}')
            self.callback.on_step_begin(self.otdd, iter)
            f_val = self.step()
            logger.info("t={:8.4f}, F(a_t)={:8.2f}".format(t,f_val))
            self.callback.on_step_end(self.otdd, iter, t, d)
            print(get_gpu_memory_map())

        logger.info('Done')

        self.end()

        return f_val

    def animate(self, display_freq=None, save_path=None, **kwargs):
        """ A shortcut to creating the plotting callback with animation.

            - kwargs: additional arguments to be passed to Plotting2DCallback
        """
        cb = Plotting2DCallback(animate=True, display_freq=display_freq,
                                save_path=save_path, **kwargs)
        self._add_callback(cb)
        self.flow()
        return cb.animation


class OTDD_Gradient_Flow(GradientFlow):
    """
    Gradient Flows According to the OT Dataset Distance.

    Attributes
    ----------
    D1 :

    Method (str): the flow method
        - xonly: gradient updates only on the features. Label distribs recomputed
                  based on these. Graph from X to Means and Covariances is dettached
        - xonly-attached: gradient updates only on the features. Label distribs
                  recomputed based on these. Graph from X to Means and Covariances is kept attached
        - xytied: gradient updates on both features and labels distribs. But the
                   assignments of examples to labels are kept fixed throughout (no
                   class creation/destruction, fixed cluster sizes)
        - xyaugm: gradient updates on both features and label distribs, without
                   tying. Relies on "augmented" representation computation for OTDD
                   and therefore on diagonal approximation of the covariance matrix.

    Note that these are relveant only for inner_ot_method =! exact. For the exact
    (nonparametric) OTDD, there's no μ, Σ, so xyaugm/xytied don't make sense, and
    xonly/xonly-attached are equivalent.

    """
    def __init__(self, src, tgt=None, objective_type = 'otdd_only', functional = None,
                 method = 'xonly', optim='sgd', step_size = 0.1, steps = 20,
                 use_torchoptim=False,
                 compute_coupling=False, entreg_π = 1e-3,
                 fixed_labels=True,
                 clustering_method='kmeans',
                 clustering_input=None,
                 callback=None,
                 noisy_update = False,
                 noise_β = 0.01,
                 device = 'cpu',
                 precision='single',
                 eigen_correction=False,
                 **kwargs):
        """
            kwargs are for DatasetDistance.
        """
        self.device = device
        self.Ds = src
        self.Dt = tgt

        assert method in ['xonly', 'xonly-attached', 'xytied', 'xyaugm']
        self.method = method
        self.optim  = optim
        self.use_torchoptim = use_torchoptim
        self.fixed_labels = fixed_labels
        self.clustering_method = clustering_method
        self.clustering_input  = clustering_input


        self.entreg_π = entreg_π
        self.precision = precision
        self.eigen_correction = eigen_correction


        assert objective_type in ['otdd_only', 'ot_only', 'F_only', 'mixed']
        assert (functional is None) or callable(functional)
        assert (functional is None) or (objective_type in ['F_only', 'mixed'])

        assert not (objective_type in ['otdd_only', 'ot_only', 'mixed']) or self.Dt is not None, 'If objective contains distance, must provide tgt dataset'
        self.objective_type = objective_type
        self.functional     = functional

        ### I had this in my oher flows script. Makes sense only for fixed time horizon t in [0,1]
        self.times = np.arange(step_size,step_size*(steps+1), step_size)
        self.step_size = step_size
        self.steps = steps
        self.callback = callback if callback is not None else Callback()
        self.initialized = False
        self.X1_init = None
        self.compute_coupling = callback.compute_coupling if hasattr(callback,'compute_coupling') else False
        self.store_trajectories = callback.store_trajectories if hasattr(callback,'store_trajectories') else False
        self.trajectory_freq = callback.trajectory_freq if hasattr(callback,'trajectory_freq') else 1

        self.X1 = None # to store trajectories

        otdd_args = {
          'inner_ot_method': 'gaussian_approx',
          'nworkers_dists': 1,
          'nworkers_stats': 1,
          'debiased_loss': True,
          'sqrt_method': 'exact',
          'sqrt_niters': None,
          'sqrt_pref': 1, # to save some bacward computation on sqrts of src side.
          'p': 2,
          'entreg': 1e-1,
          'precision': precision,
          'device': device,
          'λ_y': None if objective_type == 'ot_only' else 1.0,
          'eigen_correction': eigen_correction
         }

        otdd_args.update(kwargs)
        otdd_args['method'] = 'augmentation' if method == 'xyaugm' else 'precomputed_labeldist'
        if 'diagonal_cov' not in otdd_args:
            otdd_args['diagonal_cov'] = (method == 'xy_augm')

        assert not ((otdd_args['inner_ot_method']=='exact') and (method in ['xy_augm', 'xytied'])), \
              "If inner_ot_method == 'exact', then flow method cannot be '{}'.".format(method)

        self.otdd = DatasetDistance(src, tgt, **otdd_args)
        self.history = []

    def init(self):
        self.t = 0.0

        ### Also initial target Mean and Covs (not in computational graph, static)
        ###_, _ = self.otdd._get_label_stats(side='tgt')

        d = self.otdd.distance(return_coupling=False)

        self.class_sizes = torch.unique(self.otdd.Y1, return_counts=True)[1] #torch.tensor([(self.otdd.Y1 == c).sum().item() for c in self.otdd.V1])

        logger.info('Using method: {}'.format(self.method))
        if self.method == 'xonly' or self.method == 'xonly-attached':
            self.otdd.X1.requires_grad_(True)
            flow_params   = [self.otdd.X1] # Updated by optim / gradient

        elif self.method == 'xytied':
            self.otdd.X1.requires_grad_(True)
            self.otdd.Covs[0].requires_grad_(True)
            self.otdd.Means[0].requires_grad_(True)
            stats_lr = self.step_size/(self.class_sizes.max()*0.2)
            flow_params = [
                {'params': self.otdd.X1, 'lr': self.step_size},
                {'params': self.otdd.Means[0], 'lr': stats_lr},
                {'params': self.otdd.Covs[0], 'lr': stats_lr},
            ]
            flow_params = [self.otdd.X1, self.otdd.Means[0], self.otdd.Covs[0]]

        elif self.method == 'xyaugm':
            self.otdd.XμΣ1.requires_grad_(True)
            flow_params = [self.otdd.XμΣ1]

        self._flow_params = flow_params

        if self.use_torchoptim:
            if self.optim == 'sgd':
                optimizer = torch.optim.SGD(flow_params, lr=self.step_size, momentum=0.5)
            elif self.optim == 'adam':
                optimizer = torch.optim.Adam(flow_params, lr=self.step_size)
            elif self.optim == 'adagrad':
                optimizer = torch.optim.Adagrad(flow_params, lr=self.step_size)
            self.optimizer = optimizer

        if self.compute_coupling is not False: self.coupling_update()

        ### Trigger first forward pass, now that things require grad

        if self.X1_init is None:
            self.X1_init = self.otdd.X1.detach().clone().cpu()
            self.Y1_init = self.otdd.Y1.detach().clone().cpu()
            if self.store_trajectories:
                self.Xt = self.X1_init.unsqueeze(-1).float() # time will be last dim
                self.Yt = self.Y1_init.unsqueeze(-1)
            if self.otdd.inner_ot_method != 'exact':
                self.M1_init = self.otdd.Means[0].detach().clone().cpu()
                self.C1_init = self.otdd.Covs[0].detach().clone().cpu()

        #
        logger.info("t={:8.2f}, F(a_t)={:8.2f}".format(0.0,d.item()))
        self.initialized = True
        return d.item()

    def coupling_update(self):
        logger.info('Coupling Update Computation')
        with torch.no_grad():
            otmethod = 'emd' if self.entreg_π <= 1e-3 else 'sinkhorn_epsilon_scaling'
            self.otdd.π = self.otdd.compute_coupling(entreg=self.entreg_π, method=otmethod,
                                                  verbose = True,  numItermax=50)

    def gradient_update(self):
        if self.use_torchoptim:
            L_αβ = 0
            if self.objective_type in ['otdd_only', 'ot_only', 'mixed']:
                L_αβ += self.otdd.distance(return_coupling=False)
            if callable(self.functional):
                L_αβ += self.functional(self.otdd.X1, self.otdd.Y1)
            self.optimizer.zero_grad()
            L_αβ.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(self._flow_params, 10)
            self.optimizer.step()

        else: # Manual Updating
            L_αβ = self.otdd.distance(return_coupling=False)
            lr = self.step_size
            X  = self.otdd.X1
            if self.noisy_update: X += torch.randn(X.shape)*self.noise_β

            if self.method in ['xonly', 'xonly-attached']:
                [gX]  = torch.autograd.grad(L_αβ, [X])
                self.otdd.X1.data -= lr * len(X) * gX
            elif self.method == 'xytied':
                [gX,gM,gC]  = torch.autograd.grad(L_αβ, [X, self.otdd.Means[0], self.otdd.Covs[0]])
                for g in [gX,gM,gC]:
                    if torch.isnan(g).any(): pdb.set_trace(header='Failed Grad Check')

                class_sizes = torch.tensor([(self.otdd.Y1 == c).sum().item() for c in self.otdd.V1]) * 0.1

                self.otdd.X1.data       -= lr * len(X) * gX
                self.otdd.Means[0].data -= lr * len(self.otdd.Means[0]) * gM / class_sizes.view(-1,1) #* 0.2 # slower lr for Means and Covs
                self.otdd.Covs[0].data  -= lr * len(self.otdd.Covs[0]) * gC / class_sizes.view(-1,1,1) #* 0.2
            elif self.method == 'xyaugm':
                if self.noisy_update:
                    pdb.set_trace(header='Not sure xyaugm works with noisy update yet')
                [gXMC]  = torch.autograd.grad(L_αβ, [self.otdd.XμΣ1])#, self.otdd.Means[0], self.otdd.Covs[0]])
                self.otdd.XμΣ1.data     -= lr * len(self.otdd.XμΣ1) * gXMC
                self.otdd.X1 = self.otdd.XμΣ1.data[:,:2]
        return L_αβ.item()


    def label_update(self):
        """ Triggers an update to the categorial labels associated with each
        particle, based on current state.

        Arguments:
            cinput (str): input to the clustering method, i.e., what parts of the
                current state are used to impute labels. One of 'feats', 'stats'
                or 'both'.

        """
        cinput  = self.clustering_input
        cmethod = self.clustering_method

        if cinput == 'feats':
            U = self.otdd.X1.detach().numpy()
        elif cinput == 'stats':
            if self.method == 'xyaugm':
                d = self.otdd.X1.shape[1]
                U = self.otdd.XμΣ1.detach().numpy()[:,d:]
            else:
                raise NotImplemented()
        elif cinput == 'both':
            if self.method == 'xyaugm':
                U = self.otdd.XμΣ1.detach().numpy()
            else:
                raise NotImplemented()

        if cmethod == 'kmeans':
            k = len(self.otdd.classes2)
            C,L,_ = k_means(U, k)
        elif cmethod == 'dbscan':
            L = DBSCAN(eps=5, min_samples = 4).fit(U).labels_
        else:
            raise ValueError()

        self.otdd.Y1 = torch.LongTensor(L)
        self.otdd.targets1 = self.otdd.Y1
        self.otdd.classes1 = [str(i.item()) for i in torch.unique(self.otdd.targets1)]


    def stats_update(self):
        """ Triggers update on means and covariances of particles """
        dtype = torch.DoubleTensor if self.precision == 'double' else torch.FloatTensor

        if self.method == 'xonly':
            Ds_t = TensorDataset(self.otdd.X1.detach().clone(), self.otdd.Y1)#self.Ds.tensors[1])

            self.otdd.Means[0], self.otdd.Covs[0] = compute_label_stats(Ds_t,
                                                            targets=self.otdd.targets1,
                                                            indices=np.arange(len(self.otdd.Y1)),
                                                            classnames=self.otdd.classes1,
                                                            to_tensor=True,
                                                            diagonal_cov = self.otdd.diagonal_cov,
                                                            online=self.otdd.online_stats,
                                                            device=self.otdd.device,
                                                            dtype=dtype,
                                                            eigen_correction=self.eigen_correction,
                                                            )

        elif self.method == 'xonly-attached':
            Ds_t = TensorDataset(self.otdd.X1, self.otdd.Y1)

            self.otdd.Means[0],self.otdd.Covs[0] = compute_label_stats(Ds_t,
                                        targets=self.otdd.targets1,
                                        indices=np.arange(len(self.otdd.Y1)),
                                        classnames=self.otdd.classes1,
                                        to_tensor=True,
                                        diagonal_cov = self.otdd.diagonal_cov,
                                        online=self.otdd.online_stats,
                                        device=self.otdd.device,
                                        dtype=dtype,
                                        eigen_correction=self.eigen_correction
                                        )

            if torch.isnan(self.otdd.Covs[0]).any():
                pdb.set_trace(header='Nans in Cov Matrices')


        elif self.method == 'xytied':
            pass
        elif self.method == 'xyaugm':
            d = self.otdd.X1.shape[1]

            self.otdd.X1 = self.otdd.XμΣ1.data[:,:d]

            if self.clustering_method == 'kmeans':
                k = len(self.otdd.classes2)
                C,L,_ = k_means(self.otdd.XμΣ1.detach().numpy()[:,d:], k)
            elif self.clustering_method == 'dbscan':
                L = DBSCAN(eps=5, min_samples = 4).fit(self.otdd.XμΣ1.detach().numpy()[:,d:]).labels_

            self.otdd.Y1 = torch.LongTensor(L)
            self.otdd.targets1 = self.otdd.Y1
            self.otdd.classes1 = [str(i.item()) for i in torch.unique(self.otdd.targets1)]

            Ds_t = TensorDataset(self.otdd.X1, self.otdd.Y1)

            M, S = compute_label_stats(Ds_t,
                                       targets=self.otdd.targets1,
                                       indices=np.arange(len(self.otdd.Y1)),
                                       classnames=self.otdd.classes1,
                                       to_tensor=True,
                                       nworkers=0,
                                       diagonal_cov = True,
                                       online=self.otdd.online_stats,
                                       device=self.otdd.device,
                                       eigen_correction=self.eigen_correction
                                       )
            self.otdd.Means[0] = M
            self.otdd.Covs[0]  = S
            _, cts =  torch.unique(self.otdd.Y1, return_counts=True)
            logger.info('Counts: {}'.format(cts))
            self.class_sizes = cts

        else:
            raise ValueError('Unrecoginzed flow method')


    def step(self, iter):
        assert self.initialized

        ##### Update differentiable-dynamic params
        self.otdd.label_distances  = None
        self.otdd._pwlabel_stats_1 = None

        logger.info('Performing flow gradient step...')
        L_αβ = self.gradient_update()

        ##### Update labels (if fixed_labels != False )
        if not self.fixed_labels:
            logger.info('Performing label update...')
            self.label_update()

        ###### Trigger update of those that are not updated by gradient
        if self.otdd.inner_ot_method != 'exact':
            logger.info('Performing stats update...')
            self.stats_update()

        if self.compute_coupling == 'every_iteration':
            logger.info('Performing coupling update...')
            self.coupling_update()

        if self.store_trajectories and (iter % self.trajectory_freq == 0):
            self.Xt = torch.cat([self.Xt, self.otdd.X1.detach().clone().cpu().float().unsqueeze(-1)], dim=-1)
            self.Yt = torch.cat([self.Yt, self.otdd.Y1.detach().clone().cpu().unsqueeze(-1)], dim=-1)


        return L_αβ

    def flow(self, tol = 1e-3):
        """ """
        prev = np.Inf
        obj = self.init()
        self.history.append(obj)
        self.callback.on_flow_begin(self.otdd, obj)
        # pbar = tqdm(self.times, leave=False)
        for iter,t in enumerate(self.times):#pbar, 1):
            # pbar.set_description(f'Flow Step {iter}/{len(self.times)}, F_t={obj:8.2f}')
            self.callback.on_step_begin(self.otdd, iter)
            obj = self.step(iter)
            logger.info(f't={t:8.2f}, F(a_t)={obj:8.2f}') # Although things have been updated, this is obj of time t still
            self.history.append(obj)
            self.t = t
            self.callback.on_step_end(self, self.otdd, iter, t, obj) # Now that we pass whole flow obj, maybe no need to pass otdd
            Δ = np.abs(obj - prev)
            if tol and (Δ < tol):
                logger.warning(f'Stoping condition met (Δ = {Δ:2.2e} < {tol:2.2e} = tol). Terminating flow.')
                break
            else:
                prev = obj

        logger.info('Done')
        cbout = self.callback.on_flow_end(self, self.otdd)#, iter, t, d)
        self.end()
        return obj, cbout

    def end(self):
        self.otdd.X1       = self.X1_init
        self.otdd.Y1       = self.Y1_init
        if self.otdd.inner_ot_method != 'exact':
            self.otdd.Means[0] = self.M1_init
            self.otdd.Covs[0]  = self.C1_init



################################################################################
###################    CALLBACKS FOR GRADIENT FLOW CLASS  ######################
################################################################################

class Callback():
    compute_coupling   = False
    store_trajectories = False
    trajectory_freq    = None
    def __init__(self): pass
    def on_flow_begin(self, *args, **kwargs): pass
    def on_flow_end(self, *args, **kwargs): pass
    def on_step_begin(self, *args, **kwargs): pass
    def on_step_end(self, *args, **kwargs): pass


class CallbackList(Callback):
    def __init__(self, cbs):
        self.cbs = cbs
        ### Aggregate requierements imposed by callbacks on Flow - if at least
        ### one of them needs it, ask for it.
        coupling_attrs = [cb.compute_coupling for cb in cbs]
        if 'every_iteration' in coupling_attrs:
            self.compute_coupling = 'every_iteration'
        elif 'initial' in coupling_attrs:
            self.compute_coupling = 'initial'
        else:
            self.compute_coupling = False

        trajectory_attrs = [cb.store_trajectories for cb in cbs]
        self.store_trajectories = np.array(trajectory_attrs).any()
        trajfreq_attrs =  [cb.trajectory_freq for cb in cbs if cb.trajectory_freq is not None]
        self.trajectory_freq = np.array(trajfreq_attrs).min() if trajfreq_attrs else None

    def __getitem__(self, i):
        return self.cbs[i]

    def on_flow_begin(self, *args, **kwargs):
        for cb in self.cbs: cb.on_flow_begin(*args, **kwargs)
    def on_flow_end(self, *args, **kwargs):
        for cb in self.cbs: cb.on_flow_end(*args, **kwargs)
    def on_step_begin(self, *args, **kwargs):
        for cb in self.cbs: cb.on_step_begin(*args, **kwargs)
    def on_step_end(self, *args, **kwargs):
        for cb in self.cbs: cb.on_step_end(*args, **kwargs)


class Plotting2DCallback(Callback):
    def __init__(self, display_freq=None, animate=False, entreg_π = 1e-4,
        show_coupling = True, show_trajectories=True, trajectory_size=5, show_target = True,
        plot_pad = 2, save_format='pdf', ndim=2, save_path=None):
        self.animate = animate
        self.save_path = save_path
        self.display_freq = display_freq
        self.entreg_π = entreg_π
        self.show_coupling = show_coupling
        self.compute_coupling = 'every_iteration' if show_coupling else False
        self.show_trajectories = show_trajectories
        self.store_trajectories = self.show_trajectories
        self.trajectory_freq = self.display_freq
        self.trajectory_size = trajectory_size
        self.show_target = show_target
        self.plot_pad = plot_pad
        self.ndim = ndim

        self.ax_ranges = [None]*ndim
        self.figsize = (6,4) if not animate else (10,7)
        self.save_format = save_format

        raise DeprecationWarning('Plotting2DCallback has been deprecated in favor of PlottingCallback')


    def _plot(self, otdd, X1, Y1=None, X2=None, Y2=None, title=None, trajectories=None):
        with torch.no_grad():
            ### Now parent flow takes care of coupling computation
            if self.animate:
                ax = self.ax
                xrng, yrng = ax.get_xlim(), ax.get_ylim()
            else:
                fig, ax  = plt.subplots(figsize=self.figsize)
                xrng, yrng = self._get_ax_ranges(X1, X2)

            otdd.plot_label_stats(ax = ax, same_plot=True, show_target=self.show_target,
                                  label_means=False,
                                  label_groups=True, show=False,shift=(2,-2))

            if self.show_trajectories and trajectories is not None:
                pdb.set_trace()
                for x in trajectories:
                    ax.plot(*x, color='k', alpha=0.2, linewidth=0.5)

            if self.show_target and self.show_coupling:
                π = otdd.π
                plot2D_samples_mat(X1, X2, π, ax=ax, linewidth=0.1, thr=1e-10, linestyle=':')
            ax.set_xlim(xrng)
            ax.set_ylim(yrng)
            ax.set_title('')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if title is not None:
                ax.text(0.5, 1.01, title, transform=ax.transAxes, ha='center',size=18)

    def _get_ax_ranges(self, X1, X2):
        pad = self.plot_pad
        if all(v is not None for v in self.ax_ranges):
            return self.ax_ranges
        with torch.no_grad():
            mins, maxs = [], []
            for i in range(self.ndim):
                if self.show_target:
                    mins.append(min(X1[:,i].min(), X2[:,i].min()) - pad)
                    maxs.append(max(X1[:,i].max(), X2[:,i].max()) + pad)
                else:
                    mins.append(X1[:,i].min() - pad)
                    maxs.append(X1[:,i].max() + pad)

        self.ax_ranges = [(mins[i].item(), maxs[i].item()) for i in range(self.ndim)]
        return self.ax_ranges

    def on_flow_begin(self, otdd, d):
        if self.save_path:
            save_dir = os.path.dirname(self.save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        ax_ranges = self._get_ax_ranges(otdd.X1,otdd.X2)
        if self.animate:
            self.fig, self.ax = plt.subplots(figsize=(10,7))
            self.ax.set_xlim(xrng)
            self.ax.set_ylim(yrng)
            self.camera = Camera(self.fig)

        title = r'Time t=0, $F(\rho_t)$={:4.2f}'.format(d)
        _ = self._plot(otdd, otdd.X1, otdd.Y1, otdd.X2, otdd.Y2, title)
        if self.animate:
            self.camera.snap()
        else:
            if self.save_path:
                outpath = self.save_path + 't0.' + self.save_format
                plt.tight_layout()
                plt.savefig(outpath, dpi=300) #bbox_inches='tight',
            plt.show(block=False)
            plt.pause(1)
            plt.close()

    def on_flow_end(self, flow, otdd):
        if self.animate:
            animation = self.camera.animate()
            if self.save_path:
                animation.save(self.save_path +'.mp4')
            self.animation = animation
            plt.close(self.fig)

    def on_step_end(self, flow, otdd, iteration, t, d, **kwargs):
        if self.display_freq is None or (iteration % self.display_freq == 0): # display
            title = r'Time t={:.2f}, $F(\rho_t)$={:4.2f}'.format(t, d)
            if self.show_trajectories and 'trajectories' in kwargs:
                self._plot(otdd, otdd.X1.detach(), otdd.Y1.detach(), otdd.X2, otdd.Y2, trajectories=kwargs['trajectories'], title=title)
            elif self.show_trajectories:
                self._plot(otdd, otdd.X1.detach(), otdd.Y1.detach(), otdd.X2, otdd.Y2, trajectories=flow.Xt, title=title)
            else:
                self._plot(otdd, otdd.X1.detach(), otdd.Y1.detach(), otdd.X2, otdd.Y2, title=title)

            if self.animate:
                self.camera.snap()
            else:
                if self.save_path:
                    outpath = self.save_path + 't{}.{}'.format(iteration, self.save_format)
                    plt.tight_layout()
                    plt.savefig(outpath, dpi=300) #bbox_inches='tight',
                plt.show(block=False)
                plt.pause(1)
                plt.close()


class PlottingCallback(Callback):
    def __init__(self, display_freq=None, animate=False, entreg_π = 1e-4,
        show_coupling = True, show_trajectories=True, trajectory_size=5, show_target = True,
        plot_pad = 2, figsize=(6,4), save_format='pdf', ndim=2, azim=-80 , elev=5, save_path=None):
        self.animate = animate
        self.display_freq = display_freq
        self.entreg_π = entreg_π
        self.show_coupling = show_coupling
        self.compute_coupling = 'every_iteration' if show_coupling else False
        self.show_trajectories = show_trajectories
        self.store_trajectories = self.show_trajectories
        self.trajectory_freq = self.display_freq
        self.trajectory_size = trajectory_size
        self.show_target = show_target
        self.ndim = ndim

        ## Low-level plotting args
        self.figsize = figsize
        self.azim = azim
        self.elev = elev
        self.plot_pad = plot_pad
        self.ax_ranges = [None]*ndim

        self.save_format = save_format
        self.save_path = save_path


    def _plot(self, otdd, X1, Y1=None, X2=None, Y2=None, title=None, trajectories=None):
        with torch.no_grad():
            if self.animate:
                ax = self.ax
                ax_ranges = [ax.get_xlim(), ax.get_ylim()]
                if self.ndim == 3:
                    ax_ranges.append(ax.get_zlim())
            else:
                fig = plt.figure(figsize=self.figsize)
                if self.ndim == 2:
                    ax = fig.add_subplot(111)
                elif self.ndim == 3:
                    ax = fig.add_subplot(111, azim=self.azim, elev=self.elev, projection='3d')
                ax_ranges = self._get_ax_ranges(X1, X2)

            if self.ndim == 2:
                otdd.plot_label_stats(ax = ax, same_plot=True, show_target=self.show_target,
                                      label_means=False,
                                      label_groups=True, show=False,shift=(2,-2))
            else:
                ax.scatter(*X1.detach().T, c=Y1, cmap = 'tab10')

            if self.show_trajectories and trajectories is not None:
                for x in trajectories[:,:,-self.trajectory_size:]:
                    ax.plot(*x, color='k', alpha=0.2, linewidth=0.5)


            ax.set_title('')
            ax.set_xlim(ax_ranges[0])
            ax.set_ylim(ax_ranges[1])
            if self.ndim == 2:
                ## To remove ticks & axes grids
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
            else:
                ## To keep grid
                for _ax in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    _ax.set_ticklabels([])
                    _ax.set_ticks_position('none')
                ax.set_zlim(ax_ranges[2])

            if title is not None:
                if self.ndim == 2:
                    ax.text(0.5, 1.1, title, transform=ax.transAxes, ha='center',size=18)
                else:
                    ax.text2D(0.5, 0.85, title, transform=ax.transAxes, ha='center',size=18)

    def _get_ax_ranges(self, X1, X2):
        pad = self.plot_pad
        if all(v is not None for v in self.ax_ranges):
            return self.ax_ranges
        with torch.no_grad():
            mins, maxs = [], []
            for i in range(self.ndim):
                if self.show_target:
                    mins.append(min(X1[:,i].min(), X2[:,i].min()) - pad)
                    maxs.append(max(X1[:,i].max(), X2[:,i].max()) + pad)
                else:
                    mins.append(X1[:,i].min() - pad)
                    maxs.append(X1[:,i].max() + pad)

        self.ax_ranges = [(mins[i].item(), maxs[i].item()) for i in range(self.ndim)]
        return self.ax_ranges

    def on_flow_begin(self, otdd, d):
        if self.save_path:
            save_dir = os.path.dirname(self.save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        ax_ranges = self._get_ax_ranges(otdd.X1,otdd.X2)

        if self.animate:
            self.fig = plt.figure(figsize=self.figsize)
            if self.ndim == 2:
                self.ax = self.fig.add_subplot(111)
            elif self.ndim == 3:
                self.ax = self.fig.add_subplot(111, azim=self.azim, elev=self.elev, projection='3d')

            self.ax.set_xlim(ax_ranges[0])
            self.ax.set_ylim(ax_ranges[1])
            if self.ndim == 3:
                self.ax.set_zlim(ax_ranges[2])
            self.camera = Camera(self.fig)

        title = r'Time t=0, $F(\rho_t)$={:4.2f}'.format(d)
        _ = self._plot(otdd, otdd.X1, otdd.Y1, otdd.X2, otdd.Y2, title)
        if self.animate:
            self.camera.snap()
        else:
            if self.save_path:
                outpath = self.save_path + 't0.' + self.save_format
                plt.tight_layout()
                plt.savefig(outpath, dpi=300) #bbox_inches='tight',
            plt.show(block=False)
            plt.pause(1)
            plt.close()

    def on_flow_end(self, flow, otdd):
        if self.animate:
            animation = self.camera.animate()
            if self.save_path:
                animation.save(self.save_path +'.mp4')
            self.animation = animation
            plt.close(self.fig)

    def on_step_end(self, flow, otdd, iteration, t, d, **kwargs):
        if self.display_freq is None or (iteration % self.display_freq == 0): # display
            title = r'Time t={:.2f}, $F(\rho_t)$={:4.2f}'.format(t, d)
            if self.show_trajectories and 'trajectories' in kwargs:
                self._plot(otdd, otdd.X1.detach(), otdd.Y1.detach(), otdd.X2, otdd.Y2, trajectories=kwargs['trajectories'], title=title)
            elif self.show_trajectories:
                self._plot(otdd, otdd.X1.detach(), otdd.Y1.detach(), otdd.X2, otdd.Y2, trajectories=flow.Xt, title=title)
            else:
                self._plot(otdd, otdd.X1.detach(), otdd.Y1.detach(), otdd.X2, otdd.Y2, title=title)

            if self.animate:
                self.camera.snap()
            else:
                if self.save_path:
                    outpath = self.save_path + 't{}.{}'.format(iteration, self.save_format)
                    plt.tight_layout()
                    plt.savefig(outpath, dpi=300) #bbox_inches='tight',
                plt.show(block=False)
                plt.pause(0.2)
                plt.close()


class Embedding2DCallback(Plotting2DCallback):
    def __init__(self, method = 'tsne', joint = True, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.joint  = joint


    def _embed(self, otdd):
        with torch.no_grad():
            X = torch.cat([otdd.X1.detach().clone().cpu(),otdd.X2.clone().cpu()], dim=0)
            if tsnelib in ['sklearn', 'tsnecuda']:
                X_emb = TSNE(n_components=self.ndim, verbose=0, perplexity=50).fit_transform(X)
            else:
                if not hasattr(self, 'tsne') or self.tsne is None:
                    X_emb = TSNE(n_components=self.ndim, perplexity=50,
                             n_jobs=8,verbose=True).fit(X)
                    self.tsne = X_emb
                X_emb = self.tsne.transform(X).astype(np.float32)
            if isinstance(X_emb, np.ndarray):
                X_emb = torch.from_numpy(X_emb)
            X1_emb, X2_emb = X_emb[:otdd.X1.shape[0],:], X_emb[otdd.X1.shape[0]:,:]
        return X1_emb.to('cpu'), X2_emb.to('cpu')

    def on_flow_begin(self, otdd, d):

        X1_emb, X2_emb = self._embed(otdd)

        ## Compute Stats for Embedded Points
        Ds_emb = TensorDataset(X1_emb, otdd.Y1)
        Dt_emb = TensorDataset(X2_emb, otdd.Y2)

        Ms_emb, Cs_emb =  compute_label_stats(Ds_emb,
                                            targets=otdd.Y1.cpu() - otdd.Y1.cpu().min(),
                                            indices=np.arange(len(otdd.Y1)),
                                            classnames=otdd.classes1,
                                            to_tensor=True,
                                            nworkers=0, device=otdd.device,
                                            diagonal_cov = otdd.diagonal_cov,
                                            online=otdd.online_stats,
                                            eigen_correction=otdd.eigen_correction,
                                            )

        Mt_emb, Ct_emb =  compute_label_stats(Dt_emb,
                                            targets=otdd.Y2.cpu() - otdd.Y2.cpu().min(),
                                            indices=np.arange(len(otdd.Y2)),
                                            classnames=otdd.classes2,
                                            to_tensor=True,
                                            nworkers=0, device=otdd.device,
                                            diagonal_cov = otdd.diagonal_cov,
                                            online=otdd.online_stats,
                                            eigen_correction=otdd.eigen_correction,
                                            )



        otdd_emb = otdd.copy(keep=['classes1','classes2', 'targets1', 'targets2','Y1','Y2'])
        otdd_emb.Y1 = otdd_emb.Y1.cpu()
        otdd_emb.Y2 = otdd_emb.Y2.cpu()
        otdd_emb.X1 = X1_emb
        otdd_emb.X2 = X2_emb
        otdd_emb.Means = [Ms_emb.cpu(),Mt_emb.cpu()]
        otdd_emb.Covs  = [Cs_emb.cpu(),Ct_emb.cpu()]

        self.otdd_emb = otdd_emb

        if self.store_trajectories:
            self.Xt = self.otdd_emb.X1.detach().clone().cpu().unsqueeze(-1).float() # time will be last dim
            self.Yt = self.otdd_emb.Y1.detach().clone().cpu().unsqueeze(-1)


        super().on_flow_begin(self.otdd_emb, d)


    def on_step_end(self, flow, otdd, iteration, t, d, **kwargs):
        if self.display_freq is None or (iteration % self.display_freq == 0):
            X1_emb, X2_emb = self._embed(otdd)
            Ds_emb = TensorDataset(X1_emb, otdd.Y1)

            Ms_emb, Cs_emb =  compute_label_stats(Ds_emb,
                                                targets=otdd.Y1.cpu() - otdd.Y1.cpu().min(),
                                                indices=np.arange(len(otdd.Y1)),
                                                classnames=otdd.classes1,
                                                to_tensor=True,
                                                nworkers=0, device=otdd.device,
                                                diagonal_cov = otdd.diagonal_cov,
                                                eigen_correction=otdd.eigen_correction,
                                                online=otdd.online_stats)
            self.otdd_emb.X1.data  = X1_emb
            self.otdd_emb.Means[0] = Ms_emb
            self.otdd_emb.Covs[0]  = Cs_emb

            Dt_emb = TensorDataset(X2_emb, otdd.Y2)
            Mt_emb, Ct_emb =  compute_label_stats(Dt_emb,
                                                targets=otdd.Y2.cpu() - otdd.Y2.cpu().min(),
                                                indices=np.arange(len(otdd.Y2)),
                                                classnames=otdd.classes2,
                                                to_tensor=True,
                                                nworkers=0, device=otdd.device,
                                                diagonal_cov = otdd.diagonal_cov,
                                                eigen_correction=otdd.eigen_correction,
                                                online=otdd.online_stats)
            self.otdd_emb.X2.data  = X2_emb
            self.otdd_emb.Means[1] = Mt_emb
            self.otdd_emb.Covs[1]  = Ct_emb

            if self.store_trajectories:
                ## Convert to cpu, float (in case it was double) for dumping
                self.Xt = torch.cat([self.Xt, self.otdd_emb.X1.detach().clone().cpu().float().unsqueeze(-1)], dim=-1)
                self.Yt = torch.cat([self.Yt, self.otdd_emb.Y1.detach().clone().cpu().unsqueeze(-1)], dim=-1)

            super().on_step_end(
                flow, self.otdd_emb, iteration, t, d,
                trajectories=self.Xt if self.show_trajectories else None, # Must override non-embedded trajectories
                **kwargs)




class ImageGridCallback(Callback):
    """

            by_class:  Grid will sample so that each col contains a single class
            only_matched:  Only display properly matched particles (assumes labels
                           of src and tgt are in direct correspondence (1st <-> 1st)
                           etc, but will compensate in case Y2's are shifted. If
                           True, automatically does by_class too.
    """
    def __init__(self, display_freq=None, animate=False, entreg_π = 1e-4,
                 byclass = True, only_matched=True, nrow=10, ncol=10,
                 channels = 1, transparent=False, denormalize=None, save_path=None):
        self.animate = animate
        self.save_path = save_path
        if save_path:
            self.outdir = os.path.dirname(save_path)
        self.display_freq = display_freq
        self.channels = channels
        self.entreg_π = entreg_π
        self.byclass   = byclass
        self.only_matched = only_matched
        self.compute_coupling = 'initial' if only_matched else False
        self.transparent = transparent
        self.denormalize = denormalize

        if not self.byclass:
            self.nrow = nrow
            self.ncol = nrow if ncol is None else ncol
        else:
            self.ncol = ncol
            self.nrow = None # will be number of classes, determined later
        self.indices = None

    def _plot(self, otdd, X1, X2, title):
        with torch.no_grad():
            batch = X1[self.indices].view(len(self.indices), self.channels, self.imdim[0],self.imdim[1])
            if self.denormalize is not None:
                batch = inverse_normalize(batch, *self.denormalize)
            ## make_grid reverts nrow ncol for some strange reason.
            grid = make_grid(batch, nrow=self.ncol, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
            if self.animate:
                ax = self.ax
            else:
                fig, ax  = plt.subplots(figsize=(self.ncol,self.nrow))
            show_grid(grid, ax=ax)
            ax.text(0.5, 1.03, title, transform=ax.transAxes, ha='center',size=18)

    def _choose_examples(self, otdd):
        X1, Y1 = otdd.X1.detach().cpu(), otdd.Y1.cpu()
        X2, Y2 = otdd.X2.detach().cpu(), otdd.Y2.cpu()
        if self.only_matched:
            ## we match src -> trg
            tgt_idxs = np.argmax(otdd.π, axis=1)
            tgt_labels = Y2[tgt_idxs] - min(Y2)
            matched = (Y1 == tgt_labels)
            idxs = []
            for c in otdd.V1:
                idxs_class_matched = torch.where((Y1 == c) & matched)[0]
                if len(idxs_class_matched) >= self.nrow:
                    ## Have enough matched, select from these with prob prop ot weight on correct class
                    p = otdd.π[idxs_class_matched,:][:,Y2-min(Y2)==c].sum(axis=1)
                    p /= p.sum()
                    idxs_class_selected = np.sort(np.random.choice(idxs_class_matched, self.nrow, replace=False, p=p))
                else:
                    ## Not enough matched, complete with unmatched
                    idxs_class = torch.where(Y1 == c)[0]
                    unmatched = np.random.choice(idxs_class, self.nrow - len(idxs_class_matched), replace=False)
                    idxs_class_selected = np.concatenate((idxs_class_matched, unmatched))

                assert len(idxs_class_selected) == self.nrow
                idxs.append(idxs_class_selected)
            idxs = np.concatenate(idxs)
        elif not self.byclass:
            idxs = np.sort(np.random.choice(X1.shape[0], self.nrow*self.ncol, replace=False))
        else:
            idxs = []
            for c in otdd.V1: # V1 is never index-shifted, so this works
                idxs_class = torch.where(Y1 == c)[0]
                idxs.append(np.sort(np.random.choice(idxs_class, self.ncol, replace=False)))
            idxs = np.concatenate(idxs)
        self.indices = idxs

    def on_flow_begin(self, otdd, d):
        if not os.path.exists(self.outdir): os.makedirs(self.outdir)
        ## Choose indices of examples to show in grid
        n, d2 = otdd.X1.shape
        self.imdim = (int(np.sqrt(d2/self.channels)),int(np.sqrt(d2/self.channels)))
        if self.byclass:
            self.nrow = len(otdd.V1)
            self.ncol = self.nrow if self.ncol is None else self.ncol
        self._choose_examples(otdd)

        if self.animate:
            self.fig, self.ax = plt.subplots(figsize=(self.ncol,self.nrow))
            self.camera = Camera(self.fig)
        title = 'Time t=0, OTDD(S,T)={:4.2f}'.format(d)
        self._plot(otdd, otdd.X1.detach(), otdd.X2, title)
        if self.animate:
            self.camera.snap()
        else:
            if self.save_path:
                outpath = self.save_path + 't0'
                plt.savefig(outpath+'.pdf', dpi=300, transparent=self.transparent) #bbox_inches='tight',
                plt.savefig(outpath+'.png', dpi=300, transparent=self.transparent) #bbox_inches='tight',
            plt.show(block=False)
            plt.pause(1)
            plt.close()

    def on_flow_end(self, flow, otdd):
        if self.animate:
            animation = self.camera.animate()
            if self.save_path:
                animation.save(self.save_path +'flow.mov', codec='png', dpi=300,
                               savefig_kwargs={'transparent': self.transparent,
                               'facecolor': 'none'})
            self.animation = animation
            plt.close(self.fig)

    def on_step_end(self, flow, otdd, iteration, t, d, **kwargs):
        if not self.display_freq or (iteration % self.display_freq == 0):
            title = r'Time t={:.2f}, OTDD(S,T)={:4.2f}'.format(t, d)
            self._plot(otdd, otdd.X1.detach(), otdd.X2, title)

            if self.animate:
                self.camera.snap()
            else:
                if self.save_path:
                    outpath = self.save_path + 't{}'.format(iteration)
                    plt.savefig(outpath+'.pdf', dpi=300, transparent=self.transparent)
                    plt.savefig(outpath+'.png', dpi=300, transparent=self.transparent)
                plt.show(block=False)
                plt.pause(1)
                plt.close()


class TrainingCallback(Callback):
    def __init__(self, criterion = 'xent', lr = 0.01, momentum=0.9, iters=20):
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.iters = iters

    def init_model(self, nclasses=10):
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, nclasses),
        )
        return net

    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(k,-1).float().sum()
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train(self, model, X,Y, **kwargs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)

        for it in range(self.iters):
            output = model(X)
            loss = criterion(output, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc1, acc3 = self.accuracy(output,Y, topk=(1,3))

        logger.info('Finall Loss: {} ({}%)'.format(loss, acc1.item()))

    def on_step_end(self, flow, otdd, iteration, t, d, **kwargs):

        net = self.init_model(nclasses=len(torch.unique(otdd.Y1)))

        self.train(net, otdd.X1.detach().clone(), otdd.Y1)


class TrajectoryDump(Callback):
    def __init__(self, save_freq=None, save_path=None):
        self.save_freq = save_freq
        self.save_path = save_path
        self.store_trajectories = True
        self.trajectory_freq = save_freq
        self.outdir = save_path

    def on_flow_begin(self, otdd, d):
        if not os.path.exists(self.outdir): os.makedirs(self.outdir)

    def on_flow_end(self, flow, otdd):
        trajpath = os.path.join(self.outdir, 'trajectories_X.pt')
        torch.save(flow.Xt, trajpath)
        trajpath = os.path.join(self.outdir, 'trajectories_Y.pt')
        torch.save(flow.Yt, trajpath)
        if hasattr(flow.otdd, 'Y1_true') and (flow.otdd.Y1_true is not None):
            trajpath = os.path.join(self.outdir, 'Y_init_true.pt')
            torch.save(flow.otdd.Y1_true, trajpath)
        logger.info('Saved trajectories to: {}'.format(trajpath))

    def on_step_end(self, flow, otdd, iteration, t, d, **kwargs):
        pass
