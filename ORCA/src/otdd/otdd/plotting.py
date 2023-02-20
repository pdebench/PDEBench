"""Plotting tools for Optimal Transport Dataset Distance.


"""

import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import seaborn as sns
import torch

import scipy.stats
from scipy.stats import pearsonr, spearmanr

from mpl_toolkits.axes_grid1 import make_axes_locatable

from adjustText import adjust_text

import pdb

logger = logging.getLogger(__name__)

def as_si(x, ndp):
    """ Convert humber to latex-style x10 scientific notation string"""
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


def get_plot_ranges(X):
    x, y = X[:,0], X[:,1]
    dx = (x.max() - x.min())/10
    dy = (y.max() - y.min())/10
    xmin = x.min() - dx
    xmax = x.max() + dx
    ymin = y.min() - dy
    ymax = y.max() + dy
    return (xmin,xmax,ymin,ymax)

def gaussian_density_plot(P=None, X=None, method = 'exact', nsamples = 1000,
                          color='blue', label_means=True, cmap='coolwarm',ax=None,eps=1e-4):
    if X is None and P is not None:
        X = P.sample(sample_shape=torch.Size([nsamples])).numpy()

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        xmin, xmax, ymin, ymax = get_plot_ranges(X)
        logger.info(xmin, xmax, ymin, ymax)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()

    XY = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    xx,yy = XY[0,:,:],XY[1,:,:]


    if method == 'samples':
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = scipy.stats.gaussian_kde(X.T)
        f = np.reshape(kernel(positions).T, xx.shape)
    elif method == 'exact':
        μ,Σ = P.loc.numpy(), P.covariance_matrix.numpy()
        f = scipy.stats.multivariate_normal.pdf(XY.transpose(1,2,0),μ,Σ)

    step = 0.01
    levels = np.arange(0, np.amax(f), step) + step

    if len(levels) < 2:
        levels = [step/2, levels[0]]

    cfset = ax.contourf(xx, yy, f, levels, cmap=cmap, alpha=0.5)

    cset = ax.contour(xx, yy, f, levels, colors='k', alpha=0.5)
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if method == 'samples':
        ax.scatter(X[:,0], X[:,1], color=cmap(0.8))
        ax.set_title('2D Gaussian Kernel density estimation')
    elif method == 'exact':
        ax.scatter(μ[0],μ[1], s=5, c= 'black')
        if label_means:
            ax.text(μ[0]+eps,μ[1]+eps,'μ=({:.2},{:.2})'.format(μ[0],μ[1]), fontsize=12)
        ax.set_title('Exact Gaussian Density')


def heatmap(data, row_labels, col_labels, ax=None, cbar=True,
            cbar_kw={}, cbarlabel="", **kwargs):
    """ Create a heatmap from a numpy array and two lists of labels.

    Args:
        data: A 2D numpy array of shape (N, M).
        row_labels: A list or array of length N with the labels for the rows.
        col_labels: A list or array of length M with the labels for the columns.
        ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar: A boolear value, whether to display colorbar or not
        cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel: The label for the colorbar.  Optional.
        **kwargs: All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)


    if cbar:
        if 'alpha' in kwargs:
            cbar_kw['alpha'] = kwargs.get('alpha')
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """ A function to annotate a heatmap.

    Args:
        im: The AxesImage to be labeled.
        data: Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt: The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors: A list or array of two color specifications.  The first is used for
            values below a threshold, the second for those above.  Optional.
        threshold: Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs: All other arguments are forwarded to each call to `text` used to create
            the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def distance_scatter(d, topk=10, show=True, save_path =None):
    """ Distance vs adaptation scatter plots as used in the OTDD paper. 
    Args:
        d (dict): dictionary of task pair (string), distance (float)
        topk (int): number k of top/bottom distances that will be annotated
    """
    sorted_d = sorted(d.items(), key=lambda kv: kv[1])
    keys, dists = zip(*sorted_d)
    if type(keys[0]) is tuple and len(keys[0]) == 2:
        labels  = ['{}<->{}'.format(p,q) for (p,q) in keys]
    else:
        labels  = ['{}'.format(p) for p in keys]
    x_coord = np.linspace(0,1,len(keys))

    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x_coord, dists, s = min(100/len(keys), 1))
    texts=[]
    for i, (x, y, name) in enumerate(zip(x_coord,dists,keys)):
        if i < topk or i >= len(keys) - topk:
            label = '{}<->{}'.format(*name) if type(name) is tuple else str(name)
            texts.append(ax.text(x, y, label))
    adjust_text(texts, force_text=0.05, arrowprops=dict(arrowstyle="-|>",
                                                        color='r', alpha=0.5))

    ax.set_title('Pairwise Distance Between MNIST Binary Classification Tasks')
    ax.set_ylabel('Dataset Distance')
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300) #bbox_inches='tight',
    if show: plt.show()

def dist_adapt_joinplot(df, yvar='delta', show=True, type='joinplot', save_path = None):
    j = sns.jointplot(x='dist', y=yvar, data=df, kind="reg", height=7)
    j.annotate(scipy.stats.pearsonr)
    y_label = 'Acc. Improvement w/ Adapt'#.format(direction[yvar])
    j.set_axis_labels('OT Task Distance', y_label)
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300) #bbox_inches='tight',
    if show:
        plt.show()


def dist_adapt_regplot(df, yvar, xvar='dist', xerrvar=None, yerrvar=None,
                    figsize=(6,5), title=None,
                    show_correlation=True, corrtype='pearson', sci_pval=True,
                    annotate=True, annotation_arrows=True, annotation_fontsize=12,
                    force_text=0.5,
                    legend_fontsize=12,
                    title_fontsize=12,
                    marker_size=10,
                    arrowcolor='gray',
                    barcolor='gray',
                    xlabel = 'OT Dataset Distance',
                    ylabel = r'Relative Drop in Test Error ($\%$)',
                    color='#1f77b4',
                    lw=1,
                    ax=None,
                    show=True,
                    save_path=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        show =  False


    #### Compute Correlation
    if show_correlation:
        if corrtype == 'spearman':
            corr, p = spearmanr(df[xvar], df[yvar])
            corrsymbol = '\\rho'
        elif corrtype == 'pearson':
            corr, p = pearsonr(df[xvar], df[yvar])
            corrsymbol = 'r'
        else:
            raise ValueError('Unrecognized correlation type')
        if p < 0.01 and sci_pval:
            legend_label = r"${}: {:2.2f}$".format(corrsymbol,corr) + "\n" + r"p-value: ${:s}$".format(as_si(p,1))
        else:
            legend_label = r"${}: {:2.2f}$".format(corrsymbol,corr) + "\n" + r"p-value: ${:2.2f}$".format(p)
    else:
        legend_label = None


    ### Actual Plots - First does scatter only, second does line
    sns.regplot(x=xvar, y=yvar, data=df, ax = ax, color=color, label=legend_label,
                scatter_kws={'s':marker_size},
                line_kws={'lw': 1}
                )

    ### Add Error Bars
    if xerrvar or yerrvar:
        xerr = df[xerrvar] if xerrvar else None
        yerr = df[yerrvar] if yerrvar else None
        ax.errorbar(df[xvar], df[yvar], xerr=xerr, yerr=yerr, fmt='none', ecolor='#d6d4d4',  alpha=0.75,elinewidth=0.75)

    ### Annotate Points
    if annotate:
        texts = []
        for i,a in df.iterrows():
            lab = r'{}$\rightarrow${}'.format(a.src,a.tgt) if a.tgt is not None else r'{}'.format(a.src)
            texts.append(ax.text(a[xvar], a[yvar], lab,fontsize=annotation_fontsize))
        if annotation_arrows:
            adjust_text(texts, force_text=force_text, arrowprops=dict(arrowstyle="-", color=arrowcolor, alpha=0.5, lw=0.5))
        else:
            adjust_text(texts, force_text=force_text)

    ### Fix Legend for Correlation (otherwise don't show)
    if show_correlation:
        plt.rc('legend',fontsize=legend_fontsize)#,borderpad=0.2,handletextpad=0, handlelength=0) # using a size in points
        ax.legend([ax.get_lines()[0]], ax.get_legend_handles_labels()[-1],handlelength=1.0,loc='best')#, handletextpad=0.0)


    ### Add title and labels
    ax.set_xlabel(xlabel, fontsize=title_fontsize)
    ax.set_ylabel(ylabel, fontsize=title_fontsize)
    ax.set_title(r'Distance vs Adaptation' + (': {}'.format(title) if title else ''), fontsize=title_fontsize)

    if save_path:
        plt.savefig(save_path+'.pdf', dpi=300, bbox_inches = "tight")
        plt.savefig(save_path+'.png', dpi=300, bbox_inches = "tight")

    if show: plt.show()

    return ax


def plot2D_samples_mat(xs, xt, G, thr=1e-8, ax=None, **kwargs):
    """ (ADAPTED FROM PYTHON OT LIBRARY).
    Plot matrix M  in 2D with  lines using alpha values
    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix G between samples.
    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    b : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    thr : float, optional
        threshold above which the line is drawn
    **kwargs : dict
        paameters given to the plot functions (default color is black if
        nothing given)
    """
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'gray'
    mx = G.max()
    if not ax:
        fig,ax = plt.subplots()
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                        alpha=G[i, j] / mx, **kwargs)

    return ax


def annotate_group(name, span, ax=None, orient='h', side=None):
    """Annotates a span of the x-axis (or y-axis if orient ='v')"""
    if not side:
        side = 'left' if orient == 'v' else 'bottom'
    def annotate(ax, name, left, right, y, pad):
        xy = (left, y) if orient == 'h' else (y, left)
        xytext=(right, y+pad) if orient =='h' else (y+pad, right)
        valign = 'top' if orient =='h' else 'center'
        halign = 'center' if orient == 'h' else 'center'
        rot = 0 if orient == 'h' else 0
        if orient == 'h':
            connectionstyle='angle,angleB=90,angleA=0,rad=5'
        else:
            connectionstyle='angle,angleB=0,angleA=-90,rad=5'

        arrow = ax.annotate(name,
                xy=xy, xycoords='data',
                xytext=xytext, textcoords='data',
                annotation_clip=False, verticalalignment=valign,
                horizontalalignment=halign, linespacing=2.0,
                arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0,
                        connectionstyle=connectionstyle),
                fontsize=8, rotation=rot
                )
        return arrow
    if ax is None:
        ax = plt.gca()
    lims = ax.get_ylim() if orient=='h' else ax.get_xlim()
    range = np.abs(lims[1] - lims[0])
    lim = lims[0] if side == 'bottom' or side == 'left' else lims[1]

    if side == 'bottom':
        arrow_coord = lim + 0.01*range# if orient == 'h' else lim - 0.02*range
        text_pad    = 0.02*range
    elif side == 'right':
        arrow_coord = lim + 0.01*range# if orient == 'h' else lim - 0.02*range
        text_pad    = 0.02*range
    elif side == 'top':
        arrow_coord = lim - 0.01*range# if orient == 'h' else lim - 0.02*range
        text_pad    = -0.05*range
    else: # left
        arrow_coord = lim - 0.01*range
        text_pad    = -0.02*range



    center = np.mean(span)
    left_arrow  = annotate(ax, name, span[0], center, arrow_coord, text_pad)
    right_arrow = annotate(ax, name, span[1], center, arrow_coord, text_pad)
    return left_arrow, right_arrow


def imshow_group_boundaries(ax, gU, gV, group_names, side = 'both', alpha=0.2, lw=0.5):
    """Imshow must be sorted according to order in groups"""
    if side in ['source','both']:
        xmin,xmax = ax.get_xlim()
        ax.hlines(np.cumsum(gU[:-1]) - 0.5,xmin=xmin,xmax=xmax,lw=lw, linestyles='dashed', alpha = alpha)
    if side in ['target','both']:
        ymin,ymax = ax.get_ylim()
        ax.vlines(np.cumsum(gV[:-1]) - 0.5,ymin=ymin,ymax=ymax,lw=lw,linestyles='dashed', alpha=alpha)

    if group_names:
        offset = -0.5
        posx = np.cumsum(gU)# + offset
        posy = np.cumsum(gV)# + offset
        posx = np.insert(posx, 0, offset)
        posy = np.insert(posy, 0, offset)
        for i,y in enumerate(posy[:-1]):
            annotate_group(group_names[1][i], (posy[i], posy[i+1]), ax, orient='h', side = 'top')
        for i,x in enumerate(posx[:-1]):
            annotate_group(group_names[0][i], (posx[i], posx[i+1]), ax, orient='v', side = 'right')


def method_comparison_plot(df, hue_var = 'method', style_var = 'method',
                          figsize = (15,4), ax = None, save_path=None):
        """ Produce plots comparing OTDD variants in terms of runtime and distance """
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=figsize)

        lplot_args = {
            'hue': hue_var,
            'style': style_var,
            'data': df,
            'x': 'n',
            'markers': True
        }

        sns.lineplot(y='dist', ax= ax[0], **lplot_args)
        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
        ax[0].set_ylabel('Dataset Distance')
        ax[0].set_xlabel('Dataset Size')
        ax[0].set_xscale("log")
        ax[0].grid(True,which="both",ls="--",c='gray')

        sns.lineplot(y='time', ax= ax[1], **lplot_args)
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
        ax[1].set_ylabel('Runtime (s)')
        ax[1].set_xlabel('Dataset Size')
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].grid(True,which="both",ls="--",c='gray')

        handles, labels = ax[1].get_legend_handles_labels()
        ax[1].get_legend().remove()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + '.pdf',  dpi=300)
            plt.savefig(save_path + '.png',  dpi=300)
        plt.show()

        return ax
