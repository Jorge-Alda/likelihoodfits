from typing import Sequence
import numpy as np
import scipy.stats
import scipy.ndimage
import matplotlib.cm
import matplotlib.pyplot as plt
from likelihoodfits.classes import LikelihoodResults

plt.rcParams.update({'pgf.texsystem': 'pdflatex'})
ColorTuple = tuple[float, float, float]

pastel: Sequence[ColorTuple] = matplotlib.cm.get_cmap('tab10').colors


def delta_chi2(nsigma: float, dof: int) -> float:
    r"""Compute the $\Delta\chi^2$ for `dof` degrees of freedom corresponding
    to `nsigma` Gaussian standard deviations.

    Example: For `dof=2` and `nsigma=1`, the result is roughly 2.3."""
    if dof == 1:
        # that's trivial
        return nsigma**2
    chi2_ndof = scipy.stats.chi2(dof)
    cl_nsigma = (scipy.stats.norm.cdf(nsigma)-0.5)*2
    return chi2_ndof.ppf(cl_nsigma)


def plot(lh: LikelihoodResults,
         *, margin: float = 0.0, n_sigma: Sequence[float] = (1.0, 2.0),
         zoom: float = 5.0, loc: str = 'best', numticks: int = 6,
         palette: Sequence[ColorTuple] = pastel):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    xmin = lh.x.min + abs(lh.x.min*margin)
    xmax = lh.x.max - abs(lh.x.max*margin)
    ymin = lh.y.min + abs(lh.y.min*margin)
    ymax = lh.y.max - abs(lh.y.max*margin)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    x = scipy.ndimage.zoom(lh.x.ticks, zoom=zoom, order=1)
    y = scipy.ndimage.zoom(lh.y.ticks, zoom=zoom, order=1)
    proxies = []
    legends = []
    for i, l in enumerate(lh.likelihoods):
        chi = -2 * (l.data - l.max)
        levels = [delta_chi2(n, dof=2) for n in n_sigma]
        N = len(levels)
        z = scipy.ndimage.zoom(chi, zoom=zoom, order=2)
        colori = palette[i % len(palette)]
        colorf = [colori + (max(1-n/(N+1), 0),) for n in range(1, N+1)]
        ax.contourf(x, y, z, levels=[0, ]+levels, colors=colorf)
        ax.contour(x, y, z, levels=levels, colors=[colori])
        proxies.append(plt.Rectangle(
            (0, 0), 1, 1, fc=colori + (1-1/(N+1),)))
        legends.append(l.tex_label)
    plt.xlabel(lh.x.tex_label, fontsize=18)
    plt.ylabel(lh.y.tex_label, fontsize=18)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks(np.linspace(xmin, xmax+1e-10, numticks))
    ax.yaxis.set_ticks(np.linspace(ymin, ymax+1e-10, numticks))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(proxies, legends, fontsize=16, loc=loc)
    plt.tight_layout(pad=0.5)
    return fig
