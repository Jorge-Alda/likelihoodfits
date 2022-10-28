from likelihoodfits import *
import numpy as np
from rich.progress import Progress
from rich.text import Text
from rich.live import Live
import smelli


NDATA = 50


def calculate_points(path: str):
    textpr = Text("Progress: ", style="bold red")
    with Live(textpr, refresh_per_second=4) as live:
        msg = textpr.copy().append(
            "Preparing data structures",  style="default not bold")
        live.update(msg)
        x1 = Axis(np.linspace(0.0, 0.08, NDATA), 'x1', r'$x_1$')
        x3 = Axis(np.linspace(0.0, 0.8, NDATA), 'x3', r'$x_3$')
        lhr = LikelihoodResults(x1, x3)
        lhr.new_likelihood('likelihood_lfu_fcnc.yaml',
                           r'$b\to s \ell^+\ell^-$ LFU')
        lhr.new_likelihood('likelihood_rd_rds.yaml', r'$b \to c \ell \nu$ LFU')
        lhr.new_likelihood('likelihood_lfv.yaml', 'LFV')
        lhr.new_likelihood('likelihood_ewpt.yaml', 'EW precision')
        lhr.new_likelihood('global', 'Global')
        msg = textpr.copy().append(
            "Generating the likelihood object",  style="default not bold")
        live.update(msg)
        gl = smelli.GlobalLikelihood()

        def fun(x, y): return gl.parameter_point(
            wc.lqU1_simple(x, y, 1500)).log_likelihood_dict()
        progress = Progress()
        task = progress.add_task("[red]Calculating...", total=lhr.numdata)
        live.update(progress)
        for _ in lhr.calculate(fun):
            progress.update(task, advance=1)
        msg = textpr.copy().append(
            f"Saving to {path}",  style="default not bold")
        live.update(msg)
        lhr.to_hdf5(path)
        msg = textpr.copy().append("Finished!",  style="green not bold")
        live.update(msg)


def makeplot():
    lh = LikelihoodResults.from_hdf5('data/lqU1_simple.hdf5')
    fig = plot.plot(lh, loc='lower right')
    fig.savefig('data/lqU1.pdf')
