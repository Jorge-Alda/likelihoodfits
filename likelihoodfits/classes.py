from math import isinf
from typing import Callable, Self
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
import numpy as np
import h5py

LikelihoodFunction = Callable[[float, float], dict[str, float]]


@dataclass
class Axis:
    """Class containing an axis, subscriptable

    Arguments
    ---------
        ticks (`Sequence[float]`): Points at which the likelihood will 
        be computed

        name (`str`): Name of the axis

        tex_label (`str`): TeX representation of the name of the axis
    """

    ticks: Sequence[float]
    name: str
    tex_label: str

    @property
    def len(self) -> int:
        """Length of the axis
        """
        return len(self.ticks)

    @property
    def min(self) -> float:
        """Minimum of the axis
        """
        return min(self.ticks)

    @property
    def max(self) -> float:
        """Maximum of the axis
        """
        return max(self.ticks)

    def __getitem__(self, item: int) -> float:
        return self.ticks[item]


@dataclass
class LikelihoodValues:
    """Class containing a grid of log-likelihood values

    Arguments
    ---------
        data (`np.ndarray`): Grid of log-likelihood values

        likelihood (`str`): Name of the likelihood used by smelli

        tex_label (`str`): TeX representation of the name of the
        likelihood

        order (`int`, optional): Order in which the likelihood will
        be plotted
    """

    data: np.ndarray
    likelihood: str
    tex_label: str
    order: int = field(kw_only=True, default=0)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the likelihood data

        Returns
        -------
            tuple[int, int]: Dimensions of the data
        """
        return self.data.shape

    @property
    def max(self) -> float:
        """Maximum likelihood
        """
        return np.max(self.data)


@dataclass
class LikelihoodResults:
    """Class containing all the data needed to make a likelihood plot
    
    Arguments
    ---------
    x (`Axis`): x-axis

    y (`Axis`): y-axis

    likelihoods (`list[LikelihoodValues]`, initialized to `[]`):
    List of likelihood values 
    """

    x: Axis
    y: Axis
    likelihoods: list[LikelihoodValues] = field(
        init=False, default_factory=list)

    @property
    def numdata(self) -> int:
        """Number of data in each likelihood

        Returns
        -------
            int: Total number of data in each likelihood
        """
        return self.x.len * self.y.len

    def add_likelihood(self, lh: LikelihoodValues) -> None:
        """Adds an already-calculated likelihood

        Arguments
        ---------
            lh (LikelihoodValues): Likelihood to be added

        Raises
        ------
            ValueError:
            The dimension of the data and the axis do not match
        """

        if lh.shape[0] != self.x.len:
            raise ValueError(
                "The dimension of the data and the axis do not match along the x direction")
        if lh.shape[1] != self.y.len:
            raise ValueError(
                "The dimension of the data and the axis do not match along the y direction")
        self.likelihoods.append(lh)

    def to_hdf5(self, path: str) -> None:
        """Saves to an HDF5 file

        Arguments
        ----------
            path (str): Path to the HDF5 file
        """
        with h5py.File(path, 'w') as f:
            axes = f.create_group('axes')
            axes.create_dataset('x', data=self.x.ticks, dtype='f4')
            axes.create_dataset('y', data=self.y.ticks, dtype='f4')
            axes.attrs.update({'x name': self.x.name,
                              'x tex': self.x.tex_label,
                               'y name': self.y.name,
                               'y tex': self.y.tex_label})
            likelihoods = f.create_group('likelihoods')
            for l in self.likelihoods:
                gr = likelihoods.create_group(l.likelihood)
                gr.attrs.update({'tex': l.tex_label, 'order': l.order})
                gr.create_dataset('values', data=l.data, dtype='f4')

    @classmethod
    def from_hdf5(cls, path: str) -> Self:
        """Loads likelihood data from a HDF5 file

        Argsuments
        ----------
            path (str): Path to the HDF5 file

        Returns
        -------
            LikelihoodResults: Loaded data
        """

        with h5py.File(path, 'r') as f:
            xdata = np.array(f['axes']['x'], dtype='f4')
            x = Axis(xdata, f['axes'].attrs['x name'],
                     f['axes'].attrs['x tex'])
            ydata = np.array(f['axes']['y'], dtype='f4')
            y = Axis(ydata, f['axes'].attrs['y name'],
                     f['axes'].attrs['y tex'])
            results = LikelihoodResults(x, y)
            for k in f['likelihoods'].keys():
                data = np.array(f['likelihoods'][k]['values'], dtype='f4')
                lh = LikelihoodValues(
                    data, k, f['likelihoods'][k].attrs['tex'], order=f['likelihoods'][k].attrs['order'])
                results.add_likelihood(lh)
                results.likelihoods.sort(key=lambda x: x.order)
        return results

    def new_likelihood(self, likelihood: str, tex_label: str) -> None:
        """Creates a new empty likelihood

        Arguments
        ---------
            likelihood (`str`): Name of the likelihood used by smelli

            tex_label (`str`): TeX representation of the name of
            the likelihood
        """

        lh = LikelihoodValues(np.zeros((self.x.len, self.y.len), dtype='f4'),
                              likelihood, tex_label,
                              order=len(self.likelihoods))
        self.likelihoods.append(lh)

    def calculate_point(self, fun: LikelihoodFunction, ix: int, iy: int):
        """Calculates the likelihoods at one point of the grid

        Arguments
        ---------
            fun (Function(float, float) -> dict[str, float]):
            Function that calculates the likelihoods at each point,
            and returns a dictionary with the likelihood names and
            values.
            The likelihood names must be the same as in the definition
            of the LikelihoodValues objects.

            ix (int): x-index of the point.

            iy (int): y-index of the point
        """
        x = self.x[ix]
        y = self.y[iy]
        lhdict = fun(x, y)
        for l in self.likelihoods:
            lval = lhdict[l.likelihood]
            if isinf(lval):
                lval = -200.0
            l.data[iy, ix] = lval

    def calculate_all(self, fun: LikelihoodFunction):
        for ix, iy in product(range(self.x.len), range(self.y.len)):
            self.calculate_point(fun, ix, iy)
