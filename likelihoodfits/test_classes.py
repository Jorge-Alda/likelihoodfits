import likelihoodfits
import numpy as np
from io import BytesIO
import pytest


class TestAxis:
    ax = likelihoodfits.Axis([2.0, 3.0, 7.0, 5.0], 'test', 't')

    def test_len(self):
        assert self.ax.len == 4

    def test_max(self):
        assert self.ax.max == 7.0

    def test_min(self):
        assert self.ax.min == 2.0

    def test_item(self):
        assert self.ax[3] == 5.0

    def test_wrong_item(self):
        with pytest.raises(IndexError):
            self.ax[20]


class TestLikelihoodValues:
    lhv = likelihoodfits.LikelihoodValues(
        np.zeros((5, 5), dtype=np.float32), 'test', 'test')

    def test_shape(self):
        assert self.lhv.shape == (5, 5)

    def test_max(self):
        self.lhv.data[2, 3] = 27.5
        assert self.lhv.max == 27.5


class TestLikelihoodResults:
    x = likelihoodfits.Axis([0, 1, 2, 3, 4], 'x', 'x')
    y = likelihoodfits.Axis([6, 7, 8, 9, 10], 'y', 'y')
    lhr = likelihoodfits.LikelihoodResults(x, y)

    def test_numdata(self):
        assert self.lhr.numdata == 25

    def test_add_likelihood(self):
        lh_rand = likelihoodfits.LikelihoodValues(
            np.random.random((5, 5)), 'random', 'rnd')
        self.lhr.add_likelihood(lh_rand)
        assert self.lhr.likelihoods[-1].likelihood == 'random'

    def test_wrong_add_likelihood(self):
        lh_rand = likelihoodfits.LikelihoodValues(
            np.random.random((6, 5)), 'random', 'rnd')
        with pytest.raises(ValueError):
            self.lhr.add_likelihood(lh_rand)

    def test_new_likelihood(self):
        self.lhr.new_likelihood('testnew', 'testnew')
        assert self.lhr.likelihoods[-1].likelihood == 'testnew'

    def test_calculate_point(self):
        def fun(x, y): return {'testnew': x*y, 'random': 4}
        self.lhr.new_likelihood('testnew', 'testnew')
        self.lhr.calculate_point(fun, 3, 1)
        assert self.lhr.likelihoods[-1].data[1, 3] == 21  # Está transpuesto

    def test_calculate_all(self):
        def fun(x, y): return {'testnew': x*y, 'random': 4}
        self.lhr.new_likelihood('testnew', 'testnew')
        self.lhr.calculate_all(fun)
        assert self.lhr.likelihoods[-1].data[2, 3] == 24

    def test_calculate_missing_lh(self):
        def fun(x, y): return {'testold': x*y, 'random': 4}
        self.lhr.new_likelihood('testnew', 'testnew')
        with pytest.raises(KeyError):
            self.lhr.calculate_point(fun, 3, 1)

    def test_save(self):
        def fun(x, y): return {'testnew': x*y, 'random': 4}
        self.lhr.new_likelihood('testnew', 'testnew')
        self.lhr.calculate_all(fun)
        try:
            with BytesIO() as file:
                self.lhr.to_hdf5(file)
        except:
            saved = False
        else:
            saved = True
        assert saved

    def test_load(self):
        def fun(x, y): return {'testnew': x*y, 'random': 4}
        self.lhr.new_likelihood('testnew', 'testnew')
        self.lhr.calculate_all(fun)
        with BytesIO() as file:
            self.lhr.to_hdf5(file)
            lh2 = likelihoodfits.LikelihoodResults.from_hdf5(file)
        assert lh2.likelihoods[0].data[2, 3] == 24
