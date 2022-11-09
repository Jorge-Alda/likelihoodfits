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

    def test_numdata(self):
        lhr = likelihoodfits.LikelihoodResults(self.x, self.y)
        assert lhr.numdata == 25

    def test_add_likelihood(self):
        lhr = likelihoodfits.LikelihoodResults(self.x, self.y)
        lh_rand = likelihoodfits.LikelihoodValues(
            np.random.random((5, 5)), 'random', 'rnd')
        lhr.add_likelihood(lh_rand)
        assert lhr.likelihoods[-1].likelihood == 'random'

    def test_wrong_add_likelihood(self):
        lhr = likelihoodfits.LikelihoodResults(self.x, self.y)
        lh_rand = likelihoodfits.LikelihoodValues(
            np.random.random((6, 5)), 'random', 'rnd')
        with pytest.raises(ValueError):
            lhr.add_likelihood(lh_rand)

    def test_new_likelihood(self):
        lhr = likelihoodfits.LikelihoodResults(self.x, self.y)
        lhr.new_likelihood('testnew', 'testnew')
        assert lhr.likelihoods[-1].likelihood == 'testnew'

    def test_calculate_point(self):
        lhr = likelihoodfits.LikelihoodResults(self.x, self.y)
        def fun(x, y): return {'testnew': x*y}
        lhr.new_likelihood('testnew', 'testnew')
        lhr.calculate_point(fun, 3, 1)
        assert lhr.likelihoods[-1].data[1, 3] == 21  # Está transpuesto

    def test_calculate_all(self):
        lhr = likelihoodfits.LikelihoodResults(self.x, self.y)
        def fun(x, y): return {'testnew': x*y}
        lhr.new_likelihood('testnew', 'testnew')
        lhr.calculate_all(fun)
        assert lhr.likelihoods[-1].data[2, 3] == 24

    def test_calculate_missing_lh(self):
        lhr = likelihoodfits.LikelihoodResults(self.x, self.y)
        def fun(x, y): return {'testold': x*y}
        lhr.new_likelihood('testnew', 'testnew')
        with pytest.raises(KeyError):
            lhr.calculate_point(fun, 3, 1)

    def test_save(self):
        lhr = likelihoodfits.LikelihoodResults(self.x, self.y)
        def fun(x, y): return {'testnew': x*y}
        lhr.new_likelihood('testnew', 'testnew')
        lhr.calculate_all(fun)
        try:
            with BytesIO() as file:
                lhr.to_hdf5(file)
        except IOError as e:
            print(e)
            saved = False
        else:
            saved = True
        assert saved

    def test_load(self):
        lhr = likelihoodfits.LikelihoodResults(self.x, self.y)
        def fun(x, y): return {'testnew': x*y}
        lhr.new_likelihood('testnew', 'testnew')
        lhr.calculate_all(fun)
        with BytesIO() as file:
            lhr.to_hdf5(file)
            lh2 = likelihoodfits.LikelihoodResults.from_hdf5(file)
        assert lh2.likelihoods[0].data[2, 3] == 24
