import pytest
import numpy as np
from cluster import KMeans

def test_kmeans():
    """
    Tests to make sure that the KMeans clustering works on a trivial example.
    """
    pass


def test_kmeans_constructor_exceptions():
    """
    Tests to make sure that the KMeans constructor correctly raises exceptions on bad inputs.
    """
    # check for negative k
    with pytest.raises(ValueError):
        km = KMeans(k=-5)
    # check for negative tol
    with pytest.raises(ValueError):
        km = KMeans(tol=-1)
    # check for negative max_iter
    with pytest.raises(ValueError):
        km = KMeans(max_iter=-1)


def test_kmeans_fit_exceptions():
    """
    Tests to make sure that the KMeans fit method correctly raises exceptions on bad inputs.
    """
    km = KMeans(k=4)
    # check empty input data
    with pytest.raises(Exception):
        km.fit(np.empty(5))
    # check non-numeric input data
    with pytest.raises(Exception):
        km.fit(np.array([char for char in 'ABCDEF']))
    # check input data smaller than k
    with pytest.raises(Exception):
        km.fit(np.random.rand(3, 3))


def test_kmeans_predict_exceptions():
    """
    Tests to make sure that the KMeans predict method correctly raises exceptions on bad inputs.
    """
    km = KMeans(k=4)
    km.fit(np.random.rand(10, 3))

    # check empty input data
    with pytest.raises(Exception):
        km.predict(np.empty(5))
    # check non-numeric input data
    with pytest.raises(Exception):
        km.predict(np.array([char for char in 'ABCDEF']))