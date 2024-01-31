import pytest
import numpy as np
from cluster import KMeans, make_clusters

def test_kmeans():
    """
    Tests to make sure that the KMeans clustering runs on an example. 
    This doesn't necessarily mean it works, but since KMeans is stochastic and often bad,
    it's hard to guarantee that a non-trivial example would work.
    """
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)

    # if km is initialized but doesn't run and doesn't raise exceptions, this will fail
    assert km.get_error() != None
    assert km.get_centroids() != None


def test_kmeans_constructor_exceptions():
    """
    Tests to make sure that the KMeans constructor correctly raises exceptions on bad inputs.
    """
    # check for negative k
    with pytest.raises(ValueError):
        km = KMeans(k=-5)
    # check for negative tol
    with pytest.raises(ValueError):
        km = KMeans(k=5, tol=-1)
    # check for negative max_iter
    with pytest.raises(ValueError):
        km = KMeans(k=5, max_iter=-1)


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