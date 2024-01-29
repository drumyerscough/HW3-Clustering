import pytest
import numpy as np
from cluster.silhouette import Silhouette
from sklearn.metrics import silhouette_score

def test_silhouette():
    """
    Tests to make sure that the silhouette test works on an example 
    by comparing output to sklearn implementation.
    """

    # generate a random sample of 50 data points with 3 feature dimensions
    # and 4 labels
    X = np.random.rand(50, 3)
    y = np.random.randint(5, size=50)

    assert np.mean(Silhouette().score(X, y)) == silhouette_score(X, y)

def test_silhouette_exceptions():
    """
    Tests to make sure that the silhouette test raises exceptions on bad inputs.
    """

    # generate a random sample of 50 data points with 3 feature dimensions
    # and 4 labels
    X = np.random.rand(50, 3)
    y = np.random.randint(5, size=50)

    # check empty input data
    with pytest.raises(Exception):
        Silhouette().score(np.empty(50), y)
    # check non-numeric input data
    with pytest.raises(ValueError):
        Silhouette().score(np.array([char for char in 'ABCDEF']), np.random.randint(5, size=6))
    # check different lengths for X and y
    with pytest.raises(Exception):
        Silhouette().score(np.random.rand(5, 3), y)

