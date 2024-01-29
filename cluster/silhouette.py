import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if X.dtype != float and X.dtype != int:
            raise ValueError('X contains non-numerical data. Silhouette scoring with Euclidean distance only works on numerical data.')
        if X.shape[0] != y.shape[0]:
            raise Exception('X and y are different lengths.')
        
        # convert unique labels in y to numbers, then to 1-hot matrix
        lbl_map = {lbl: idx for idx, lbl in enumerate(np.unique(y))}
        y_sanitized = list(map(lambda x: lbl_map[x], y))
        y_1hot = np.eye(len(lbl_map))[y_sanitized]

        # compute average intracluster distances
        dists = cdist(X, X)
        a = np.sum(dists*np.matmul(y_1hot,y_1hot.T), axis=0) / \
            np.sum((y_1hot * (np.sum(y_1hot, axis=0).T - 1).T), axis=1)
        
        # compute average distances to other clusters and take the minimum
        b = np.min(
            np.matmul(dists, y_1hot) / np.sum(y_1hot, axis=0), 
            axis=1, initial=np.inf, where=(y_1hot == 0))

        # compute silhouette scores
        s = (b - a) / np.maximum(a, b)

        return s