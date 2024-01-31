import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if k <= 1:
            raise ValueError("This k doesn't make sense. k should be an integer greater than 1.")
        if tol < 0:
            raise ValueError("This tolerance doesn't make sense. tol should be a float greater than or equal to 0.")
        if max_iter <= 0:
            raise ValueError("This max_iter doesn't make sense. max_iter should be an integer greater than 0.")
                
        self._k = k
        self._tol = tol
        self._max_iter = max_iter

        self._mu = None
        self._error = None

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if self._k >= len(mat):
            raise Exception('This dataset is smaller than the number of clusters k.')
        
        if mat.size == 0:
            raise Exception('The input matrix is empty.')
        elif mat.dtype != float and mat.dtype != int:
            raise Exception('The input matrix contains non-numerical data. KMeans only works on numerical data.')
                
        num_iter = 0
        sse = np.inf
        delta_sse = np.inf

        # initialize centroids to be on random data points
        #mu = mat[np.random.choice(mat.shape[0], size=self._k, replace=False), :]
        
        # initialize centroids using kmeans++
        dists = cdist(mat, mat)
        mu_idxs = [np.random.randint(mat.shape[0])]
        for _ in range(self._k - 1):
            probs = dists[mu_idxs[-1], :]**2 / np.sum(dists[mu_idxs[-1], :]**2)
            mu_idxs.append(np.random.choice(mat.shape[0], p=probs))
        mu = mat[mu_idxs, :]

        while num_iter < self._max_iter and delta_sse > self._tol:
            r = cdist(mat, mu)

            # get argmin of each row of r and make 1-hot
            r = np.equal(r, np.min(r, axis=1, keepdims=True))

            # update centroids based on 1-hot r
            mu = np.matmul(r.T,mat) / np.sum(r, axis=0, keepdims=True).T

            new_sse = np.sum(cdist(mat, mu)**2 * r)
            if num_iter == 0:
                delta_sse = new_sse
            else:
                delta_sse = abs(sse - new_sse)
            sse = new_sse
            num_iter += 1

        self._mu = mu
        self._error = sse
        

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self._mu.all() == None:
            raise Exception("This instance of KMeans hasn't been fit yet.")
        if mat.size == 0:
            raise Exception('The input matrix is empty.')
        elif mat.dtype != float and mat.dtype != int:
            raise Exception('The input matrix contains non-numerical data. KMeans only works on numerical data.')
        
        return np.argmin(cdist(mat, self._mu), axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self._error == None:
            raise Exception("This instance of KMeans hasn't been fit yet.")

        return self._error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self._mu is None:
            raise Exception("This instance of KMeans hasn't been fit yet.")

        return self._mu