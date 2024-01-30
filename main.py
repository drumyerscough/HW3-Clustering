import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)


def main():

    # create tight clusters
    clusters, labels = make_clusters(scale=0.3)
    plot_clusters(clusters, labels, filename="figures/tight_clusters.png")

    # create loose clusters
    clusters, labels = make_clusters(scale=2)
    plot_clusters(clusters, labels, filename="figures/loose_clusters.png")
    
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    #km = KMeans(n_clusters=4,init='random', n_init=1, max_iter=300, tol=0.0001, verbose=1, random_state=r)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    plot_multipanel(clusters, labels, pred, scores)


if __name__ == "__main__":
    main()
