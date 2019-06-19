# functions imported from ScikitLearn.jl
#@sk_import neighbors: KNeighborsClassifier
@sk_import neighbors: KNeighborsClassifier
"""
Structure for implementing KNeighborsClassifier. The `clf` attribute is the python object of KNeighborsClassifier that is
constructed by the constructor function of this structure as soon as a kneighborClf
instance is created by the user.

Parameters :

- n_neighbors :: Int64
- weights:: String
- algorithm:: String
- leaf_size:: Int64
- p:: Int64
- metric:: String
- metric_params :: Dict
- n_jobs :: Int

One may refer to the ScikitLearn documentation of KNeighborsClassifier for the better understanding of these parameters with respect to kneighborClf.

    ## Example
    using RiemannianML
    model = kneighborClf(n_neighbors=3)



"""
mutable struct kneighborClf
    clf
    n_neighbors :: Int64
    weights:: String
    algorithm:: String
    leaf_size:: Int64
    p:: Int64
    metric:: String
    metric_params
    n_jobs
    function kneighborClf(;n_neighbors=5,
                            weights="uniform",
                            algorithm="auto",
                            leaf_size=30, p=2,
                            metric="minkowski",
                            metric_params= nothing,
                            n_jobs= nothing )
        clf = KNeighborsClassifier(n_neighbors, weights, algorithm,
                                    leaf_size, p, metric, metric_params, n_jobs)
        new(clf, n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)
    end
end
