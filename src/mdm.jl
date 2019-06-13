
"""
This is a structure of the MDM model. It has two attributes namely metric and
class_means.
- **metric :: Metric**        :- The user needs to specify the metric space in which all the distance
  computations is to be done. The possible options are:

| Metric   | Mean estimation |
|:---------- |:----------- |
| Euclidean  | distance: δ_e; mean: Arithmetic |
| invEuclidean | distance: δ_i; mean: Harmonic |
| ChoEuclidean | distance: δ_c; mean: Cholesky Euclidean |
| logEuclidean | distance: δ_l; mean: Log Euclidean |
| logCholesky  | distance: δ_c; mean: Log-Cholesky |
| Fisher       | distance: δ_f; mean: Fisher (Cartan, Karcher, Pusz-Woronowicz,...) |
| logdet0      | distance: δ_s; mean: LogDet (S, α, Bhattacharyya, Jensen,...) |
| Jeffrey      | distance: δ_j; mean: Jeffrey (symmetrized Kullback-Leibler) |
| VonNeumann   | distance: δ_v; mean: Not Availale |
| Wasserstein  |   distance: δ_w; mean: Wasserstein (Bures, Hellinger, ...) |

- **class_means**             :- This is not to be specified by the user. This comes to play when
                                a model is fit with a set of training data so we have to store the means corresponding
                                to each of the classes. We can directly access this for an already fit model
                                while using functions like predict.


"""
mutable struct MDM
    metric :: Metric
    class_means
    function MDM(metric :: Metric; class_means = nothing)
        new(metric, class_means)
    end
end


"""
This function is kind of an interface to the mean function of PostDefManifold
for computing means for mdm classifier whith metrics other than Fisher, logdet0
and Wasserstein. For these three types of metrics specially it is an interface to
their respective mean functions in PostDefManifold. Since the above three metrics
do not have any closed form solution for estimating the mean, hence we follow
iterative algorithms to compensate for that. As iterative algorithms are
always accompanied by the question of convergence, so, in order to answer
that we need a convergence check for these three metric types which is
facilitated by their respective mean functions. Thereby calling them instead
of the generalised mean function specifically for these three kinds of metric.

Arguments taken by this function are:

- **metric :: Metric**        :-The user needs to specify the metric space in which
                                all the distance computations is to be done.
- **𝐏::ℍVector**              :-Vector of Hermitian matrices or simply a HermitianVector.
                                     The vector of points in the *Symmetric Positive Definite*
                                     manifold to be transformed into the the tangent space.
- **w::Vector(optional)**:- Vector containing weights corresponding to every point in 𝐏.
- **✓w = true(optional)**    :- Boolean to determine whether to calculate weighted mean or just take w = [].
- **⏩ = false (optional)**   :- Boolean to allow threading or not.

Return value :

- **G :: ℍ**                  :- Mean of the set of ℍVector.


"""
function mean_mdm(metric::Metric, 𝐏::ℍVector;
              w::Vector=[],
              ✓w=true,
              ⏩=false)
    tolerance=100*√eps(real(eltype(𝐏[1])))
    if metric == Fisher
        (G, iter, conv) = gMean(𝐏; w=w, ✓w=✓w, tol=tolerance, ⏩=⏩)
        if conv > tolerance
            @warn "The given data set does not converge to a mean value.
            It is not expected to give decent results. If possible try to pick a
            healthy data set."
        end
        return G
    elseif metric == logdet0
        (G, iter, conv) = ld0Mean(𝐏; w=w, ✓w=✓w, tol=tolerance, ⏩=⏩)
        if conv > tolerance
            @warn "The given data set does not converge to a mean value.
            It is not expected to give decent results. If possible try to pick a
            healthy data set."
        end
        return G
    elseif metric == Wasserstein
        (G, iter, conv) = wasMean(𝐏; w=w, ✓w=✓w, tol=tolerance, ⏩=⏩)
        if conv > tolerance
            @warn "The given data set does not converge to a mean value.
            It is not expected to give decent results. If possible try to pick a
            healthy data set."
        end
        return G
    else return mean(metric, 𝐏, w = w, ✓w = ✓w, ⏩ = ⏩)
    end
end



"""
This is a function to find the distance of each sample case from the so found means
of all the classes. Distance is caluclated in the metric space opted by the user
while creating the instance of mdm.

Arguments taken by this function are:

- **sample :: ℍVector**       :-The vector of Hermitian matrices or points in
                                       the positive definite manifold for which the
                                       prediction is to be made using the model(argument 1)
                                       already been trained specially for it.
- **class_means**             :- Vector of Hermitian matrices that represent the class
                                means or the centroid of all the classes in the training set.
- **metric :: Metric**        :-The user needs to specify the metric space in which
                                all the distance computations is to be done.

Return value:

- **A :: Array{Float64,2}**   :- Array of distances of each of the sample case from
                                    each of the class_means.


"""
function find_dist(sample, class_means, metric :: Metric)
    A = [distance(metric,sample[1], m) for m in class_means ]
    for i = 2:length(sample)
        a = [distance(metric,sample[i], m) for m in class_means ]
        A = hcat(A, a)
    end
    return A
end
"""
This is a function to predict the class for each sample case depending on its distance
from the respective means. The class associated with the closest mean of the sample
is alloted to that sample case. So the name comes Minimum distance to Mean as the
mean which is at the minimum distance to the sample decides the class of the sample.

Arguments taken by this function are:

- **sample :: ℍVector**       :- The vector of Hermitian matrices or points in
                                           the positive definite manifold for which the
                                           prediction is to be made using the model(argument 1)
                                           already been trained specially for it.
- **class_means**             :- Vector of Hermitian matrices that represent the class
                                    means or the centroid of all the classes in the training set.
- **metric :: Metric**        :- The user needs to specify the metric space in which
                                all the distance computations is to be done.

Return value:

- **result :: Array{Int, 1}** :- The List of the predicted classes for the given
                                    sample set.


"""
function predict_mdm(sample, class_means, metric:: Metric)
    A = find_dist(sample, class_means, metric)
    result = [findmin(A[:,j])[2] for j = 1:dim(A,2)]
    println(result)
    return result
end



"""
This is a function to predict the probability of each class for each sample case
depending on its distance from the respective means. The class associated with the
closest mean is having the highest probability. All the probability sums to 1.
This function makes use of the softmax function from thr PostDefManifold to
calculate the probability values.

Arguments taken by this function are:

- **sample :: ℍVector**       :-The vector of Hermitian matrices or points in
                                           the positive definite manifold for which the probability
                                           prediction is to be made using the model(argument 1)
                                           already been trained specially for it.
- **class_means**             :- Vector of Hermitian matrices that represent the class
                                    means or the centroid of all the classes in the training set.
- **metric :: Metric**        :-The user needs to specify the metric space in which
                                all the distance computations is to be done.

Return value:

- **Prob :: Array{Float64,1}**:- The List of the predicted probabilities corresponding
                                    to each of the classes for the given sample set.


"""
function predict_prob(sample, class_means, metric :: Metric)
    A = find_dist(sample, class_means, metric)
    Prob = [softmax(-A[:,j]) for j = 1:dim(A,2)]
    println(Prob)
    return Prob
end
