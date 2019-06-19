using LinearAlgebra, Statistics, Base.Threads, PosDefManifold, ScikitLearn, Random

"""
This is a structure of the MDM model. It has two attributes namely metric and
class_means.
- `metric :: Metric`        :- The user needs to specify the metric space in which all the distance
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

- `class_means`             :- This is not to be specified by the user. This comes to play when
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
Given a metric, ℍVector( Vector of Hermitian matrices), weights(optional)
returns the mean of these Hermitian matrices for mdm classifier.This function
is kind of an interface to the mean function of PostDefManifold. Refer to the
[Mean](https://marco-congedo.github.io/PosDefManifold.jl/latest/riemannianGeometry/#Means-1)
of PosDefManifold.

Arguments :

- `metric :: Metric`        :-The user needs to specify the metric space in which
                                all the distance computations is to be done.
- `𝐏::ℍVector`              :-Vector of Hermitian matrices or simply a HermitianVector.
                                     The vector of points in the *Symmetric Positive Definite*
                                     manifold to be transformed into the the tangent space.
- `w::Vector(optional)`:- Vector containing weights corresponding to every point in 𝐏.
- `✓w = true(optional)`    :- Boolean to determine whether to calculate weighted mean or
                                 just take w = []. It also checks if the weights
                                 sum up to 1. If not does normalization.
- `⏩ = false (optional)`   :- Boolean to allow threading or not.

Returns :

- `G :: ℍ`                  :- Mean of the set of ℍVector.


"""
function mean_mdm(metric::Metric, 𝐏::ℍVector;
              w::Vector=[],
              ✓w=true,
              ⏩=false)
    tolerance=√eps(real(eltype(𝐏[1])))
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
Given a sample set, class_means, type of metric returns the distance of each sample case
from the points in class_means. Distance should be caluclated in the metric space opted by the user
while creating the instance of mdm.

Arguments :

- `sample :: ℍVector`       :-The vector of Hermitian matrices or points in
                                       the positive definite manifold for which the
                                       prediction is to be made using the model(argument 1)
                                       already been trained specially for it.
- `class_means`             :- Vector of Hermitian matrices that represent the class
                                means or the centroid of all the classes in the training set.
- `metric :: Metric`        :-The user needs to specify the metric space in which
                                all the distance computations is to be done.

Return :

- `A :: Array{Float64,2}`   :- Array of distances of each of the sample case from
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
Given a sample set, class_means, type of metric returns the predicted classes for
each sample case according to the Minimum Distance to the Means scheme.

Arguments :

- `sample :: ℍVector`       :- The vector of Hermitian matrices or points in
                                           the positive definite manifold for which the
                                           prediction is to be made using the model(argument 1)
                                           already been trained specially for it.
- `class_means`             :- Vector of Hermitian matrices that represent the class
                                    means or the centroid of all the classes in the training set.
- `metric :: Metric`        :- The user needs to specify the metric space in which
                                all the distance computations is to be done.

Returns :

- `result :: Array{Int, 1}` :- The List of the predicted classes for the given
                                    sample set.


"""
function predict_mdm(sample, class_means, metric:: Metric)
    A = find_dist(sample, class_means, metric)
    result = [findmin(A[:,j])[2] for j = 1:dim(A,2)]
    return result
end



"""
Given a sample set, class_means, type of metric returns the
probability of each class for each sample case according to
the Minimum Distance to the Means scheme. All the probability sums to 1.
This function makes use of the softmax function from thr PostDefManifold to
calculate the probability values.

Arguments :

- `sample :: ℍVector`       :-The vector of Hermitian matrices or points in
                                           the positive definite manifold for which the probability
                                           prediction is to be made using the model(argument 1)
                                           already been trained specially for it.
- `class_means`             :- Vector of Hermitian matrices that represent the class
                                    means or the centroid of all the classes in the training set.
- `metric :: Metric`        :-The user needs to specify the metric space in which
                                all the distance computations is to be done.

Returns :

- `Prob :: Array{Float64,1}`:- The List of the predicted probabilities corresponding
                                    to each of the classes for the given sample set.


"""
function predict_prob(sample, class_means, metric :: Metric)
    A = find_dist(sample, class_means, metric)
    Prob = [softmax(-A[:,j]) for j = 1:dim(A,2)]
    return Prob
end



"""
Given the length, nCV( number of cross-validations) returns the vectors
containing indices of training and testing samples for each cross validation
iteration. This is a helper function to implement cross_val_mdm.
It uses shuffle! of the Random.jl package. This function holds the
prime basis of CrossValidation implementation.

Arguments :

- `k::Int`                 :- Last number of the sequence of natural numbers to be
                                    shuffled starting from 1
- `nCV:: Int`              :- Number of cross-validation for which the indices are
                                    to be generated.
- `shuffle`                :- Boolean to inform whether to do shuffling or not.
                                 Default is set to True.

Returns :

- `nTest`                  :- The size of each testing set.
- `nTrain`                 :- The size of each training set.
- `indTrain`               :- The list of all the vectors that contain the training
                                    indices for each iteration.
- `indTest`                :- The list of all the vectors that contain the testing
                                    indices for each iteration.


"""
function indCV(k::Int, nCV::Int, shuffle=false)
    if nCV == 1 @error "The number of cross-validation must be bigger than one" end
    nTest = k÷nCV
    nTrain = k-nTest
    #rng = MersenneTwister(1900)
    shuffle ? a=shuffle!( Vector(1:k)) : a=Vector(1:k)
    indTrain = [Vector{Int64}(undef, 0) for i=1:nCV]
    indTest  = [Vector{Int64}(undef, 0) for i=1:nCV]
    # vectors of indices for test and training sets
    j=1
    for i=1:nCV-1
        indTest[i]=a[j:j+nTest-1]
        for g=j+nTest:length(a) push!(indTrain[i], a[g]) end
        for l=i+1:nCV, g=j:j+nTest-1 push!(indTrain[l], a[g]) end
        j+=nTest
    end
    indTest[nCV]=a[j:end]
    return nTest, nTrain, indTrain, indTest
end


"""
Given a ℍVector(training set), y(labels), cv(number of cross-validations) returns
the the list containing the score for each cross-validation iteration. The score
may be average balanced accuracy or average regular accuracy depending on
the choice of the user. *It is better to call for balanced accuracy as a better
estimation of the model performance. Overall, if the number of ases in for each
class are same, balanced accuracy equals the regular accuracy.

This is the main function implementing cross validation for MDM classifier.
It uses indCV as its basic helper function. It also returns the final confusion
matrix.

Arguments :

- `𝐗 :: ℍVector`          :- The training set of type Vector of Hermitian matrices.
- `y`                      :- The list of labels corresponding to each trial
- `cv :: Int`              :- The number of cross-validation desired by the user
- `scoring :: String`      :- If Balanced Accuracy is requied or the Regular Accuracy.
                                    The default is set to Balanced Accuracy.
- `metric :: Metric`       :- The metric space in which to do the computations. To be
                                    specified by the user as one of the many metric spaces options provided in PostDefManifold.
                                    One may refer to [mdm.jl documentations] for exploring all the possible options.
                                    The default is set to Fisher.
- `cnfmat :: Bool`         :- Boolean to notify if the user wants the confusion matrix also.
                               Default is set to false.

Returns :

- `cross_val score`       :- The list of the balanced accuracy score for each
                                cross_val iteration.


"""
function cross_val_mdm(𝐗:: ℍVector, y, cv; scoring :: String = "bal", cnfmat :: Bool = false , metric :: Metric = Fisher)
    y1 = copy(y)
    classes = unique!(y1)
    nc = length(classes)
    𝐋 = [ℍ[] for i = 1: nc] # All data by classes
    𝐏 = [ℍ[] for i = 1: nc] # training data by classes
    𝐓 = [ℍ[] for i = 1: nc] # test data by classes
    acc = Array{Float64, 1}(undef, cv)
    bal_acc = Array{Float64, 1}(undef, cv)
    cnf_mat = [zeros(Float64 , (nc,  nc)) for i = 1:cv] # confusion matrix final
    for j = 1:dim(𝐗,1)  push!(𝐋[y[j]],𝐗[j]) end
    for k = 1:cv
        for i = 1:nc
            nTest, nTrain, indTrain, indTest = indCV(length(𝐋[i]), cv)
            𝐏[i]  = [𝐋[i][j] for j in(indTrain[k])  ]
            𝐓[i] =  [𝐋[i][j] for j in(indTest[k])  ]
        end
        if nc<=2 # check convrgence !!!!
            class_means = [mean_mdm(Fisher, 𝐏[Int(l)], ⏩=true ) for l= 1:length(classes)]
        else
            class_means = Vector(undef, nc)
            @threads for l in classes class_means[l]=mean_mdm(metric, 𝐏[Int(l)]) end
        end
        result = [Int[] for i = 1:nc]
        for i = 1: nc
            result[i] = predict_mdm(𝐓[i], class_means, metric)
            for s = 1: length(result[i]) cnf_mat[k][i, result[i][s]] = cnf_mat[k][i, result[i][s]] + 1. end
        end
        acc[k] = 𝚺( [ cnf_mat[k][i,i] for i = 1:nc ] )/ 𝚺( cnf_mat[k] )
        bal_acc[k] = (𝚺( [ cnf_mat[k][i,i] / 𝚺( cnf_mat[k][i,:] ) for i = 1:nc ])) / nc
    end
    cnf_mat_f = 𝚺(cnf_mat)
    #acc_f = 𝚺( [ cnf_mat_f[i,i] for i = 1:nc ] )/ 𝚺( cnf_mat_f )
    #bal_acc_f = (𝚺( [ cnf_mat_f[i,i] / 𝚺( cnf_mat_f[i,:] ) for i = 1:nc ])) / nc
    scores = (scoring == "bal") ? bal_acc : acc
    return scores
end
