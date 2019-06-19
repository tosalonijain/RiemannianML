
"""
Given a ℍVector( Vector of Hermitian matrices), weights(optional), returns a vector containing
the mapping of these matrices in the tangent space of the data set mean. Vectorization is
done along with the mapping.
The relation employed for mapping is the following logarithmic relation:
        ``(G½ * log(ℍ(G⁻½ * 𝐏[i] * G⁻½)) * G½)``
where G is the data set mean and 𝐏[i] the set of points to be mapped.
For the better understanding of this mapping, one may refer to the [logMap function
in PostDefManifold](https://marco-congedo.github.io/PosDefManifold.jl/latest/riemannianGeometry/#PosDefManifold.logMap).

Arguments:

- `𝐏::ℍVector`              :- Vector of Hermitian matrices or simply a HermitianVector.
                                     The vector of points in the *Symmetric Positive Definite*
                                     manifold to be transformed into the the tangent space.

*The following parameters are needed for mean computation only:*

- `w::Vector(optional)`:- Vector containing weights corresponding to every point
                                     in 𝐏.
- `✓w = true(optional)`     :- Boolean to determine whether to calculate weighted mean
                                     or just take w = []. It also checks if the weights
                                     sum up to 1. If not does normalization.
- `⏩ = false (optional)`    :- Boolean to allow threading or not.

Returns:

- `Vec :: Array{Float64, 2}` :-Vector of all the points in the training set.

"""
function logMap(𝐏::ℍVector; metric :: Metric = Fisher, w::Vector=[], ✓w=true, ⏩=false)
    G = mean(metric, 𝐏; w=w, ✓w=✓w, ⏩=⏩)
    len = dim(𝐏,1)
    Vec = Array{Float64, 2}(undef, dim(𝐏,1), Int(dim(𝐏,2)*(dim(𝐏,2)+1)/2) )
    G½, G⁻½ = pow(G, 0.5, -0.5)
    @threads for i = 1:len
        Vec[i,:] = vecP(ℍ( log(ℍ(G⁻½ * 𝐏[i] * G⁻½)) ))
        #Vec[i,:] = vecP(ℍ(G½ * log(ℍ(G⁻½ * 𝐏[i] * G⁻½)) * G½))
    end
    return Vec
end

"""
Given a model, ℍVector(training set), y(labels), weights(optional), checks
the type of model and then fits the model to the given training set.
This function is an overwriting of the default fit! function available in the
ScikitLearn.jl package. The function also prints the average
regular score for all models except mdm.

Arguments :

- `model::RiemannianML object`:- Classifier model instance eg. kneighbhorClf(),
                                   LogisticReg() or others. The model which is to
                                   be trained or to which the given data is to be fit.
                                   The instance should be created before calling fit! to
                                   train it.

- `𝐗::ℍVector`               :- Vector of Hermitian matrices or simply a HermitianVector.
                                  The vector of points in the training set consisting of
                                  *Symmetric Positive Definite* manifold matrices.
- `y :: Int[]`               :- Vector of intrger labels corresponding to each sample in the
                                  training set.
- `w::Vector(optional)` :- Vector containing weights corresponding to every point
                                  in 𝐗. *Only for mdm models.*
- `✓w = true(optional)`      :- Boolean to determine whether to calculate weighted mean
                                  or just take w = [].It also checks if the weights
                                  sum up to 1. If not does normalization.

Returns:
(A value is returned only in case the model is an mdm object)

- `class_means`              :- List of means corresponding to all the classes for the
                                    given training set.


    ## Example
    model1 = kneighborClf(n_neighbors=3)
    model2 = LogisticReg()
    model3 = MDM(Fisher)
    𝐗 = *load data...*
    y = *load labels...*
    fit!(model1, 𝐗,y)
    fit!(model2, 𝐗,y)
    fit!(model3, 𝐗,y)

"""
function fit!(model, 𝐗 :: ℍVector, y; w::Vector=[], ✓w = true)
    if isa(model, MDM)
        y1 = copy(y)
        classes = unique!(y1)
        𝐋 = [ℍ[] for i = 1: length(classes)]
        W = [Float64[] for i = 1:length(classes)]
        for j = 1:dim(𝐗,1)
            push!(𝐋[Int(y[j])],𝐗[j])
            if !(isempty(w))    push!(W[Int(y[j])], w[j])  end #---non efficient-----------------------
        end
        model.class_means = [mean_mdm(Fisher, 𝐋[Int(l)], w = W[Int(l)], ⏩=true ) for l= 1:length(classes)]
        return model.class_means
    else
        Z = logMap(𝐗)
        fit!(model.clf, Z, y)
        println(score(model.clf, Z,y))
    end
end

"""
Given a trained model and the sample set, gives the predicted class for
the data points in the sample set. This function is an overwriting of the default predict function available in
the ScikitLearn.jl package.

Arguments :

- `model::RiemannianML object`:- Classifier model instance eg. kneighbhorClf(),
                                       LogisticReg() or others. The model which is already
                                       been trained according to a training set can only be
                                       used as an argument here. The instance should be
                                       created and fit before calling predict on it.
- `samp::ℍVector`             :- The vector of Hermitian matrices or points in
                                       the positive definite manifold for which the
                                       prediction is to be made using the model(argument 1)
                                       already been trained specially for it.

Returns :

- `Predicted classes`        :- The List of the predicted classes for the given
                                    sample set.


    ## Example
    # following the above code for fit!
    predict(model1, 𝐗)
    predict(model2, 𝐗)
    predict(model3, 𝐗)


"""
function predict(model,samp)
    if isa(model,MDM)
        predict_mdm(samp, model.class_means, model.metric)
    else
        return (model.clf.predict(logMap(samp)))
    end
end


"""
Given a model, ℍVector(training set), y(labels), cv(number of cross-validations) returns
the the list containing the score for each cross-validation iteration.
This function is an overwriting of the default cross_val_score function available in
the ScikitLearn.jl package.

Arguments :

- `model::RiemannianML object`:- Classifier model instance eg. kneighbhorClf(),
                                       LogisticReg() or others i,e. the model whose
                                       evaluation using cross-validation is to be done.
- `𝐗::ℍVector`                :- Vector of Hermitian matrices or simply a HermitianVector.
                                       The vector of points in the training set consisting of
                                       *Symmetric Positive Definite* manifold matrices. The
                                       training set on the basis of which the evaluation
                                       of the given model is to be done.
- `y :: Int[]`                :- Vector of intrger labels corresponding to each sample in the
                                       training set.
- `cv :: Int(optional)`       :- The number of cross-validation desired by the user.
                                       The default value is set to 5.

*These arguments are only applicable if the model is mdm type* :

- `scoring :: String`      :- If Balanced Accuracy is requied or the Regular Accuracy.
                                    The default is set to Balanced Accuracy.
- `cnfmat :: Bool`         :- Boolean to notify if the user wants the confusion matrix also.
                               Default is set to false.

Returns :

- `cross_val score`          :- The list containing the score for each cross_val
                                    iteration.


       ## Example
       model1 = kneighborClf(n_neighbors=3)
       model2 = LogisticReg()
       model3 = MDM(Fisher)
       𝐗 = *load data...*
       y = *load labels...*
       println(cross_val_score(model1,𝐗,y))
       println(cross_val_score(model2,𝐗,y))
       cross_val_score(model5, 𝐗,y)


"""
function cross_val_score(model, 𝐗::ℍVector, y;cv = 5, scoring :: String = "bal", cnfmat :: Bool = false)
     if isa(model, MDM)
         cross_val_mdm(𝐗,y,cv)
     else
         return (cross_val_score(model.clf, logMap(𝐗), y,cv = cv))
     end
end
