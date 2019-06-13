

@sk_import linear_model: LogisticRegression
@sk_import linear_model: LogisticRegressionCV
@sk_import neighbors: KNeighborsClassifier
@sk_import svm: LinearSVC
@sk_import svm: SVC

"""
This is an internal function which performs the transformation of
data from the manifold of positive definite matrices to the eucledian
space of the data set mean. We find the mean of the entire data set
with the help of the mean function from PostDefManifold. Once the
mean is speculated we do the transformation of all the data points
to their corresponding values in the tangent space of the data set mean.
The relation employed for transformation is the following logarithmic relation:
        (G½ * log(ℍ(G⁻½ * 𝐏[i] * G⁻½)) * G½)
where G is the data set mean and 𝐏[i] the set of points to be transformed.
For the better understanding of this transformation, one may refer to the
papers.

Arguments taken by this function are:

- **𝐏::ℍVector**              :- Vector of Hermitian matrices or simply a HermitianVector.
                                     The vector of points in the *Symmetric Positive Definite*
                                     manifold to be transformed into the the tangent space.
- **w::Vector(optional)**:- Vector containing weights corresponding to every point
                                     in 𝐏.
- **✓w = true(optional)**     :- Boolean to determine whether to calculate weighted mean
                                     or just take w = [].
- **⏩ = false (optional)**    :- Boolean to allow threading or not.

Return Value:

- **Vec :: Array{Float64, 2}** :-Vector of all the points in the training set.

"""
function _transform_ts(𝐏::ℍVector; w::Vector=[], ✓w=true, ⏩=false)
    G = mean(Fisher, 𝐏; w=w, ✓w=✓w, ⏩=⏩)
    len = dim(𝐏,1)
    Vec = Array{Float64, 2}(undef, dim(𝐏,1), Int(dim(𝐏,2)*(dim(𝐏,2)+1)/2) )
    G½, G⁻½ = pow(G, 0.5, -0.5)
    @threads for i = 1:len
        Vec[i,:] = vecP(ℍ(G½ * log(ℍ(G⁻½ * 𝐏[i] * G⁻½)) * G½))
    end
    return Vec
end

"""
This function is an overwriting of the default fit! function available in the
ScikitLearn.jl package. It checks the type of model, if it is mdm it runs
a block of code that fits the data which is in positive definite matrices
manifold directly by finding mean of datasets from all the classes. This
mean list is stored in the class_means attribute of the mdm instance.
If the model is not of type mdm, it takes a different and simpler path.
Then the function just calls the internal function _transform_ts to make
the transformation into the tangent space. This tangent space behaves
like an eucledian space. So, now the default fit! of ScikitLearn.jl
can directly be put to use. The function also prints the average
regular score in this case.

Arguments taken by this function are:

- **model::RiemannianML object**:- Classifier model instance eg. kneighbhorClf(),
                                   LogisticReg() or others. The model which is to
                                   be trained or to which the given data is to be fit.
                                   The instance should be created before calling fit! to
                                   train it.

- **𝐗::ℍVector**               :- Vector of Hermitian matrices or simply a HermitianVector.
                                  The vector of points in the training set consisting of
                                  *Symmetric Positive Definite* manifold matrices.
- **y :: Int[]**               :- Vector of intrger labels corresponding to each sample in the
                                  training set.
- **w::Vector(optional)** :- Vector containing weights corresponding to every point
                                  in 𝐗.
- **✓w = true(optional)**      :- Boolean to determine whether to calculate weighted mean
                                  or just take w = [].

A value is returned only in case the model is an mdm object. The return value in that case:

- **class_means**              :- List of means corresponding to all the classes for the
                                    given training set.


    ## Example
    model1 = kneighborClf(n_neighbors=3)
    model2 = LogisticReg()
    model3 = MDM(Fisher)
    n=10
    k1=25
    k2=25
    k=k1+k2
    A1=randP(n)
    A2=randP(n)
    P=randP(n, k1+k2)
    gm=0.5
    P2=[geodesic(Fisher, P[i], A1, gm) for i=1:k1]
    Q2=[geodesic(Fisher, P[i], A2, gm) for i=k1+1:k1+k2 ]
    𝐗=ℍVector([P2; Q2])
    Y = ones(Int64, 50)
    for j = 25:50
        Y[j] = 2
    end
    fit!(model1, 𝐗,Y)
    fit!(model2, 𝐗,Y)
    fit!(model3, 𝐗,Y)

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
        model.class_means = [mean_mdm(Fisher, 𝐋[Int(l)], w = W[Int(l)], ⏩=true ) for l in classes]
        return model.class_means
    else
        Z = _transform_ts(𝐗)
        fit!(model.clf, Z, y)
        println(score(model.clf, Z,y))
    end
end

"""
This function is an overwriting of the default predict function available in
the ScikitLearn.jl package. It works similar to the above fit! function. Depending
on the type of model i.e. mdm or rest, two different paths are opted. If its mdm,
then the predict_mdm function from the mdm.jl unit is called that helps in
making the prediction. If its not of type mdm, the function just calls the
internal function _transform_ts to make the transformation into the tangent
space. This tangent space behaves like an eucledian space. So, now the
default predict of ScikitLearn.jl can directly be put to use.

*Arguments taken by this function are:*

- **model::RiemannianML object**:- Classifier model instance eg. kneighbhorClf(),
                                       LogisticReg() or others. The model which is already
                                       been trained according to a training set can only be
                                       used as an argument here. The instance should be
                                       created and fit before calling predict on it.
- **samp::ℍVector**             :- The vector of Hermitian matrices or points in
                                       the positive definite manifold for which the
                                       prediction is to be made using the model(argument 1)
                                       already been trained specially for it.

*Return value:*

- **Predicted classes**        :- The List of the predicted classes for the given
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
        return (model.clf.predict(_transform_ts(samp)))
    end
end


"""
This function is an overwriting of the default cross_val_score function available in
the ScikitLearn.jl package. It works similar to the above mentioned fit!
and predict functions. Depending on the type of model i.e. mdm or rest,
two different paths are opted. If its mdm, then the cross_val_mdm function
from the cross_mdm.jl unit is called that helps in doing cross-validation
evaluation. If its not of type mdm, the function just calls the
internal function _transform_ts to make the transformation into the tangent
space. This tangent space behaves like an eucledian space. So, now the
default cross_val_score of ScikitLearn.jl can directly be put to use.

Arguments taken by this function are:

- **model::RiemannianML object**:- Classifier model instance eg. kneighbhorClf(),
                                       LogisticReg() or others i,e. the model whose
                                       evaluation using cross-validation is to be done.
- **𝐗::ℍVector**                :- Vector of Hermitian matrices or simply a HermitianVector.
                                       The vector of points in the training set consisting of
                                       *Symmetric Positive Definite* manifold matrices. The
                                       training set on the basis of which the evaluation
                                       of the given model is to be done.
- **y :: Int[]**                :- Vector of intrger labels corresponding to each sample in the
                                       training set.
- **cv :: Int(optional)**       :- The number of cross-validation desired by the user.
                                       The default value is set to 5.

*Return value*:

- **cross_val score**          :- The list containing the score for each cross_val
                                    iteration.


       ## Example
       model1 = kneighborClf(n_neighbors=3)
       model2 = LogisticReg()
       model3 = MDM(Fisher)
       n=10
       k1=25
       k2=25
       k=k1+k2
       A1=randP(n)
       A2=randP(n)
       P=randP(n, k1+k2)
       gm=0.5
       P2=[geodesic(Fisher, P[i], A1, gm) for i=1:k1]
       Q2=[geodesic(Fisher, P[i], A2, gm) for i=k1+1:k1+k2 ]
       𝐗=ℍVector([P2; Q2])
       Y = ones(Int64, 50)
       for j = 25:50
           Y[j] = 2
       end
       println(cross_val_score(model1,𝐗,Y))
       println(cross_val_score(model2,𝐗,Y))
       cross_val_score(model5, 𝐗,Y)


"""
function cross_val_score(model, 𝐗::ℍVector, y;cv = 5)
     if isa(model, MDM)
         cross_val_mdm(𝐗,y,cv)
     else
         return (cross_val_score(model.clf, _transform_ts(𝐗), y,cv = cv))
     end
end


#=
n=30
k1=20
k2=20
k=k1+k2
A1=randP(n)
A2=randP(n)
P=randP(n, k1+k2)
gm=0.1
P2=[geodesic(Fisher, P[i], A1, gm) for i=1:k1]
Q2=[geodesic(Fisher, P[i], A2, gm) for i=k1+1:k1+k2 ]
𝐗=ℍVector([P2; Q2])
Y = [repeat([1], k1); repeat([2],k2)]

model1 = kneighborClf(n_neighbors=3)
model2 = LogisticReg()
model3 = LinearSVM()
model4 = SVM()
model5 = MDM(Fisher)
model6 = LogisticRegCV()
cv_fold = 4
fit!(model1, 𝐗,Y)
fit!(model2, 𝐗,Y)
fit!(model3, 𝐗,Y)
fit!(model4, 𝐗,Y)
fit!(model5, 𝐗,Y)
fit!(model6, 𝐗,Y)
predict(model1, 𝐗)
predict(model5, 𝐗)
println(cross_val_score(model1,𝐗,Y, cv = cv_fold))
println(cross_val_score(model2,𝐗,Y, cv = cv_fold))
cross_val_score(model5, 𝐗,Y, cv = cv_fold)
T = randP(n,5)
S = randP(n,3)
tm = 0.8
T1 = [geodesic(Fisher, T[i], A1, tm) for i=1:length(T)]
S1 = [geodesic(Fisher, S[i], A2, tm) for i=1:length(S)]
samp = ℍVector([T1; S1])
predict(model1, samp)
=#
