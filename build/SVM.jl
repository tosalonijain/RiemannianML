#function mean(metric::Metric, 𝐏::ℍVector;
            #w::Vector=[], ✓w=true, ⏩=false)
#@sk_import svm: LinearSVC
#@sk_import svm: SVC
# An object of scikit-learn KNeighborsClassifier is automatically created by the constructor
# as soon as an object of this struct is created by the user. This structure is created to
# that the same fit! function from ScikitLearn.jl could be used. To overwrite fit! with the
# same number of sttributes and similar type we needed to make a change in the argument types,
# but the sample labels could not differ and the training samples are not of any specified
# type in ScikitLearn.jl. This gives rise to ambiguities. So, the only option was to change the
# model type. Now our fit!(the one we have written) takes a julia structure, training samples
# and labels y. This difference solves the ambiguity between the two available fit! options.
"""
This module implements the Support Vector models for the data points in the manifold
    of positive definite matrices. The structures below are similar to classes of
    LinearSVC and SVC we have in ScikitLearn in Python. The user has to create an
    object/instance for the classifier class(here structure). The user can create a
    model instance of all the desired specifications.The specifications are added
    as atributes to these structures. This Module incorporates two models of
    Support Vector Machine type :
    1) LinearSVM
    2) SVM

    For further information on LinearSVM and SVM one may refer to the scikit-learn
    documentations of [LinearSVC]
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) and
    [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    respectively.
"""
mutable struct LinearSVM
    clf
    penalty:: String
    loss :: String
    dual :: Bool
    tol :: Float64
    C :: Float64
    multi_class :: String
    fit_intercept :: Bool
    intercept_scaling :: Float64
    class_weight
    verbose :: Int64
    random_state
    max_iter :: Int64
    dense_coeff_mat
    sparse_coeff_mat
    function LinearSVM(; penalty="l2", loss="squared_hinge", dual=true, tol=0.0001, C=1.0,
        multi_class="ovr", fit_intercept=true, intercept_scaling=1, class_weight=nothing,
        verbose=0, random_state=nothing, max_iter=1000)
        clf = LinearSVC(penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling,
        class_weight, verbose, random_state, max_iter)
        new(clf, penalty, loss, dual, tol, C,multi_class, fit_intercept, intercept_scaling,
        class_weight, verbose, random_state, max_iter)
    end
end

mutable struct SVM
    clf
    C:: Float64
    kernel :: String
    degree :: Int64
    gamma
    coef0::Float64
    shrinking :: Bool
    probability :: Bool
    tol:: Float64
    cache_size :: Float64
    class_weight
    verbose :: Bool
    max_iter :: Int
    decision_function_shape ::String
    random_state
    dense_coeff_mat
    sparse_coeff_mat
    function SVM(; C=1.0, kernel="rbf", degree=3, gamma="auto_deprecated", coef0=0.0, shrinking=true,
        probability=false, tol=0.001, cache_size=200, class_weight=nothing,
        verbose=false, max_iter=-1, decision_function_shape="ovr", random_state=nothing)
        clf = SVC( C,kernel, degree, gamma, coef0, shrinking,
            probability, tol, cache_size, class_weight,
            verbose, max_iter, decision_function_shape, random_state)
        new(clf, C,kernel, degree, gamma, coef0, shrinking,
            probability, tol, cache_size, class_weight,
            verbose, max_iter, decision_function_shape, random_state)
    end
end
