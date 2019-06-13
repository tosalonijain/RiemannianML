#function mean(metric::Metric, 𝐏::ℍVector;
            #w::Vector=[], ✓w=true, ⏩=false)
#@sk_import svm: LinearSVC
#@sk_import svm: SVC
"""
Structure for implementing LinearSVC, with all the arguments as attributes to
this structure. The *'clf'* attribute is the python object of LinearSVC that is
constructed by the constructor function of this structure as soon as a LinearSVM
instance is created by the user.

The available specifications for creating a model of LinearSVM are :

- penalty:: String
- loss :: String
- dual :: Bool
- tol :: Float64
- C :: Float64
- multi_class :: String
- fit_intercept :: Bool
- intercept_scaling :: Float64
- class_weight :: Dict
- verbose :: Int64
- random_state :: Int
- max_iter :: Int64

One should refer to the ScikitLearn documentation of LinearSVC whose link
is provided above for the better understanding of these parameters with respect
to Linear Support Vector Machine.

    ## Example
    using RiemannianML
    model = LinearSVM()


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
    function LinearSVM(; penalty="l2", loss="squared_hinge", dual=true, tol=0.0001, C=1.0,
        multi_class="ovr", fit_intercept=true, intercept_scaling=1, class_weight=nothing,
        verbose=0, random_state=nothing, max_iter=1000)
        clf = LinearSVC(penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling,
        class_weight, verbose, random_state, max_iter)
        new(clf, penalty, loss, dual, tol, C,multi_class, fit_intercept, intercept_scaling,
        class_weight, verbose, random_state, max_iter)
    end
end

"""
Structure for implementing SVC, with all the arguments as attributes to
this structure. The *'clf'* attribute is the python object of SVC that is
constructed by the constructor function of this structure as soon as a SVM
instance is created by the user.

The available specifications for creating a model of SVM are :

- penalty:: String
- loss :: String
- dual :: Bool
- tol :: Float64
- C :: Float64
- multi_class :: String
- fit_intercept :: Bool
- intercept_scaling :: Float64
- class_weight :: Dict
- verbose :: Int64
- random_state :: Int
- max_iter :: Int64

One should refer to the ScikitLearn documentation of SVC whose link
is provided above for the better understanding of these parameters with respect
to Support Vector Machine.

    ## Example
    using RiemannianML
    model = SVM()

"""
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
