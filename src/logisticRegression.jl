# Classes imported from ScikitLearn python version.
#@sk_import linear_model: LogisticRegression
#@sk_import linear_model: LogisticRegressionCV
@sk_import linear_model: LogisticRegression
@sk_import linear_model: LogisticRegressionCV
"""
Structure for implementing LogisticRegression, with all the arguments as attributes to
this structure. The *'clf'* attribute is the python object of LogisticRegression that is
constructed by the constructor function of this structure as soon as a LogisticReg
instance is created by the user.

Parameters :

- penalty:: String
- dual :: Bool
- tol :: Float64
- C :: Float64
- fit_intercept :: Bool
- intercept_scaling :: Float64
- class_weight :: Dict
- random_state :: Int
- solver :: String
- max_iter :: Int64
- multi_class :: String
- verbose :: Int64
- warm_start :: Bool
- n_jobs :: Int
- l1_ratio :: Float64

One may refer to the ScikitLearn documentation of LogisticRegression for the better understanding of these parameters with respect
to LogisticRegression.

    ## Example
    using RiemannianML
    model = LogisticReg()


"""
mutable struct LogisticReg
    clf
    penalty:: String
    dual :: Bool
    tol :: Float64
    C :: Float64
    fit_intercept :: Bool
    intercept_scaling :: Float64
    class_weight
    random_state
    solver :: String
    max_iter :: Int64
    multi_class :: String
    verbose :: Int64
    warm_start :: Bool
    n_jobs
    l1_ratio
    function LogisticReg(;penalty="l2", dual=false, tol=0.0001, C=1.0, fit_intercept=true, intercept_scaling=1,
    class_weight=nothing, random_state=nothing, solver="warn", max_iter=100, multi_class="warn",
    verbose=0, warm_start=false, n_jobs=nothing, l1_ratio=nothing)
        clf = LogisticRegression(penalty, dual, tol, C, fit_intercept, intercept_scaling,
        class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)
        new(clf, penalty, dual, tol, C, fit_intercept, intercept_scaling,
        class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)
    end
end

"""
Structure for implementing LogisticRegressionCV, with all the arguments as attributes to
this structure. The *'clf'* attribute is the python object of LogisticRegressionCV that is
constructed by the constructor function of this structure as soon as a LogisticRegCV
instance is created by the user.

Parameters :

- Cs :: Int[] or Float64[]
- fit_intercept :: Bool
- cv :: Int
- dual :: Bool
- penalty :: String
- scoring :: String
- solver :: String
- tol :: Float64
- max_iter :: Int
- class_weight :: Dict
- n_jobs :: Int
- verbose :: Int
- refit :: Bool
- intercept_scaling :: Float64
- multi_class :: String
- random_state :: Int
- l1_ratios :: Float64

One should refer to the ScikitLearn documentation of LogisticRegressionCV for the better understanding of these parameters with respect
to LogisticRegressionCV.

    ## Example
    using RiemannianML
    model = LogisticRegCV()



"""
mutable struct LogisticRegCV
    clf
    Cs
    fit_intercept :: Bool
    cv
    dual :: Bool
    penalty :: String
    scoring
    solver :: String
    tol :: Float64
    max_iter :: Int
    class_weight
    n_jobs
    verbose :: Int
    refit :: Bool
    intercept_scaling :: Float64
    multi_class :: String
    random_state
    l1_ratios
    function LogisticRegCV(;Cs=10, fit_intercept = true, cv = "warn", dual = false,
    penalty = "l2", scoring = nothing, solver = "lbfgs", tol = 0.0001, max_iter = 100,
    class_weight = nothing, n_jobs = nothing, verbose=0, refit = true,
    intercept_scaling = 1.0, multi_class = "warn", random_state = nothing, l1_ratios = nothing)
        clf = LogisticRegressionCV(Cs, fit_intercept, cv, dual, penalty, scoring,
        solver, tol, max_iter, class_weight, n_jobs,
        verbose, refit, intercept_scaling, multi_class, random_state, l1_ratios)
        new(clf, Cs, fit_intercept, cv, dual, penalty, scoring, solver, tol, max_iter, class_weight, n_jobs,
        verbose, refit, intercept_scaling, multi_class, random_state, l1_ratios)
    end
end
