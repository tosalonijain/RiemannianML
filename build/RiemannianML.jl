#module RiemannianML

#   Main Module of the RiemannianML Package for julia language
#   v 0.0.1 - last update 16th of May 2019
#
#   MIT License
#   Copyright (c) 2019, Saloni Jain, CNRS, Grenobe, France:
#
#   This pachwge works in conjunction with the PosDefManifold package
#   https://github.com/Marco-Congedo/PosDefManifold.jl
#   and with the ScikitLearn packqge
#   XX


# __precompile__()

module RiemannianML

using LinearAlgebra, Statistics, Base.Threads, PosDefManifold, ScikitLearn,
        ScikitLearn.CrossValidation


# Special instructions and variables
BLAS.set_num_threads(Sys.CPU_THREADS-Threads.nthreads())


# constants

# aliases

# types

# import
@sk_import linear_model: LogisticRegression
@sk_import linear_model: LogisticRegressionCV
@sk_import neighbors: KNeighborsClassifier
@sk_import svm: LinearSVC
@sk_import svm: SVC
import
    ScikitLearn.fit!,
    ScikitLearn.CrossValidation.cross_val_score,
    ScikitLearn.predict

export
    # From this module

    # From knn.jl
    kneighbhorClf,

    # From logisticRegression.jl
    LogisticReg,
    LogisticRegCV,

    # From SVM.jl
    LinearSVM,
    SVM,

    #From mdm.jl
    mean_mdm,
    MDM,
    predict_mdm,
    find_dist,

    #From cross_mdm.jl
    indCV,
    cross_val_mdm,

    # From Train_test.jl
    fit!,
    testing,
    training,
    cross_val_score,
    predict


include("knn.jl")
include("logisticRegression.jl")
include("SVM.jl")
include("mdm.jl")
include("cross_mdm.jl")
include("train_test.jl")

println("\n⭐ "," Welcome to the RiemannianML package", " ⭐\n")
@info(" ")
println(" Your Machine ",gethostname()," (",Sys.MACHINE, ")")
println(" runs on kernel ",Sys.KERNEL," with word size ",Sys.WORD_SIZE,".")
println(" CPU  Threads: ",Sys.CPU_THREADS)
# Sys.BINDIR # julia bin directory
println(" Base.Threads: ", "$(Threads.nthreads())")
println(" BLAS Threads: ", "$(Sys.CPU_THREADS-Threads.nthreads())", "\n")



end # module
