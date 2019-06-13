
using LinearAlgebra, Statistics, Base.Threads, PosDefManifold, ScikitLearn, Random
"""
This is a helper function to implement cross_val_mdm. It returns the vectors
containing indices of training and testing samples for each cross validation
iteration. The indices so received are shuffled well and not contiguous
strings or blocks of data samples. This function uses shuffle! to shuffle
all the indices and then divides the shuffled set into training and testing
sets. This function holds the prime basis of CrossValidation implementation.

Arguments taken by this function are:

- **k::Int**                 :- Last number of the sequence of natural numbers to be
                                    shuffled starting from 1
- **nCV:: Int**              :- Number of cross-validation for which the indices are
                                    to be generated.
- **shuffle**                :- Boolean to inform whether to do shuffling or not.
                                 Default is set to True.

Return values :

- **nTest**                  :- The size of each testing set.
- **nTrain**                 :- The size of each training set.
- **indTrain**               :- The list of all the vectors that contain the training
                                    indices for each iteration.
- **indTest**                :- The list of all the vectors that contain the testing
                                    indices for each iteration.


"""
function indCV(k::Int, nCV::Int, shuffle=true)
    if nCV == 1 @error "The number of cross-validation must be bigger than one" end
    nTest = k÷nCV
    nTrain = k-nTest
    shuffle ? a=shuffle!(Vector(1:k)) : a=Vector(1:k)
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
This function performs cv number of iterations in each of which it runs the mdm model
over a part of the shuffled data. This way helps us to have a better evaluation of
our classifier.

This is the main function implementing cross validation for MDM classifier.
It uses indCV as its basic helper function. It returns the vector of confusion matrices,
coreesponding to each cross valdation iteration and also the final confusion
matrix i.e. the sum of all the previous ones. The function also returns average balanced
accuracy or average regular accuracy depending on the choice of the user. The default
is set to balanced accuracy type. It takes the following arguments:

Arguments taken by this function are:

- **𝐗 :: ℍVector**          :- The training set of type Vector of Hermitian matrices.
- **y**                      :- The list of labels corresponding to each trial
- **cv :: Int**              :- The number of cross-validation desired by the user
- **scoring :: String**      :- If Balanced Accuracy is requied or the Regular Accuracy.
                                    The default is set to Balanced Accuracy.
- **metric :: Metric**       :- The metric space in which to do the computations. To be
                                    specified by the user as one of the many metric spaces options provided in PostDefManifold.
                                    One may refer to [mdm.jl documentations] for exploring all the possible options.
                                    The default is set to Fisher.

Return value:

- **cross_val score**       :- The list of the balanced accuracy score for each
                                cross_val iteration.


"""
function cross_val_mdm(𝐗:: ℍVector, y, cv; scoring :: String = "bal", metric :: Metric = Fisher)
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
            nTest, nTrain, indTrain, indTest = indCV(length(𝐋[i]), cv, true)
            𝐏[i]  = [𝐋[i][j] for j in(indTrain[k])  ]
            𝐓[i] =  [𝐋[i][j] for j in(indTest[k])  ]
        end
        if nc<=2 # check convrgence !!!!
            class_means = [mean_mdm(metric, 𝐏[Int(l)], ⏩=true) for l in classes]
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
        println(acc[k], " <------ acc and bal_acc ------> ", bal_acc[k])
    end
    cnf_mat_f = 𝚺(cnf_mat)
    acc_f = 𝚺( [ cnf_mat_f[i,i] for i = 1:nc ] )/ 𝚺( cnf_mat_f )
    bal_acc_f = (𝚺( [ cnf_mat_f[i,i] / 𝚺( cnf_mat_f[i,:] ) for i = 1:nc ])) / nc
    score = (scoring == "bal") ? bal_acc_f : acc_f
    println(cnf_mat_f, " <---- cnf_mat     ", scoring, "  ---->  ", score)
    return bal_acc
end
