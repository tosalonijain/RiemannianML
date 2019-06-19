# SVM.jl

This unit implements the Support Vector models for the data points in the manifold
of positive definite matrices. The structures below are similar to classes of
LinearSVC and SVC we have in ScikitLearn in Python. This Module incorporates two models of
Support Vector Machine type :
    - LinearSVM
    - SVM

For further information on LinearSVM and SVM one may refer to the scikit-learn documentations of [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) and [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) respectively.

An object of scikit-learn **LinearSVC** or **SVC** is automatically
created by the constructor as soon as an instance of their structs is created by the user.


## LinearSVM  structure 

```@docs
LinearSVM
```

## SVM  structure

```@docs
SVM
```

