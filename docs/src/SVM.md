# SVM.jl

This module implements the Support Vector models for the data points in the manifold
of positive definite matrices. The structures below are similar to classes of
LinearSVC and SVC we have in ScikitLearn in Python. The user has to create an
object/instance for the classifier class(here structure). The user can create a
model instance of all the desired specifications.The specifications are added
as atributes to these structures. This Module incorporates two models of
Support Vector Machine type :
    - LinearSVM
    - SVM

For further information on LinearSVM and SVM one may refer to the scikit-learn documentations of [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) and [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) respectively.

An object of scikit-learn **LinearSVC** or **SVC** is automatically
created by the constructor as soon as an instance of their structs is created by the user.
These structures are implemented so that the same fit! function from ScikitLearn.jl could be used.
To overwrite fit! with the same number of sttributes and similar type we needed to make a
change in the argument types, but the sample labels could not differ and the training
samples are not of any specified type in ScikitLearn.jl. This gives rise to ambiguities.
So, the only option was to change the model type. Now our fit!(the one we have written) takes
a julia structure, training samples and labels y. This difference solves the ambiguity
between the two available fit! options.

## LinearSVM  structure 

```@docs
LinearSVM
```

## SVM  structure

```@docs
SVM
```

