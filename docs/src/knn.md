# knn.jl

This module implements the **KNeighborsClassifier** model for the data points in the Riemannian manifold of *Symmetric Positive Definite (SPD)*. The structure below is similar to a class of
KNeighborsClassifier we have in ScikitLearn in Python. The user has to create an
object/instance for the classifier class(here structure). The user can create an
instance of all the desired specifications.The specifications are added as
atributes to this structure. For further information on kneighborClf,
one may refer to the scikit-learn documentation of [KNeighborsClassifier]
(https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

An object of scikit-learn **KNeighborsClassifier** is automatically created by the constructor
as soon as an object of this struct is created by the user. This structure is created to
that the same fit! function from ScikitLearn.jl could be used. To overwrite fit! with the
same number of sttributes and similar type we needed to make a change in the argument types,
but the sample labels could not differ and the training samples are not of any specified
type in ScikitLearn.jl. This gives rise to ambiguities. So, the only option was to change the
model type. Now our fit!(the one we have written) takes a julia structure, training samples
and labels y. This difference solves the ambiguity between the two available fit! options.

## kneighborClf  structure

```@docs
kneighborClf
```

