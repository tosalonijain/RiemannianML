# train_test.jl

This unit implements the tranformation function. Along with that this unit
overwrites the fit!, predict and cross_val_score from the ScikitLearn.jl package.
This enables us to use the same functions even for data in the manifold of positive
definite matrices.

It imports the required machine learning models from scikit-learn python using PyCall. This unit includes the following functions :

| Function   | Description |
|:---------- |:----------- |
| [_transform_ts](@ref) | internal function that projects the points in the SPD manifold into the tangent space|
| [`fit!`](@ref) | fits the model for the given training set|
| [`predict`](@ref) | makes prediction for the points in the test set|
| [`cross_val_score`](@ref) | evaluates the cross-validation score of the estimator or model|


## _transform_ts

```@docs
_transform_ts
```

## fit!

```@docs
fit!
```

## predict

```@docs
predict
```

## cross_val_score

```@docs
cross_val_score
```


 
