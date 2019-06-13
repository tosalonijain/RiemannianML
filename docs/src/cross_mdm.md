# cross_mdm.jl

This unit contains **cross validation algorithm for MDM**( Minimum Distance to Mean)
classifier i.e. applied directly on Positive Definite Manifold. Similar to
ScikitLearn in Python, one can have a better evaluation of the classifier
by using cross validation.
Unit contains supporting function **indCV** and function **cross_val_mdm**.

| Function   | Description |
|:---------- |:----------- |
| [`indCV`](@ref) | returns the vectors containing shuffled indices of training and testing samples for each cross validation iteration|
| [`cross_val_mdm`](@ref) | implements cross validation for MDM classifier|


## indCV

```@docs
indCV
```

## cross_val_mdm

```@docs
cross_val_mdm
```
