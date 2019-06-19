# mdm.jl

This unit implemets the **MDM (Minimum Distance to Mean)** classifier for the
manifold of positive definite matrices. Similarly to what is done in
ScikitLearn in Python, a type is created (struct in Julia) of the desired specifications. 
It also implements **cross validation algorithm for MDM**( Minimum Distance to Mean)
classifier again similar to ScikitLearn in Python, one can have a better evaluation of the classifier by using cross validation.
Module incorporates supporting functions :   **find_dist**,    **predict_mdm**,    **predict_prob**,   **indCV**.

It implemens a structure **MDM** and includes the following functions :

| Function   | Description |
|:---------- |:----------- |
| [`mean_mdm`](@ref) | calculates the mean of all the classes in the training set and also    		 		notifies the user if the mean is not convergent in case of some metric 				spaces|
| [`find_dist`](@ref) | finds the distance of each sample case from the so found means of all the classes|
| [`predict_mdm`](@ref) | predicts the class for each sample case depending on its distance from the respective means|
| [`predict_prob`](@ref) | predicts the probability of each class for each sample case depending on its distance from the respective means|
| [`indCV`](@ref) | returns the vectors containing shuffled indices of training and testing samples for each cross validation iteration|
| [`cross_val_mdm`](@ref) | implements cross validation for MDM classifier|

For a detailed understanding of mdm, one should know the basics of Riemannian Geometry and its application in the classification of positive definite matrices. One may refer to the following papers for getting the feel of the process.

A. Barachant, S. Bonnet, M. Congedo, C. Jutten (2012)[🎓](@ref)

A. Barachant, S. Bonnet, M. Congedo, C. Jutten (2013)[🎓](@ref)

M. Congedo, A. Barachant, R. Bhatia R (2017a)[🎓](@ref)

M. Congedo, A. Barachant, E. Kharati Koopaei (2017b)[🎓](@ref)

Or one may directly look into the [Intro to Riemannian Geometry](https://marco-congedo.github.io/PosDefManifold.jl/latest/introToRiemannianGeometry/) section of the PostDefManifold documentation and quench all their doubts. 

 
## MDM structure

```@docs
MDM
```

## mean_mdm

```@docs
mean_mdm
```

## find_dist

```@docs
find_dist
```

## predict_mdm

```@docs
predict_mdm
```

## predict_prob

```@docs
predict_prob
```

## indCV

```@docs
indCV
```

## cross_val_mdm

```@docs
cross_val_mdm
```
