# mdm.jl

This unit implemets the **MDM (Minimum Distance to Mean)** classifier for the
manifold of positive definite matrices. Similarly to what is done in
ScikitLearn in Python, a type is created (struct in Julia) of the desired specifications. Module incorporates supporting functions :   **find_dist,   predict_mdm,   predict_prob**.

It implemens a structure **MDM** and includes the following functions :

| Function   | Description |
|:---------- |:----------- |
| [`mean_mdm`](@ref) | calculates the mean of all the classes in the training set and also    		 		notifies the user if the mean is not convergent in case of some metric 				spaces|
| [`find_dist`](@ref) | finds the distance of each sample case from the so found means of all the classes|
| [`predict_mdm`](@ref) | predicts the class for each sample case depending on its distance from the respective means|
| [`predict_prob`](@ref) | predicts the probability of each class for each sample case depending on its distance from the respective means|


For a detailed understanding of mdm, one should know the basics of Riemannian Geometry and its application in the classification of positive definite matrices. One may refer to the following papers for getting the feel of the process.

A. Barachant, S. Bonnet, M. Congedo, C. Jutten (2012) [Multi-class Brain Computer Interface Classification by Riemannian Geometry, IEEE Transactions on Biomedical Engineering, 59(4), 920-928](https://hal.archives-ouvertes.fr/hal-00681328/document).

A. Barachant, S. Bonnet, M. Congedo, C. Jutten (2013) [Classification of covariance matrices using a Riemannian-based kernel for BCI applications, Neurocomputing, 112, 172-178](https://hal.archives-ouvertes.fr/hal-00820475/document).

M. Congedo, A. Barachant, R. Bhatia R (2017a) [Riemannian Geometry for EEG-based Brain-Computer Interfaces; a Primer and a Review, Brain-Computer Interfaces, 4(3), 155-174](https://bit.ly/2HOk5qN).

M. Congedo, A. Barachant, E. Kharati Koopaei (2017b) [Fixed Point Algorithms for Estimating Power Means of Positive Definite Matrices, IEEE Transactions on Signal Processing, 65(9), 2211-2220](https://bit.ly/2HKEcGk).

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

