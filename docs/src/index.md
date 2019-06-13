# RiemannianML Documentation

## Requirements

Julia version ≥ 1.0.3
Python ScikitLearn [(Refer for installation)](https://scikit-learn.org/stable/install.html) 

## Installation

Execute the following command in Julia's REPL:

    ]add RiemannianML

To obtain the latest development version execute instead

    ]add RiemannianML#master


## Overview


**Riemannian geometry** studies smooth manifolds, multi-dimensional curved spaces with peculiar geometries endowed with non-Euclidean metrics. In these spaces Riemannian geometry allows the definition of **angles**, **geodesics** (shortest path between two points), **distances** between points, **centers of mass** of several points, etc.

In several fields of research such as *computer vision* and *brain-computer interface*, treating data in the **P** manifold has allowed the introduction of machine learning approaches with remarkable characteristics, such as simplicity of use, excellent classification accuracy, as demonstrated by the [winning score](http://alexandre.barachant.org/challenges/) obtained in six international data classification competitions, and the ability to operate transfer learning [(Congedo et *al.*, 2017a, ](https://bit.ly/2HOk5qN)[Congedo et *al.*, 2017b)](https://bit.ly/2HKEcGk).

In this package we are concerned with making use of Riemannian Geometry in classification of data in the manifold of Positive Definite Matrices. This can be done in two ways, either in the **Positive Definite Manifold** or in the **Eucledian Space** of transformed data.   
- **Positive Definite Manifold :** Here we can use different distance metrics to compute the distances between the points represented by positive definite matrices. Using thsi we apply MDM(Minimum Distance to Mean) criteria for classifying test data in the positive defininite manifold.
- **Eucledian Space :** Here the data points from the manifold of positive definite matrices are transformed into corresponding points in the tangent space of the mean point of the set. Since the manifold of positive definite matrices forms a Riemannian manifold, so this brings us the opportunity to implement all the machine learning algorithms from scikit-learn. The reason being that the tangent space behaves like an Eucledian space, so it opens the way to all the general machine learning algorithms implemented in sckit-learn as they assume the metric apace to be Eucledian. The transformation is done using a specific relation()

This package makes extensive use of the functions from the package PostDefManifold. The reader might have a look into [this](https://marco-congedo.github.io/PosDefManifold.jl/latest/) for a better understanding of the functions used. 

For a formal introduction to the **P** manifold the reader is referred to the monography written by Bhatia (2007).

For an introduction to Riemannian geometry and an overview of mathematical tools implemented in this package, see [Intro to Riemannian Geometry](https://marco-congedo.github.io/PosDefManifold.jl/latest/introToRiemannianGeometry/) in the documentation of PostDefManifold.

For starting using this package, browse the code units listed here below and execute the many **code examples** you will find therein.

## Code units

**RiemannianML** includes six code units (.jl files):

| Unit   | Description |
|:----------|:----------|
| [MainModule (RiemannianML.jl)](@ref) | Main module, constants, types, aliases, tips & tricks |
| [knn.jl](@ref) | Unit implementing Kneighbhor Classification |
| [logisticRegression.jl](@ref) | Unit implementing LogisticRegression and LogisticRegressionCV |
| [SVM.jl](@ref) | Unit implementing LinearSVC and SVC |
| [mdm.jl](@ref) | Unit implementing MDM( Minimum Distance to Mean) classification |
| [cross_mdm.jl](@ref) | Unit implementing cross-validation for MDM classifier |
| [train_test.jl](@ref) | Unit implementing the tranformation function. Along with overwriting fit!, predict and cross_val_score from the ScikitLearn.jl package |
| [example.jl](@ref) | Unit containing examples for understanding and execution |

## Contents

```@contents
Pages = [       "index.md",
                "MainModule.md",
                "knn.md",
                "logisticRegression.md",
                "SVM.md",
                "mdm.md",
                "cross_mdm.md",
		"train_test.md",
		"example.md"]
Depth = 1
```

## Index

```@index
```
