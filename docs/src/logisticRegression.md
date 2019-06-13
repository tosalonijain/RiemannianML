# logisticRegression.jl

This unit implements the **LogisticRegression models** for the data points in the Riemannian manifold of *Symmetric Positive Definite (SPD)*. The structures below are similar to classes of
LogisticRegression and LogisticRegressionCV we have in ScikitLearn in Python.
The user has to create an object/instance for the classifier class(here structure).
The user can create a model instance of all the desired specifications.The specifications
are added as atributes to these structures. This Module incorporates two models of
LogisticRegression :
    - LogisticReg -   Simple logisticRegression in which the value of the penalty
                       coefficient alpha is set manually by the user. The user is
                       responsible for choosing a specific value of alpha.
    - LogisticRegCV - LogisticRegression with cross validation, to find the best
                       suit for alpha. The user can provide a range of alpha and the
                       algorithm finds the best suited value of alpha out of those.

For further information on LogisticReg and LogisticRegCV one may refer to the scikit-learn    documentations of [LogisticRegression]
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    and [LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
    respectively.

An object of scikit-learn **LogisticRegression** or **LogisticRegressionCV** is automatically
created by the constructor as soon as an instance of their structs is created by the user.
These structures are implemented so that the same fit! function from ScikitLearn.jl could be used.
To overwrite fit! with the same number of sttributes and similar type we needed to make a
change in the argument types, but the sample labels could not differ and the training
samples are not of any specified type in ScikitLearn.jl. This gives rise to ambiguities.
So, the only option was to change the model type. Now our fit!(the one we have written) takes
a julia structure, training samples and labels y. This difference solves the ambiguity
between the two available fit! options.

## LogisticReg structure 

```@docs
LogisticReg
```

## LogisticRegCV structure

```@docs
LogisticRegCV
```

