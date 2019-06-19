# Tutorial


This is a tutorial guide for RiemannianML module. **If you hold previous experience in working with ScikitLearn in Python, you may skip this tutorial.** For rest, you may have a look. 

## Classification in RiemannianML

We begin with deciding the model for our data. Currently this package serves 5 models/classifiers:

- K Nearest Neighbor
- Logistic Regression 
- Logistic Regression with Cross-Validation
- Linear SVC( Support Vector Classification)
- SVC( Support Vector Classification)
- MDM( Minimum Distance to Mean) in Positive Definite Manifold

### Creating Models

The user has to create an object/instance for the classifier class(here structure). The user can create an instance by specifying all the available arguments. Since, this module sets a uniform interface between ScikitLearn python,all the arguments are accepted in ScikitLearn python for all the classifiers except MDM are are accepted here also.

	# Model instance for kneighborClf that does KNeighborsClassifier classification.
	model1 = kneighborClf(n_neighbors=3)

	# Model instance for LogisticReg that applies LogisticRegression.
	model2 = LogisticReg(solver = "liblinear", max_iter = 4000)

	# Model instance for LinearSVM that applies LinearSVC.
	model3 = LinearSVM()

	# Model instance for SVM that applies SVC.
	model4 = SVM()

	# Model instance for MDM( Minimum Distance to Mean ) that applies mdm classification.
	model5 = MDM(Fisher)

	# Model instance for LogisticRegCV that applies LogisticRegressionCV
	# (CV - cross-validation).
	model6 = LogisticRegCV(solver = "liblinear", max_iter = 4000, cv = 5)

### Loading the data

After a classifier instance is created we need to fit the classifier model with the data. For this we first need to load the data in our code. Say my data is stored in npz format(.npz) in my local
computer at the following address.

	using NPZ

	path = "/home/saloni/RiemannianML/src/" # where files are stored
	filename = "subject_1.npz" # for subject number i
	data = npzread(path*filename)
	X = data["data"] # retrive the epochs
	y = data["labels"] # retrive the corresponding labels
	
### Making the data ready for use	

This corresponds to the training part. The training data should be in the form of :

- `𝐗` :-  Vector of Hermitian Matrices. These Hermitian Matrices are the covariance matrices formed   		    out of the raw data. The raw data of signals first need to be converted into their 		   corresponding covariance matrices. This can be very easily done using the [gram](https://marco-congedo.github.io/PosDefManifold.jl/latest/signalProcessing/#PosDefManifold.gram) function of 		    **PosDefManifold**. This is what is done in the below code.     
- `y` :-  Labels corresponding to each training sample. Labels should be integers from 1 to n, where  		   n is the number of classes.


	train_size = size(X,1)
	𝐗 = ℍVector(undef, sam_size)
	@threads for i = 1:train_size
    	 	𝐗[i] = gram(X[i,:,:])
	end

### Fitting your data to the model

Once you are ready with your data, you are all set to train your model. This is done easily using the `fit!` function.

	fit!(model1,𝐗 ,y)
	fit!(model2,𝐗 ,y)

### Making predictions

We say now that the model has been trained, it can be used to make predictions for the test cases. This is again done using a simple function, `predict`. Say we have the test cases stored the same way as the training data. We load it the same way and apply the same gram function to convert them into covariance matrices. Now we again have a vector of Hermitian Matrices as our **testing set 𝐓.**
We predict the class for each sample in the testing set. *We can only give a trained model as input*.

	predict(model1,𝐓)
	predict(model2,𝐓)

### Evaluating the models

Last but not the least, we can check the performance of our model by cross-validation. For models imported from ScikitLearn.jl we can simply calculate the score also using the `score!` function.

	score!(model1.clf, 𝐗 ,y)

But its always better to evaluate a model's performance by cross-validation and not simply calculating the score. So, we make use of `cross_val_score`. We directly feed the model into it without even training it before. This function returns The list containing the score for each cross-validation iteration. The number of iteration can be managed using the `cv` argument.


	println(cross_val_score(model1,𝐗,Y, cv = 5))

	# Prints the average of all the cross-validation scores
	println(𝚺(cross_val_score(model2,𝐗,Y, cv = 5))/ cv)

	# Special arguments available only for mdm model types
	println(cross_val_score(model5,𝐗,Y, cv = 5, cnfmat = true))


## Example( continuing from above)

	model = [model1, model2, model3, model4, model5, model6]
	cv_fold = 4
	table = Array{Float64, 2}(undef, 6, cv_fold)
	for i = 1:6
    		table[i,:] = (cross_val_score(model[i],𝐗,y, cv = cv_fold))
	end
	println([𝚺(table[i,:])/cv_fold for i = 1:6])


	>>> [0.493056, 0.673611, 0.701389, 0.5, 0.708333, 0.715278]


