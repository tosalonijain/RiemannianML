# example.jl

This unit demonstrates the use of RiemannianML using various examples. It can be used as a referrence guide while working in RiemannianML. Examples are provided for all the functions and structure objects. The corresponding unit example.jl contains the code corresponding to each and every example citation, one may directly copy these codes from there and run on their own to check its working. The results cited here might be different then the ones obtained by the user because random points generation is done to create stimulated data, so it may vary every time but the overall range varies roughly over the same stretch. 


**For these examples below, simulated data are used. These data are created by few simple lines of codes. Let us assume that the number of eclectrodes in our situation is 30.**

## Simulated data creation (not necessary)

So, we fix n = 30. This implies our data will consist of 30 x 30 Hermitian matrices. For the
real cases these simply be Real matrices.

	n=30


In the following examples, binary classification is demonstrated. Let the training examples for each of the two classes be 80. So, k1 = 80 and k2 = 80. The total training set size =k1 + k2
	
	k1=80
	k2=80
	k=k1+k2


To create stimulated data for EEG, we randomly pick two points A1, A2 in the Positive Definite manifold. Let A1 and A2 be the standard cases representing class 1 and class 2 respectively.

	A1=randP(n)
	A2=randP(n)


For creating the entire training set, that could behave as the training samples for classes 1 and 2. These should be similar somewhat to either of A1 or A2. To ensure this, first of all a set containing k1 + k2 random points in the post def manifold is generated.

	P=randP(n, k1+k2)


Now, we move each of these points closer to either A1 or A2 so they have some
resemblence to them. This will ensure that our data is worth classifying. We move
these points slowly and slowly closer to the standards chosen( A1, A2 ). This
closeness is monitored by the value gm here. gm = 0 means, no shifting is done,
they are just random set of points. gm = 1 means, all the points are shifted exactly
to A1 or A2. All the intermediate values of gm will take the points to intermediate
distance ratios on their geodesic joining A1/A2. So, half of the points are taken
closer to A1, and the other half to A2. This means half are closer to class 1
while the other half to class 2.

	gm=0.1
	P2=[geodesic(Fisher, P[i], A1, gm) for i=1:k1]
	Q2=[geodesic(Fisher, P[i], A2, gm) for i=k1+1:k1+k2 ]
	𝐗=ℍVector([P2; Q2])

The new shifted set of points are contained in P2 and Q2. We concatenate them into
one set 𝐗, which represent now the simulated training set.
Then the label is created for this training set.

	Y = [repeat([1], k1); repeat([2],k2)]

## Model declaration

Like in ScikitLearn, for applying a model on your data, first of all create an instance or object of the corresponding model class with all specifications. Here they are not classes but structures.

Model instance for **kneighborClf** that does KNeighborsClassifier classification.

	model1 = kneighborClf(n_neighbors=3)


Model instance for **LogisticReg** that applies LogisticRegression.

	model2 = LogisticReg()


Model instance for **LinearSVM** that applies LinearSVC.
	
	model3 = LinearSVM()


Model instance for **SVM** that applies SVC.

	model4 = SVM()


Model instance for **MDM( Minimum Distance to Mean )** that applies mdm classification.

	model5 = MDM(Fisher)


Model instance for **LogisticRegCV** that applies LogisticRegressionCV( CV - cross-validation ).

	model6 = LogisticRegCV()

## Fitting to the model

The so created simulated training sets are then fit to the models. This is done simply by calling the fit! function that takes 3 arguments, model, training set and labels. fit! is the first function i.e. to be called so that our model is ready to make predictions.

	fit!(model1, 𝐗,Y)
	fit!(model2, 𝐗,Y)
	fit!(model5, 𝐗,Y)
	fit!(model3, 𝐗,Y)
	fit!(model4, 𝐗,Y)
	fit!(model6, 𝐗,Y)


## Cross-Validation

In order to evaluate the model performance, cross-validation is done. The score of cross-validation is returned by the cross_val_score function which is then printed here.

	println(cross_val_score(model1,𝐗,Y, cv = cv_fold))
	println(cross_val_score(model2,𝐗,Y, cv = cv_fold))
	println(cross_val_score(model5,𝐗,Y, cv = cv_fold))


## Making prediction using the predict function

Now, in order to make the predictions, predict function is to be employed. Here also, a simulated sample set is feeded into the predict function to check if the prediction is made correctly or not. Again the same procedure as the one used for training set generation, is followed. After doing the desired amount of shifting, the points are stored in samp.

	T = randP(n,5)
	S = randP(n,3)
	tm = 0.198
	T1 = [geodesic(Fisher, T[i], A1, tm) for i=1:length(T)]
	S1 = [geodesic(Fisher, S[i], A2, tm) for i=1:length(S)]
	samp = ℍVector([T1; S1])


Calling the predict function on the testing sample samp with different fit models.

	predict(model2, samp)
	predict(model5, samp)


## Table representing performance score for all the models with simulated data.

A table is constructed that holds the cross-validation score for each of the models corresponding to different training sets. These different training sets are simulated training sets for an increasing value of gm i.e. extent of shifting. For this all the models are put in the list model. The following for loop fills values into this table. The table is printed.

	model = [model1, model2, model3, model4, model5, model6]
	println(model)
	cv_fold = 4
	table = Array{Float64, 2}(undef, 6,9)
	for i = 1:6
	    for j = 1:9
	        gm = 0.035 * j
	        P2=[geodesic(Fisher, P[i], A1, gm) for i=1:k1]
	        Q2=[geodesic(Fisher, P[i], A2, gm) for i=k1+1:k1+k2 ]
	        𝐗=ℍVector([P2; Q2])
	        #fit!(model[i], 𝐗,Y )
	        table[i,j] = (𝚺(cross_val_score(model[i],𝐗,Y, cv = cv_fold)))/ cv_fold
	    end
	end
	
	println(table)



	>> [0.493827 0.512346 0.556268 0.625594 0.719611 0.825024 0.90622 0.974834 0.993827; 
	0.650285 0.937559 0.987654 0.993827 0.993827 1.0 1.0 1.0 1.0; 
	1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0; 
	0.524454 0.54321 0.655508 0.755223 0.855413 0.981007 0.993827 0.993827 1.0; 
	0.59375 0.55 0.6 0.7875 0.8625 0.975 0.9875 0.99375 1.0; 
	1.0 1.0 1.0 1.0 1.0 0.993827 1.0 1.0 1.0]




Where as for a value of **gm = 0.015 * [1:9]**, we get the following set of values. Values are tabulated.


| **Model\gm** | 0.015| 0.030| 0.045| 0.060| 0.075| 0.900| 0.105| 0.120| 0.135|
|:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |
| **kneighbor** | 0.49375| 0.4875| 0.49375| 0.5125| 0.5125| 0.53125| 0.5375| 0.575| 0.625| 
| **LogisticReg(lasso)** | 0.54375| 0.7| 0.83125| 0.93125| 0.98125| 0.99375| 0.99375| 0.99375| 0.99375| 
| **LinearSVM** | 0.89375| 1.0| 1.0| 1.0| 1.0| 1.0| 1.0| 1.0| 1.0| 
| **SVM** | 0.53125| 0.53125| 0.55625| 0.56875| 0.59375| 0.61875| 0.6625| 0.68125| 0.71875| 
| **MDM** | 0.49375| 0.53125| 0.5375| 0.5875| 0.6625| 0.65625| 0.63125| 0.65625| 0.6875| 
| **LogisticRegCV(lasso)** | 1.0| 1.0| 1.0| 1.0| 1.0| 1.0| 1.0| 1.0| 1.0|


## Table representing performance score for all the models with real data.

**For real data taken from BNCI2014001 :**

The dataset contains the recording of 9 sujbects during two different sessions.
Each is stored in a different file and has a different subject number.
- 1 = subject 1, session 1
- 2 = subject 1, session 2
- 
- 
- 
- 17 = subject 9, session 1
- 18 = subject 9, session 2

We get the following results. Tabulation is done for few subjects only.

| **Model\gm** | **kneighbor**| **LogisticReg(lasso)**| **LinearSVM**|**SVM**| **MDM** | **LogisticRegCV(lasso)** |
|:---- |:---- |:---- |:---- |:---- |:---- |:---- |
|**subject_1** |0.770833| 0.930556| 0.861111| 0.798611| 0.861111| 0.909722|
| **subject_2** |0.819444| 0.902778| 0.895833| 0.861111| 0.888889| 0.909722| 
|**subject_3** |0.451389| 0.645833|0.645833| 0.590278| 0.611111| 0.618056|
|**subject_4**  | 0.583333| 0.618056| 0.666667| 0.513889| 0.541667| 0.638889|
|**subject_15**| 0.930556| 0.972222| 0.965278| 0.930556| 0.951389| 0.965278|
|**subject_16** | 0.930556| 0.972222| 0.965278| 0.958333| 0.965278| 0.958333|
|**subject_17** |0.784722| 0.840278| 0.8125| 0.722222| 0.791667| 0.826389|
| **subject_18** |0.868056| 0.902778| 0.875| 0.881944| 0.847222| 0.909722| 

