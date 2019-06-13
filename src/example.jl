
@sk_import linear_model: LogisticRegression
@sk_import linear_model: LogisticRegressionCV
@sk_import neighbors: KNeighborsClassifier
@sk_import svm: LinearSVC
@sk_import svm: SVC

#=
n=30

# In the following examples, binary classification is demonstrated. Let the training examples
# for each of the two classes be 80. So, k1 = 80 and k2 = 80. The total training set size =
# k1 + k2
k1=80
k2=80
k=k1+k2

# To create stimulated data for EEG, we randomly pick two points A1, A2 in the Positive
# Definite manifold. Let A1 and A2 be the standard cases
# representing class 1 and class 2 respectively.
A1=randP(n)
A2=randP(n)

# For creating the entire training set, that could behave as the training samples
# for classes 1 and 2. These should be similar somewhat to either of A1 or A2. To
# ensure this, first of all a set containing k1 + k2 random points in the post
# def manifold is generated.
P=randP(n, k1+k2)
#=
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
=#
Y = [repeat([1], k1); repeat([2],k2)]

# Like in ScikitLearn, for applying a model on your data, first of all create an
# instance or object of the corresponding model class with all specifications.
# Here they are not classes but structures.

# Model instance for kneighborClf that does KNeighborsClassifier classification.
model1 = kneighborClf(n_neighbors=3)

# Model instance for LogisticReg that applies LogisticRegression.
model2 = LogisticReg()

# Model instance for LinearSVM that applies LinearSVC.
model3 = LinearSVM()

# Model instance for SVM that applies SVC.
model4 = SVM()

# Model instance for MDM( Minimum Distance to Mean ) that applies mdm classification.
model5 = MDM(Fisher)

# Model instance for LogisticRegCV that applies LogisticRegressionCV
# (CV - cross-validation).
model6 = LogisticRegCV()

# The so created simulated training sets are then fit to the models. This is done
# simply by calling the fit! function that takes 3 arguments, model, training set
# and labels. fit! is the first function i.e. to be called so that our model is
# ready to make predictions.
fit!(model1, 𝐗,Y)
fit!(model2, 𝐗,Y)
fit!(model5, 𝐗,Y)
fit!(model3, 𝐗,Y)
fit!(model4, 𝐗,Y)
fit!(model6, 𝐗,Y)

# In order to evaluate the model performance, cross-validation is done.
# The score of cross-validation is returned by the cross_val_score function
# which is printed here.
println(cross_val_score(model1,𝐗,Y, cv = cv_fold))
println(cross_val_score(model2,𝐗,Y, cv = cv_fold))
println(cross_val_score(model5,𝐗,Y, cv = cv_fold))

# Now, in order to make the predictions, predict function is to be employed.
# Here also, a simulated sample set is feeded into the predict function to
# check if the prediction is made correctly or not. Again the same procedure
# as the one used for training set generation, is followed. After doing the
# desired amount of shifting, the points are stored in samp.
T = randP(n,5)
S = randP(n,3)
tm = 0.198
T1 = [geodesic(Fisher, T[i], A1, tm) for i=1:length(T)]
S1 = [geodesic(Fisher, S[i], A2, tm) for i=1:length(S)]
samp = ℍVector([T1; S1])

# Calling the predict function on the testing sample samp with different fit models.
predict(model2, samp)
predict(model5, samp)



# A table is constructed that holds the cross-validation score for each of
# the models corresponding to different training sets. These different training sets are
# simulated training sets for an increasing value of gm i.e. extent of
# shifting. For this all the models are put in the list model. The following
# for loop fills values into this table. The table is printed.
model = [model1, model2, model3, model4, model5, model6]
println(model)
cv_fold = 4
table = Array{Float64, 2}(undef, 6,9)
for i = 1:6
    for j = 1:9
        gm = 0.015 * j
        P2=[geodesic(Fisher, P[i], A1, gm) for i=1:k1]
        Q2=[geodesic(Fisher, P[i], A2, gm) for i=k1+1:k1+k2 ]
        𝐗=ℍVector([P2; Q2])
        #fit!(model[i], 𝐗,Y )
        table[i,j] = (𝚺(cross_val_score(model[i],𝐗,Y, cv = cv_fold)))/ cv_fold
    end
end

println(table)
=#
