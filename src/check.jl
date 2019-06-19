#=
@sk_import linear_model: LogisticRegression
@sk_import linear_model: LogisticRegressionCV
@sk_import neighbors: KNeighborsClassifier
@sk_import svm: LinearSVC
@sk_import svm: SVC
@sk_import metrics: confusion_matrix
@sk_import model_selection: KFold


using NPZ

path = "/home/saloni/RiemannianML/BNCI2014001/" # where files are stored
filename = "subject_15.npz" # for subject number i
data = npzread(path*filename)
X = data["data"] # retrive the epochs
y = data["labels"] # retrive the corresponding labels
sam_size = size(X,1)
𝐗 = ℍVector(undef, sam_size)
@threads for i = 1:sam_size
    𝐗[i] = gram(X[i,:,:])
end




model5 = MDM(Fisher)
cv_fold = 6

#(fit!(model5,𝐗 ,y))
#find_distt(𝐗, model5.class_means,Fisher)
#predict_mdmm(𝐗, model5.class_means,Fisher)


model1 = kneighborClf(n_neighbors=3)

# Model instance for LogisticReg that applies LogisticRegression.
model2 = LogisticReg()

# Model instance for LinearSVM that applies LinearSVC.
model3 = LinearSVM()

# Model instance for SVM that applies SVC.
model4 = SVM()

# Model instance for LogisticRegCV that applies LogisticRegressionCV
# (CV - cross-validation).
model6 = LogisticRegCV(solver = "liblinear", cv = 5)

# Model instance for MDM( Minimum Distance to Mean ) that applies mdm classification.

model = [model1, model2, model3, model4, model5, model6]
table = Array{Float64, 2}(undef, 6, cv_fold)
for i = 1:6
    table[i,:] = (cross_val_score(model[i],𝐗,y, cv = cv_fold))
end

println([𝚺(table[i,:])/cv_fold for i = 1:6])

=#
