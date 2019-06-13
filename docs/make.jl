push!(LOAD_PATH,"../src/")
using Documenter, RiemannianML

makedocs(
   sitename="RiemannianML",
   modules=[RiemannianML],
   pages =
   [
      "index.md",
      "MainModule.md",
      "knn.md",
      "logisticRegression.md",
      "SVM.md",
      "train_test.md",
      "mdm.md",
      "cross_mdm.md",
      "example.md"

   ]
)

deploydocs(
    repo = "https://github.com/tosalonijain/riemannianML.git"
)
