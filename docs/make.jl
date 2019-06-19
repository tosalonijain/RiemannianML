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
      "example.md",
      "tutorial.md"

   ]
)

deploydocs(
    repo = "https://github.com/tosalonijain/RiemannianML.git"
)
