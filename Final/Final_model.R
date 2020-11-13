library(gbm)
# Read_in test and train data
rm(list = ls())
train_df = read.csv("Data/Data2020.csv")
test_df = read.csv("Data/Data2020testX.csv")


set.seed(1)
gbm.model = gbm(Y ~ ., data =train_df, distribution = "gaussian", 
                   n.trees = 10000, interaction.depth = 8, shrinkage = 0.01)
n.trees.best = gbm.perf(gbm.model, plot.it = F) * 2 # Number of trees
prediction = predict(gbm.model, test_df, n.trees.best)
write.table(prediction, "Final/prediction.csv", sep = ",", row.names = F, col.names =F)
