rm(list = ls())
library(MASS)
library(glmnet)
library(pls)
library(mgcv)
library(randomForest)
library(gbm)
library(nnet)
source("Helper Functions.R")


df = read.csv("Data/Data2020.csv")
X = df[,-1]
Y = df[,1]

data.matrix.raw = model.matrix(Y ~ ., data = df)
data.matrix = data.matrix.raw[,-1]

# set.seed(100)
V = 10    # 10-fold Cross-Validation
R = 5     # 5 Replicate
n = nrow(X)
folds = get.folds(n,V) 
names = c('lm','StepWise','Ridge',"LASSO-min","LASSO-1SE","PLS","RF","Boosting","NN")
MSPE = matrix(NA,ncol = length(names), nrow = V*R)
colnames(MSPE) <- names

current_row = 1
for (i in 1:R) {
  for(v in 1:V){
    train_df = df[folds!=v,]
    valid_df = df[folds==v,]
    train_matrix = model.matrix(Y ~ . ,train_df)
    valid_matrix = model.matrix(Y ~ . ,valid_df)
    
    # Linear Regression
    cv.lm <- lm(Y~.,data = train_df)
    cv.lm.pred <- predict(cv.lm,newdata = valid_df)
    MSPE[current_row,1] =  mean((valid_df[,"Y"] - cv.lm.pred)^2)
    
    # Stepwise Selection
    initial <- lm(formula = Y~ 1,data = train_df)
    final <- lm(formula = Y ~ .,data = train_df)
    cv.step_model = step(object=initial, scope=list(upper=final))
    cv.step_pred = predict(cv.step_model, newdata = valid_df)
    MSPE[current_row,2] =  mean(as.vector((valid_df[,"Y"] - cv.step_pred)^2))
    
    # Ridge Regression
    cv.ridge<- lm.ridge(Y ~., lambda = seq(0, 100, .05), data=train_df)
    cv.ridge.coef = coef(cv.ridge)[which.min(cv.ridge$GCV),]
    cv.ridge.pred = valid_matrix %*% cv.ridge.coef
    MSPE[current_row,3] =  mean((valid_df[,"Y"] - cv.ridge.pred)^2)
    
    # LASSO
    cv.lasso = cv.glmnet(x=train_matrix[,-1], y = train_df[,'Y'])
    cv.lasso.min.pred = predict(cv.lasso, newx = valid_matrix[,-1],
                                s = cv.lasso$lambda.min, type = "response")
    cv.lasso.1se.pred = predict(cv.lasso, newx = valid_matrix[,-1],
                                s = cv.lasso$lambda.1se, type = "response")
    
    # LASSO-min
    MSPE[current_row,4] =  mean((valid_df[,"Y"] - cv.lasso.min.pred)^2)
    # LASSO-1se
    MSPE[current_row,5] =  mean((valid_df[,"Y"] - cv.lasso.1se.pred)^2)
    
    # PLS
    cv.pls <- plsr(Y ~ ., data = train_df, validation = "CV")
    pls.valid = cv.pls$validation # All the CV information
    pls.PRESS = pls.valid$PRESS    # Sum of squared CV residuals
    pls.MSPE = pls.PRESS / nrow(train_df)  # MSPE for internal CV
    pls.ind.best = which.min(pls.MSPE) # Optimal number of components
    cv.pls.pred = predict(cv.pls, valid_df, ncomp = pls.ind.best)
    MSPE[current_row,6] = get.MSPE(valid_df[,"Y"], cv.pls.pred)
    
    # RF
    cv.rf <- randomForest(Y~.,data=train_df,mtry = 11, nodesize = 15,ntree = 2000)
    cv.rf.pred = predict(cv.rf,newdata = valid_df)
    MSPE[current_row,7] = get.MSPE(valid_df[,"Y"], cv.rf.pred)
    
    
    
    # Boosting
    fit.gbm.best = gbm(Y ~ ., data =train_df, distribution = "gaussian", 
                       n.trees = 10000, interaction.depth = 8, shrinkage = 0.01)
    n.trees.best = gbm.perf(fit.gbm.best, plot.it = F) * 2 # Number of trees
    pred.best = predict(fit.gbm.best, valid_df, n.trees.best)
    MSPE[current_row,8] = get.MSPE(valid_df[,"Y"], pred.best)
    
    #NN
    X.train.raw =train_df[,-1]
    X.train = rescale(X.train.raw, X.train.raw)
    Y.train = train_df$Y
    
    X.valid.raw = valid_df[,-1]
    X.valid = rescale(X.valid.raw, X.train.raw)
    Y.valid = valid_df$Y
   
    fit.nnet = nnet(y = Y.train, x = X.train, linout = TRUE, size = 6,
                    decay = 1, maxit = 500)
    pred.nnet = predict(fit.nnet, X.valid)
    MSPE[current_row,9] = get.MSPE(Y.valid, pred.nnet)
    current_row = current_row + 1
  }
}
# Relative Boxplot
low.s = apply(MSPE, 1, min) 
boxplot(MSPE/low.s, ylim = c(1,1.5),
        main=paste0("Plot for RMSPE on ",V,"-folds validation"))

######################################### Random Forest Tuning #############################################################
######################################### Code From Tom's Lecture ##########################################################
# Default is mtry = 5, nodesize = 5
# Best so far 11|11
reps=20 # Doing lots of reps here because it's cheap
varz = c(9,10,11,12)
nodez = c(11,12,13,14,15,16,17,18,19)

NS = length(nodez)
M = length(varz)
rf.oob = matrix(NA, nrow=M*NS, ncol=reps)

for(r in 1:reps){
  print(paste0(r," of 20"))
  counter=1
  for(m in varz){
    for(ns in nodez){
      pro.rfm <- randomForest(data=df, Y~., ntree=500, 
                              mtry=m, nodesize=ns)
      rf.oob[counter,r] = mean((predict(pro.rfm) - df$Y)^2)
      counter=counter+1
    }
  }
}

parms = expand.grid(nodez,varz)
row.names(rf.oob) = paste(parms[,2], parms[,1], sep="|")

mean.oob = apply(rf.oob, 1, mean)
min.oob = apply(rf.oob, 2, min)
x11()
boxplot(t(rf.oob)/min.oob, use.cols=TRUE, las=2, 
        main="RF Tuning Variables and Node Sizes")
write.table(rf.oob,"ChrisL/RF_tuned_3.txt",sep = ",")
##################################################################################################################################
rf.oob = read.table("ChrisL/RF_tuned_3.txt",sep = ",")
x11()
mean.oob = apply(rf.oob, 1, mean)
min.oob = apply(rf.oob, 2, min)
boxplot(t(rf.oob)/min.oob, use.cols=TRUE, las=2, 
        main="RF Tuning Variables and Node Sizes")
