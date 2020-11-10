library(caret)
library(MASS)
library(glmnet)
library(pls)
library(mgcv)
library(randomForest)
library(gbm)
library(nnet)

df = read.csv("Data2020.csv")
set.seed(100)
custom =trainControl(method="repeatedcv",
                     number=10,
                     repeats=5,
                     verboseIter = T)

#lm
lm=train(Y~., df,method='lm',trControl=custom)
lm$results
plot(varImp(lm,scale=F))

#ridge
ridge=train(Y~., df,
            method="glmnet", 
            tuneGrid=expand.grid(alpha=0, lambda = seq(0, 100, .05)),
            trControl=custom)
plot(varImp(ridge,scale=F))

#LASSO
lasso=train(Y~., df,
            method="glmnet",
            tuneGrid=expand.grid(alpha=1,lambda = seq(0, 100, .05)),
            trControl=custom)
plot(varImp(lasso,scale=F))

step <- train(Y ~.,df, 
              method = "lmStepAIC", 
              trControl = custom, trace = FALSE)
plot(varImp(step,scale=F))

#boosting(don't further tune, will take a long time)
tune.boosting <- expand.grid(shrinkage = 0.1, 
                  interaction.depth = 4,
                  n.minobsinnode = 10,
                  n.trees = 10000)
gbm_model <- train(Y~ .,df, 
                   method = "gbm", 
                   trControl = custom, 
                   tuneGrid =tune.boosting)
plot(varImp(gbm_model,scale=F))

#NN
tuned.nnet <- train(Y~.,df, method="nnet", 
                    trace=FALSE, linout=TRUE, 
                    trControl=custom, preProcess="range", 
                    tuneGrid = expand.grid(size=c(1,2,3, 4, 5,6,7,8),decay=c(0.1,0.2,0.3,0.5)))
plot(varImp(tuned.nnet))

#RF
tg <- data.frame(mtry = 3:12)
r.f <- train(Y~., df, 
             method = "rf", 
             tuneGrid = tg,
             nodesize=c(4,5,6,7,8,9),
             ntree=500,
             trControl=custom)
r.f$results
plot(varImp(r.f,scale=F))

#second approach
cv.rf <- randomForest(Y~.,data=df,mtry = 11, nodesize = 11,ntree = 500)
varImpPlot(cv.rf)


model.list=list(lm=lm,ridge=ridge,lasso=lasso,step=step,nn=tuned.nnet,boosting=gbm_model,rf=r.f)
summary(resamples(model.list))


