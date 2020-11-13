library(caret)
library(MASS)
library(glmnet)
library(pls)
library(mgcv)
library(randomForest)
library(gbm)
library(nnet)
# Parallel computing
library(doParallel)
# Change this number for the exact computer clusters you have in you computer
# usually CPU core *2 
rm(list = ls())
cl <- makePSOCKcluster(12)
registerDoParallel(cl)

df = read.csv("Data2020.csv")
# set.seed(100)
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

# Boosting 
# Best is 0.001,8,10000 over this large grid 
tune.boosting <- expand.grid(shrinkage = c(0.0001,0.001,0.01,0.1), 
                  interaction.depth = c(3,4,5,6,7,8,9,10),
                  n.minobsinnode = 10,
                  n.trees = c(2000,5000,10000,15000))
gbm_model <- train(Y~ .,df, 
                   method = "gbm", 
                   trControl = custom, 
                   tuneGrid =tune.boosting)
plot(varImp(gbm_model,scale=F))

# NN
# Best is size = 33, decay = 0.8
tuned.nnet <- train(Y~.,df, method="nnet", 
                    trace=FALSE, linout=TRUE, 
                    trControl=custom, preProcess="range", 
                    tuneGrid = expand.grid(size=c(1:200),decay=c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)))
plot(varImp(tuned.nnet))

# RF
# Best is mtry = 11, ntree = 2000, nodesize = 15
all.MSEs = rep(NA, times = 80)
all.RF = list(1:80) 
tg <- data.frame(mtry = 3:12)
i = 1
for (this_ntree in c(500,1000,1500,2000)) {
  for (this_node in 2:21) {
    r.f <- train(Y~., df, 
                 method = "rf", 
                 tuneGrid = tg,
                 nodesize= this_node,
                 ntree=this_ntree,
                 trControl=custom)
    all.MSEs[i] = min(r.f$results[,2])
    all.RF[[i]] = r.f
    i = i+1
    print(paste0(i," of 80"))
  }
}
all.RF[which.min(all.MSEs)]
r.f$results[,2]
plot(varImp(r.f,scale=F))

#second approach
cv.rf <- randomForest(Y~.,data=df,mtry = 11, nodesize = 11,ntree = 500)
varImpPlot(cv.rf)


model.list=list(lm=lm,ridge=ridge,lasso=lasso,step=step,nn=tuned.nnet,boosting=gbm_model,rf=r.f)
summary(resamples(model.list))

stopCluster(cl)

