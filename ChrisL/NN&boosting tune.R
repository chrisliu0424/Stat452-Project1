
library(gbm)

source("Helper Functions-2.R")

df = read.csv("Data2020.csv")


##############################
#           tunning          #
##############################

#BOOSTING
max.trees = 10000
all.shrink = c(0.001, 0.01, 0.1)
all.depth = c(1, 2, 3)
all.pars = expand.grid(shrink = all.shrink, depth = all.depth)
n.pars = nrow(all.pars)

set.seed(100)
### Number of folds
K = 10
### Get folds
n = nrow(df)
folds = get.folds(n, K)
### Create container for CV MSPEs
CV.MSPEs = array(0, dim = c(K, n.pars))

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  ### Split data
  data.train = df[folds != i,]
  data.valid = df[folds == i,]
  Y.valid = data.valid$Y
  ### Fit boosting models for each parameter combination
  for(j in 1:n.pars){
    ### Get current parameter values
    this.shrink = all.pars[j,"shrink"]
    this.depth = all.pars[j,"depth"]
    
    ### Fit model using current parameter values and resampling rate of 0.6
    fit.gbm = gbm(Y ~ ., data = data.train, distribution = "gaussian", 
                  n.trees = max.trees, interaction.depth = this.depth, shrinkage = this.shrink, 
                  bag.fraction = 0.6)
    
    ### use Tom's rule
    n.trees = gbm.perf(fit.gbm, plot.it = F) * 2
    
    ### Check to make sure that Tom's rule doesn't tell us to use more than 1000
    ### trees. If it does, add extra trees as necessary
    if(n.trees > max.trees){
      extra.trees = n.trees - max.trees
      fit.gbm = gbm.more(fit.gbm, extra.trees)
    }
    ### Get predictions and MSPE, then store MSPE
    pred.gbm = predict(fit.gbm, data.valid, n.trees)
    MSPE.gbm = get.MSPE(Y.valid, pred.gbm)
    
    CV.MSPEs[i, j] = MSPE.gbm # Be careful with indices for CV.MSPEs
  }
}
### We can now make an MSPE boxplot. First, add column names to indicate
### which parameter combination was used. Format is shrinkage-depth
names.pars = paste0(all.pars$shrink,"-",
                    all.pars$depth)
colnames(CV.MSPEs) = names.pars

### Make boxplot
boxplot(CV.MSPEs, las = 2, main = "MSPE Boxplot")

### Get relative MSPEs and make boxplot
CV.RMSPEs = apply(CV.MSPEs, 1, function(W) W/min(W))
CV.RMSPEs = t(CV.RMSPEs)
boxplot(CV.RMSPEs, las = 2, main = "RMSPE Boxplot")



###############################################################################
### 1. higer interaction depth seems better than lower.                     ###
### 2. small shrinkage does not seem to perform better                      ###
### 3. Add on larger depths and larger shrinkage                            ###
###############################################################################

set.seed(100)
### Set parameter values
### We will stick to resampling rate of 0.6, maximum of 10000 trees, and Tom's rule
max.trees = 10000
all.shrink = c(0.1,0.2,0.3,0.5)
all.depth = c(2,3, 4, 5)
all.pars = expand.grid(shrink = all.shrink, depth = all.depth)
n.pars = nrow(all.pars)

### Create container for CV MSPEs
CV.MSPEs2 = array(0, dim = c(K, n.pars))

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data
  data.train = df[folds != i,]
  data.valid = df[folds == i,]
  Y.valid = data.valid$Y
  
  
  
  ### Fit boosting models for each parameter combination
  for(j in 1:n.pars){
    ### Get current parameter values
    this.shrink = all.pars[j,"shrink"]
    this.depth = all.pars[j,"depth"]
    
    ### Fit model using current parameter values.
    fit.gbm = gbm(Y~ ., data = data.train, distribution = "gaussian", 
                  n.trees = max.trees, interaction.depth = this.depth, shrinkage = this.shrink, 
                  bag.fraction = 0.6)
    
    ### use Tom's rule
    n.trees = gbm.perf(fit.gbm, plot.it = F) * 2
    
    ### Check to make sure that Tom's rule doesn't tell us to use more than 1000
    ### trees. If it does, add extra trees as necessary
    if(n.trees > max.trees){
      extra.trees = n.trees - max.trees
      fit.gbm = gbm.more(fit.gbm, extra.trees)
    }
    
    ### Get predictions and MSPE, then store MSPE
    pred.gbm = predict(fit.gbm, data.valid, n.trees)
    MSPE.gbm = get.MSPE(Y.valid, pred.gbm)
    
    CV.MSPEs2[i, j] = MSPE.gbm # Be careful with indices for CV.MSPEs
  }
}

### We can now make an MSPE boxplot. First, add column names to indicate
### which parameter combination was used. Format is shrinkage-depth
names.pars = paste0(all.pars$shrink,"-",
                    all.pars$depth)
colnames(CV.MSPEs2) = names.pars

### Make boxplot
boxplot(CV.MSPEs2, las = 2, main = "MSPE Boxplot")


### Get relative MSPEs and make boxplot
CV.RMSPEs2 = apply(CV.MSPEs2, 1, function(W) W/min(W))
CV.RMSPEs2 = t(CV.RMSPEs2)
boxplot(CV.RMSPEs2, las = 2, main = "RMSPE Boxplot")



###############################################################################
### 1. Shrinkage = 0.1 obviously is better than other shrinkage parameter.  ###
### 2. try multiple depths                                                  ###
###############################################################################

set.seed(100)
### Set parameter values
### We will stick to resampling rate of 0.6, maximum of 10000 trees, and Tom's rule
max.trees = 10000
all.shrink = 0.1
all.depth = c(1,2,3, 4, 5,6,7,8)
all.pars = expand.grid(shrink = all.shrink, depth = all.depth)
n.pars = nrow(all.pars)

### Create container for CV MSPEs
CV.MSPEs3 = array(0, dim = c(K, n.pars))

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data
  data.train = df[folds != i,]
  data.valid = df[folds == i,]
  Y.valid = data.valid$Y
  
  
  
  ### Fit boosting models for each parameter combination
  for(j in 1:n.pars){
    ### Get current parameter values
    this.shrink = all.pars[j,"shrink"]
    this.depth = all.pars[j,"depth"]
    
    ### Fit model using current parameter values.
    fit.gbm = gbm(Y ~ ., data = data.train, distribution = "gaussian", 
                  n.trees = max.trees, interaction.depth = this.depth, shrinkage = this.shrink, 
                  bag.fraction = 0.6)
    
    ### use Tom's rule
    n.trees = gbm.perf(fit.gbm, plot.it = F) * 2
    
    ### Check to make sure that Tom's rule doesn't tell us to use more than 1000
    ### trees. If it does, add extra trees as necessary
    if(n.trees > max.trees){
      extra.trees = n.trees - max.trees
      fit.gbm = gbm.more(fit.gbm, extra.trees)
    }
    
    ### Get predictions and MSPE, then store MSPE
    pred.gbm = predict(fit.gbm, data.valid, n.trees)
    MSPE.gbm = get.MSPE(Y.valid, pred.gbm)
    
    CV.MSPEs3[i, j] = MSPE.gbm # Be careful with indices for CV.MSPEs
  }
}

### We can now make an MSPE boxplot. First, add column names to indicate
### which parameter combination was used. Format is shrinkage-depth
names.pars = paste0(all.pars$shrink,"-",
                    all.pars$depth)
colnames(CV.MSPEs3) = names.pars

### Make boxplot
boxplot(CV.MSPEs3, las = 2, main = "MSPE Boxplot")


### Get relative MSPEs and make boxplot
CV.RMSPEs3 = apply(CV.MSPEs3, 1, function(W) W/min(W))
CV.RMSPEs3 = t(CV.RMSPEs3)
boxplot(CV.RMSPEs3, las = 2, main = "RMSPE Boxplot")



### Based on the RMSPE boxplot, the model with shrinkage = 0.1 and depth = 4 looks the best
fit.gbm.best = gbm(Y ~ ., data =df, distribution = "gaussian", 
                   n.trees = 10000, interaction.depth = 4, shrinkage = 0.1, bag.fraction = 0.6)

n.trees.best = gbm.perf(fit.gbm.best, plot.it = F) * 2 # Number of trees
















#Neural net

M = 20 # Number of times to re-fit each model

### Define parameter values and use expand.grid() to get all combinations
all.n.hidden = c(1, 3,5,7,9 )
all.shrink = c(0.001, 0.1, 0.5, 1,2)
all.pars = expand.grid(n.hidden = all.n.hidden,
                       shrink = all.shrink)
n.pars = nrow(all.pars) # Number of parameter combinations


### Create container for MSPEs
CV.nn.MSPEs = array(0, dim = c(K, n.pars))

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data and rescale predictors
  data.train = df[folds != i,]
  X.train.raw = data.train[,-1]
  X.train = rescale(X.train.raw, X.train.raw)
  Y.train = data.train$Y
  
  data.valid = df[folds == i,]
  X.valid.raw = data.valid[,-1]
  X.valid = rescale(X.valid.raw, X.train.raw)
  Y.valid = data.valid$Y
  
  
  ### Fit neural net models for each parameter combination. A second 
  ### for loop will make our life easier here
  for(j in 1:n.pars){
    ### Get current parameter values
    this.n.hidden = all.pars[j,1]
    this.shrink = all.pars[j,2]
    
    ### We need to run nnet multiple times to avoid bad local minima. Create
    ### containers to store the models and their errors.
    all.nnets = list(1:M)
    all.SSEs = rep(0, times = M)
    
    ### We need to fit each model multiple times. This calls for another
    ### for loop.
    for(l in 1:M){
      ### Fit model
      fit.nnet = nnet(X.train, Y.train, linout = TRUE, size = this.n.hidden,
                      decay = this.shrink, maxit = 500, trace = FALSE)
      
      ### Get model SSE
      SSE.nnet = fit.nnet$value
      
      ### Store model and its SSE
      all.nnets[[l]] = fit.nnet
      all.SSEs[l] = SSE.nnet
    }
    
    ### Get best fit using current parameter values
    ind.best = which.min(all.SSEs)
    fit.nnet.best = all.nnets[[ind.best]]
    
    ### Get predictions and MSPE, then store MSPE
    pred.nnet = predict(fit.nnet.best, X.valid)
    MSPE.nnet = get.MSPE(Y.valid, pred.nnet)
    
    CV.nn.MSPEs[i, j] = MSPE.nnet # Be careful with indices for CV.MSPEs
    
    
  }
}

### We can now make an MSPE boxplot. It would be nice to have more 
### informative names though. We can construct names from all.pars
### using the paste0() function.
names.pars = paste0(all.pars$n.hidden,",",
                    all.pars$shrink)
colnames(CV.nn.MSPEs) = names.pars

### Make boxplot
boxplot(CV.nn.MSPEs, las = 2, main = "MSPE Boxplot",ylim=c(1,3))


### Get relative MSPEs and make boxplot
CV.nn.RMSPEs = apply(CV.nn.MSPEs, 1, function(W) W/min(W))
CV.nn.RMSPEs = t(CV.nn.RMSPEs)
boxplot(CV.nn.RMSPEs, las = 2, main = "RMSPE Boxplot",ylim=c(1,1.2))


###############################################################################
### 1. Shrinkage=0.001 and 0.1 is not stable and there's an increasing trend### 
###    as we increase hidden nodes.                                         ###
### 2. large shrinkage is more stable                                       ###
### 3. Try larger shrinkage parameters                                      ###
###############################################################################


M = 20 # Number of times to re-fit each model

### Define parameter values and use expand.grid() to get all combinations
all.n.hidden = c(1, 3,5,7,9 )
all.shrink = c(1,1.5,2,2.5,3)
all.pars = expand.grid(n.hidden = all.n.hidden,
                       shrink = all.shrink)
n.pars = nrow(all.pars) # Number of parameter combinations


### Create container for MSPEs
CV.nn.MSPEs = array(0, dim = c(K, n.pars))

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data and rescale predictors
  data.train = df[folds != i,]
  X.train.raw = data.train[,-1]
  X.train = rescale(X.train.raw, X.train.raw)
  Y.train = data.train$Y
  
  data.valid = df[folds == i,]
  X.valid.raw = data.valid[,-1]
  X.valid = rescale(X.valid.raw, X.train.raw)
  Y.valid = data.valid$Y
  
  
  ### Fit neural net models for each parameter combination. A second 
  ### for loop will make our life easier here
  for(j in 1:n.pars){
    ### Get current parameter values
    this.n.hidden = all.pars[j,1]
    this.shrink = all.pars[j,2]
    
    ### We need to run nnet multiple times to avoid bad local minima. Create
    ### containers to store the models and their errors.
    all.nnets = list(1:M)
    all.SSEs = rep(0, times = M)
    
    ### We need to fit each model multiple times. This calls for another
    ### for loop.
    for(l in 1:M){
      ### Fit model
      fit.nnet = nnet(X.train, Y.train, linout = TRUE, size = this.n.hidden,
                      decay = this.shrink, maxit = 500, trace = FALSE)
      
      ### Get model SSE
      SSE.nnet = fit.nnet$value
      
      ### Store model and its SSE
      all.nnets[[l]] = fit.nnet
      all.SSEs[l] = SSE.nnet
    }
    
    ### Get best fit using current parameter values
    ind.best = which.min(all.SSEs)
    fit.nnet.best = all.nnets[[ind.best]]
    
    ### Get predictions and MSPE, then store MSPE
    pred.nnet = predict(fit.nnet.best, X.valid)
    MSPE.nnet = get.MSPE(Y.valid, pred.nnet)
    
    CV.nn.MSPEs[i, j] = MSPE.nnet # Be careful with indices for CV.MSPEs
  
    
  }
}


### We can now make an MSPE boxplot. It would be nice to have more 
### informative names though. We can construct names from all.pars
### using the paste0() function.
names.pars = paste0(all.pars$n.hidden,",",
                    all.pars$shrink)
colnames(CV.nn.MSPEs) = names.pars

### Make boxplot
boxplot(CV.nn.MSPEs, las = 2, main = "MSPE Boxplot",ylim=c(1,2))


### Get relative MSPEs and make boxplot
CV.nn.RMSPEs = apply(CV.nn.MSPEs, 1, function(W) W/min(W))
CV.nn.RMSPEs = t(CV.nn.RMSPEs)
boxplot(CV.nn.RMSPEs, las = 2, main = "RMSPE Boxplot",ylim=c(1,1.05))


#MSPE very similar, might not be wrong choosing any of those.
fit.nnet = nnet(y = Y.train, x = X.train, linout = TRUE, size = 3,
                decay = 1.5, maxit = 500)
