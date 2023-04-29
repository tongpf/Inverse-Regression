library('mlr3')
library('mlr3learners')
library('ranger')
#library("xlsx")
library('reshape2')
library('ggplot2')
library("ggsci")
library('foreach')
library('doParallel')
library('grid')
library('gridExtra')


rm(list = ls())

################################################################################
# we first study the performance with different number of repeated measurements
################################################################################

set.seed(2023)                                  # Fix seed
n = 2500                                        # Sample size
q = 10                                          # dimension of Y
p = 2                                           # dimension of X
pprime = 10                                     # dimension of Z
sigma = 1                                       # residual variance
sigma_z = 0.1                                   # Z variance

llist = c(1,2,4,6,8,10)                         # candidate l

repeat_trial = 2                               # number of simulation

mserecord_list = list()
biasrecord_list = list()

cl=makeCluster(2)
registerDoParallel(cl)
mserecord_list = foreach (trial = 1:repeat_trial,
                          .packages=c('mlr3','mlr3learners','ranger')) %dopar% {
  
  mserecord = as.data.frame(matrix(0,5,length(llist)))
  biasrecord = as.data.frame(matrix(0,5,length(llist)))
  colnames(mserecord) = paste0('l=',llist)
  colnames(biasrecord) = paste0('l=',llist)
  rownames(mserecord) = c('IR','GLS','BAY','OPT','RF')
  rownames(biasrecord) = c('IR','GLS','BAY','OPT','RF')
  
  print(trial)
  datadf_mean = matrix(NA,n,q+1+p+pprime)
  datadf_mean[,(q+1)] = 1                                         # intercept
  datadf_mean[,(q+2):(q+1+p)] = matrix(rnorm(n*p), nrow = n)      # mean of X
  datadf_mean[,(q+p+2):(q+1+p+pprime)] = matrix(rnorm(n*pprime), nrow = n) # mean of Z
  coeff = matrix(rnorm((1+p+pprime)*q), nrow = 1+p+pprime)        # generate coefficient (alpha B D)
  
  # split training and validation set
  obs_num = n
  ss <- sample(factor(rep(1:10, length.out=obs_num,)))
  table(ss)

  lmax=max(llist)

  nl = n*lmax
  noise_full = matrix(rnorm(nl*q,mean = 0, sd = sigma), nrow = nl) # generate residual E with l repeated measurements
  noise_z_full = matrix(rnorm(nl*pprime,mean = 0, sd = sigma_z), nrow = nl) # generate Z with l repeated measurements
  datadf_full = datadf_mean[rep(seq_len(nrow(datadf_mean)), each = lmax), ]
  datadf_full[,(q+p+2):(q+1+p+pprime)] = datadf_full[,(q+p+2):(q+1+p+pprime)]+noise_z_full
  datadf_full[,1:q] = datadf_full[,(q+1):(q+1+p+pprime)]%*%coeff + noise_full # generate Y
  
  datadf = datadf_full[seq(1,nrow(datadf_full),lmax),]
  datadf_bar_list = list()
  
  for (i in 1:length(llist)){ # take average of the first l repeated measurements
    l = llist[i]
    selectrows = rep(seq(1,nrow(datadf_full),lmax),each = l) + rep(c(0:(l-1)),times = n)
    datadf_bar = aggregate(datadf_full[selectrows,], list(rep(1:(n + 1), 
                                                 each = l, len = n*l)), mean)[-1]
    datadf_bar_list[[i]] = datadf_bar
  }
  
  datadf = as.data.frame(datadf)
  
  colnames(datadf)[q+1]='intercept'
  yname = colnames(datadf)[1:q]
  xname = colnames(datadf)[(q+2):(q+1+p)]
  zname = colnames(datadf)[(q+p+2):(q+1+p+pprime)]
  
  MSEIR = rep(0,length(llist))
  biasIR = rep(0,length(llist))

  MSEGLS = rep(0,length(llist))
  biasGLS = rep(0,length(llist))

  MSEBAY = rep(0,length(llist))
  biasBAY = rep(0,length(llist))

  MSEOPT = rep(0,length(llist))
  biasOPT = rep(0,length(llist))

  MSERF = rep(0,length(llist))
  biasRF = rep(0,length(llist))

  
  for (val_index in c(1:10)){ # ten-fold cross validation
    train_index = which(ss!=val_index)
    test_index = which(ss==val_index)
    datatrain = datadf[train_index,]
    
    #save the mean and standard deviation information
    meanadj = colMeans(datatrain[,c(1:q,(q+2):(q+1+p+pprime))])
    stdadj = sqrt(diag(var(datatrain[,c((q+2):(q+1+p+pprime))])))
    
    #demean of Y, X, Z
    datatrain[,c(1:q,(q+2):(q+1+p+pprime))] = 
      sweep(datatrain[,c(1:q,(q+2):(q+1+p+pprime))],2,meanadj)
    #standardize X and Z
    datatrain[,(q+2):(q+1+p+pprime)] = 
      sweep(datatrain[,(q+2):(q+1+p+pprime)],2,stdadj,"/")

    # inverse regression
    modelIR = as.formula(paste0('cbind(',paste(c(xname),collapse = ','),')~',
                                paste(c(yname,zname),collapse = '+')))
    modelIR = lm(modelIR, data = datatrain)
    summary(modelIR)
    
    # estimate coefficients of GLS and OPT manually
    X = as.matrix(datatrain[,(q+2):(q+1+p)])
    Z = as.matrix(datatrain[,(q+p+2):(q+1+p+pprime)])
    Y = as.matrix(datatrain[,1:q])
    
    Px = X%*%solve(t(X)%*%X)%*%t(X)
    Pz = Z%*%solve(t(Z)%*%Z)%*%t(Z)
    Mx = diag(rep(1,nrow(Px)))-Px
    Mz = diag(rep(1,nrow(Pz)))-Pz
    Bhat = solve(t(X)%*%Mz%*%X)%*%t(X)%*%Mz%*%Y
    Dhat = solve(t(Z)%*%Mx%*%Z)%*%t(Z)%*%Mx%*%Y
    ahat = colMeans(Y)#-t(Bhat)%*%colMeans(X)-t(Dhat)%*%colMeans(Z)
    Ehat = Y-rep(1,nrow(Y))%*%t(ahat)-X%*%Bhat-Z%*%Dhat

    S =t(Ehat)%*%Ehat
    Sinv = solve(S)
    
    # random forest
    learner_list = list()
    for (i in c(1:length(xname))){
      xname_i = xname[i]
      task = as_task_regr(datatrain[,c(yname, xname_i, zname)],target = xname_i)
      learner = lrn("regr.ranger")
      learner$train(task)
      learner_list[[i]] = learner
    }

    # now we calibrate X in the test dataset
    for (i in 1:length(llist)){
      l = llist[i]
      datatest = datadf_bar_list[[i]][test_index,]
      # preprocess Y and Z in the same way as what in the training dataset
      datatest[,c(1:q,(q+p+2):(q+1+p+pprime))] = 
        sweep(datatest[,c(1:q,(q+p+2):(q+1+p+pprime))],2,meanadj[c(1:q,(q+p+1):(q+p+pprime))])
      datatest[,c((q+p+2):(q+1+p+pprime))] = 
        sweep(datatest[,c((q+p+2):(q+1+p+pprime))],2,stdadj[c((p+1):(p+pprime))],"/")
      
      #inverse regression
      xhat = t(t(predict(modelIR,datatest[,c(1:q,(q+p+2):(q+1+p+pprime))]))*stdadj[(1):(p)]+
                 meanadj[(q+1):(q+p)])
      MSEIR[i] = MSEIR[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasIR[i] = biasIR[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
      
      #GLS
      Zprime = as.matrix(datatest[,(q+p+2):(q+1+p+pprime)])
      Yprime = as.matrix(datatest[,c(1:q)])

      xhat = (Yprime-rep(1,nrow(Yprime))%*%t(ahat)-Zprime%*%Dhat)%*%Sinv%*%t(Bhat)%*%
        solve(Bhat%*%Sinv%*%t(Bhat))
      xhat = t(t(xhat)*stdadj[(1):(p)]+meanadj[(q+1):(q+p)])
      MSEGLS[i] = MSEGLS[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasGLS[i] = biasGLS[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
      
      #Bayes
      xhat = (Yprime-rep(1,nrow(Yprime))%*%t(ahat)-Zprime%*%solve(t(Z)%*%Z)%*%t(Z)%*%X%*%Bhat
              -Zprime%*%Dhat)%*%Sinv%*%t(Bhat)%*%solve(solve(t(X)%*%Mz%*%X)/1+Bhat%*%Sinv%*%t(Bhat))+
        Zprime%*%solve(t(Z)%*%Z)%*%t(Z)%*%X
      xhat = t(t(xhat)*stdadj[(1):(p)]+meanadj[(q+1):(q+p)])
      MSEBAY[i] = MSEBAY[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasBAY[i] = biasBAY[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
      
      #optimal shrinkage
      xhat = (Yprime-rep(1,nrow(Yprime))%*%t(ahat)-Zprime%*%solve(t(Z)%*%Z)%*%t(Z)%*%X%*%Bhat
              -Zprime%*%Dhat)%*%Sinv%*%t(Bhat)%*%solve(solve(t(X)%*%Mz%*%X)/l+Bhat%*%Sinv%*%t(Bhat))+
        Zprime%*%solve(t(Z)%*%Z)%*%t(Z)%*%X
      xhat = t(t(xhat)*stdadj[(1):(p)]+meanadj[(q+1):(q+p)])
      MSEOPT[i] = MSEOPT[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasOPT[i] = biasOPT[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
      
      #random forest
      for (j in c(1:length(xname))){
        xhat[,j] = learner_list[[j]]$predict_newdata(datatest[,c(yname, zname)])$response
      }
      xhat = t(t(xhat)*stdadj[(1):(p)]+meanadj[(q+1):(q+p)])
      MSERF[i] = MSERF[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasRF[i] = biasRF[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
    }
    rm(Mx,Mz,Px,Pz)
    gc()
  }
  mserecord[1,] = mserecord[1,] + MSEIR/10
  biasrecord[1,] = biasrecord[1,] + biasIR/10
  
  mserecord[2,] = mserecord[2,] + MSEGLS/10
  biasrecord[2,] = biasrecord[2,] + biasGLS/10
  
  mserecord[3,] = mserecord[3,] + MSEBAY/10
  biasrecord[3,] = biasrecord[3,] + biasBAY/10
  
  mserecord[4,] = mserecord[4,] + MSEOPT/10
  biasrecord[4,] = biasrecord[4,] + biasOPT/10
  
  mserecord[5,] = mserecord[5,] + MSERF/10
  biasrecord[5,] = biasrecord[5,] + biasRF/10
  
  #mserecord_list[[trial]] = mserecord
  #biasrecord_list[[trial]] = biasrecord
  mserecord
}
stopCluster(cl)
stopImplicitCluster()


mserecord_final = do.call(rbind, mserecord_list)
#biasrecord_final = do.call(rbind, biasrecord_list)

save(mserecord_list,file='mserecord_list_l.RData')

#write.csv(biasrecord_final,'bias.csv')
write.csv(mserecord_final,'mse.csv')

#load('mserecord_list_l.RData')

mserecord_mean = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), mean))
mserecord_90quantile = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), 
                                        c(1,2), function(x){quantile(x,0.90)}))
mserecord_10quantile = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), 
                                        c(1,2), function(x){quantile(x,0.10)}))
mserecord_sd = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), sd))

mserecord_mean

################################################################################
# we then study the performance of different noise ratio                       #
################################################################################

sigmalist = c(0.5,1,2,4,8,16)                           # candidate noise level
l = 3                                                   # fix the number of l

mserecord_list = list()
biasrecord_list = list()

cl=makeCluster(2)
registerDoParallel(cl)
mserecord_list = foreach (trial = 1:repeat_trial, 
                          .packages=c('mlr3','mlr3learners','ranger')) %dopar% {
  
  mserecord = as.data.frame(matrix(0,5,length(sigmalist)))
  biasrecord = as.data.frame(matrix(0,5,length(sigmalist)))
  colnames(mserecord) = paste0('sigma=',sigmalist)
  colnames(biasrecord) = paste0('sigma=',sigmalist)
  rownames(mserecord) = c('IR','GLS','BAY','OPT','RF')
  rownames(biasrecord) = c('IR','GLS','BAY','OPT','RF')
  
  print(trial)
  datadf_mean = matrix(NA,n,q+1+p+pprime)
  datadf_mean[,(q+1)] = 1
  datadf_mean[,(q+2):(q+1+p)] = matrix(rnorm(n*p), nrow = n)
  datadf_mean[,(q+p+2):(q+1+p+pprime)] = matrix(rnorm(n*pprime), nrow = n)
  coeff = matrix(rnorm((1+p+pprime)*q), nrow = 1+p+pprime)
  
  obs_num = n
  ss <- sample(factor(rep(1:10, length.out=obs_num,)))
  table(ss)
  
  lmax=l
  
  nl = n*lmax
  
  noise_z_full = matrix(rnorm(nl*pprime,mean = 0, sd = sigma_z), nrow = nl) 
  datadf_full = datadf_mean[rep(seq_len(nrow(datadf_mean)), each = lmax), ]
  datadf_full[,(q+p+2):(q+1+p+pprime)] = datadf_full[,(q+p+2):(q+1+p+pprime)]+noise_z_full
  
  #generate Y at different noise levels
  datadf_list = list()
  datadf_bar_list = list()
  for (i in 1:length(sigmalist)){
    sigma = sqrt(sigmalist[i])                           #select the noise level
    noise_full = matrix(rnorm(nl*q,mean = 0, sd = sigma), nrow = nl) 
    datadf_full[,1:q] = datadf_full[,(q+1):(q+1+p+pprime)]%*%coeff + noise_full
    datadf = datadf_full[seq(1,nrow(datadf_full),lmax),]
    datadf_list[[i]] = datadf
    
    datadf_bar = aggregate(datadf_full, list(rep(1:(n + 1), each = l, len = n*l)), mean)[-1]
    datadf_bar_list[[i]] = datadf_bar
  }
  
  MSEIR = rep(0,length(sigmalist))
  biasIR = rep(0,length(sigmalist))
  
  MSEGLS = rep(0,length(sigmalist))
  biasGLS = rep(0,length(sigmalist))
  
  MSEBAY = rep(0,length(sigmalist))
  biasBAY = rep(0,length(sigmalist))
  
  MSEOPT = rep(0,length(sigmalist))
  biasOPT = rep(0,length(sigmalist))
  
  MSERF = rep(0,length(sigmalist))
  biasRF = rep(0,length(sigmalist))
  
  for (i in 1:length(sigmalist)){
    print(i)
    datadf = as.data.frame(datadf_list[[i]])
    colnames(datadf)[q+1]='intercept'
    yname = colnames(datadf)[1:q]
    xname = colnames(datadf)[(q+2):(q+1+p)]
    zname = colnames(datadf)[(q+p+2):(q+1+p+pprime)]
    
    for (val_index in c(1:10)){
      train_index = which(ss!=val_index)
      test_index = which(ss==val_index)
      datatrain = datadf[train_index,]

      #save the mean and standard deviation information
      meanadj = colMeans(datatrain[,c(1:q,(q+2):(q+1+p+pprime))])
      stdadj = sqrt(diag(var(datatrain[,c((q+2):(q+1+p+pprime))])))
      
      #demean of Y, X, Z
      datatrain[,c(1:q,(q+2):(q+1+p+pprime))] = 
        sweep(datatrain[,c(1:q,(q+2):(q+1+p+pprime))],2,meanadj)
      #standardize X and Z
      datatrain[,(q+2):(q+1+p+pprime)] = 
        sweep(datatrain[,(q+2):(q+1+p+pprime)],2,stdadj,"/")

      #inverse regression
      modelIR = as.formula(paste0('cbind(',paste(c(xname),collapse = ','),')~',
                                  paste(c(yname,zname),collapse = '+')))
      modelIR = lm(modelIR, data = datatrain)
      summary(modelIR)
      
      #coefficients
      X = as.matrix(datatrain[,(q+2):(q+1+p)])
      Z = as.matrix(datatrain[,(q+p+2):(q+1+p+pprime)])
      Y = as.matrix(datatrain[,1:q])
      
      Px = X%*%solve(t(X)%*%X)%*%t(X)
      Pz = Z%*%solve(t(Z)%*%Z)%*%t(Z)
      Mx = diag(rep(1,nrow(Px)))-Px
      Mz = diag(rep(1,nrow(Pz)))-Pz
      Bhat = solve(t(X)%*%Mz%*%X)%*%t(X)%*%Mz%*%Y
      Dhat = solve(t(Z)%*%Mx%*%Z)%*%t(Z)%*%Mx%*%Y
      ahat = colMeans(Y)#-t(Bhat)%*%colMeans(X)-t(Dhat)%*%colMeans(Z)
      Ehat = Y-rep(1,nrow(Y))%*%t(ahat)-X%*%Bhat-Z%*%Dhat
      
      S =t(Ehat)%*%Ehat
      Sinv = solve(S)
      
      #random forest
      learner_list = list()
      for (j in c(1:length(xname))){
        xname_i = xname[j]
        task = as_task_regr(datatrain[,c(yname, xname_i, zname)],target = xname_i)
        learner = lrn("regr.ranger")
        learner$train(task)
        learner_list[[j]] = learner
      }
      
      # now we begin the prediction
      # preprocess Y and Z in the same way as what in the training dataset
      datatest = datadf_bar_list[[i]][test_index,]
      datatest[,c(1:q,(q+p+2):(q+1+p+pprime))] = 
        sweep(datatest[,c(1:q,(q+p+2):(q+1+p+pprime))],2,meanadj[c(1:q,(q+p+1):(q+p+pprime))])
      datatest[,c((q+p+2):(q+1+p+pprime))] = 
        sweep(datatest[,c((q+p+2):(q+1+p+pprime))],2,stdadj[c((p+1):(p+pprime))],"/")
      
      #IR
      xhat = t(t(predict(modelIR,datatest[,c(1:q,(q+p+2):(q+1+p+pprime))]))*stdadj[(1):(p)]+meanadj[(q+1):(q+p)])
      MSEIR[i] = MSEIR[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasIR[i] = biasIR[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
      
      #GLS
      Zprime = as.matrix(datatest[,(q+p+2):(q+1+p+pprime)])
      Yprime = as.matrix(datatest[,c(1:q)])
      
      xhat = (Yprime-rep(1,nrow(Yprime))%*%t(ahat)-Zprime%*%Dhat)%*%Sinv%*%t(Bhat)%*%
        solve(Bhat%*%Sinv%*%t(Bhat))
      xhat = t(t(xhat)*stdadj[(1):(p)]+meanadj[(q+1):(q+p)])
      MSEGLS[i] = MSEGLS[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasGLS[i] = biasGLS[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
      
      #Bayes
      xhat = (Yprime-rep(1,nrow(Yprime))%*%t(ahat)-Zprime%*%solve(t(Z)%*%Z)%*%t(Z)%*%X%*%Bhat
              -Zprime%*%Dhat)%*%Sinv%*%t(Bhat)%*%solve(solve(t(X)%*%Mz%*%X)/1+Bhat%*%Sinv%*%t(Bhat))+
        Zprime%*%solve(t(Z)%*%Z)%*%t(Z)%*%X
      xhat = t(t(xhat)*stdadj[(1):(p)]+meanadj[(q+1):(q+p)])
      MSEBAY[i] = MSEBAY[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasBAY[i] = biasBAY[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
      
      #OPT
      xhat = (Yprime-rep(1,nrow(Yprime))%*%t(ahat)-Zprime%*%solve(t(Z)%*%Z)%*%t(Z)%*%X%*%Bhat
              -Zprime%*%Dhat)%*%Sinv%*%t(Bhat)%*%solve(solve(t(X)%*%Mz%*%X)/l+Bhat%*%Sinv%*%t(Bhat))+
        Zprime%*%solve(t(Z)%*%Z)%*%t(Z)%*%X
      xhat = t(t(xhat)*stdadj[(1):(p)]+meanadj[(q+1):(q+p)])
      MSEOPT[i] = MSEOPT[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasOPT[i] = biasOPT[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
      
      #RF
      for (j in c(1:length(xname))){
        xhat[,j] = learner_list[[j]]$predict_newdata(datatest[,c(yname, zname)])$response
      }
      xhat = t(t(xhat)*stdadj[(1):(p)]+meanadj[(q+1):(q+p)])
      MSERF[i] = MSERF[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat)^2)
      biasRF[i] = biasRF[i]+mean(as.matrix(datatest[,c((q+2):(q+1+p))]-xhat))
      rm(Mx,Mz,Px,Pz)
      gc()
    }
  }
  
  mserecord[1,] = mserecord[1,] + MSEIR/10
  biasrecord[1,] = biasrecord[1,] + biasIR/10
  
  mserecord[2,] = mserecord[2,] + MSEGLS/10
  biasrecord[2,] = biasrecord[2,] + biasGLS/10
  
  mserecord[3,] = mserecord[3,] + MSEBAY/10
  biasrecord[3,] = biasrecord[3,] + biasBAY/10
  
  mserecord[4,] = mserecord[4,] + MSEOPT/10
  biasrecord[4,] = biasrecord[4,] + biasOPT/10
  
  mserecord[5,] = mserecord[5,] + MSERF/10
  biasrecord[5,] = biasrecord[5,] + biasRF/10
  
  mserecord
}
stopCluster(cl)
stopImplicitCluster()

mserecord_final = do.call(rbind, mserecord_list)
#biasrecord_final = do.call(rbind, biasrecord_list)

save(mserecord_list,file='mserecord_list_sigma.RData')
#write.csv(biasrecord_final,'bias_sigma.csv')
write.csv(mserecord_final,'mse_sigma.csv')

#load('mserecord_list_sigma.RData')

mserecord_mean = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), mean))
mserecord_sd = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), sd))

mserecord_90quantile = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), 
                                        c(1,2), function(x){quantile(x,0.90)}))
mserecord_10quantile = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), 
                                        c(1,2), function(x){quantile(x,0.10)}))
mserecord_sd = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), sd))
mserecord_mean






















