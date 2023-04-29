library("FRK")
library("sp")
library("ggpubr")

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

rm(list=ls())

################################################################################
# we first study the performance with different number of repeated measurements
################################################################################

## Setup
set.seed(2023)                                                # Fix seed
n <- 2500                                                     # Sample size
p = 5
k = 9
q1 = 2
q2 = 10
sigma = 1 
sigma_z = 0.1

llist = c(1,2,4,6,8,10)

repeat_trial = 10

mserecord_list = list()
biasrecord_list = list()

cl=makeCluster(2)
registerDoParallel(cl)
mserecord_list = foreach (trial = 1:repeat_trial, 
                          .packages=c('mlr3','mlr3learners','ranger','FRK','sp')) %dopar% {
  
  mserecord = as.data.frame(matrix(0,5,length(llist)))
  biasrecord = as.data.frame(matrix(0,5,length(llist)))
  colnames(mserecord) = paste0('l=',llist)
  colnames(biasrecord) = paste0('l=',llist)
  rownames(mserecord) = c('IR-lm','GLS','IR','opt','RF')
  rownames(biasrecord) = c('IR-lm','GLS','IR','opt','RF')
  
  print(trial)
  
  zdf <- data.frame(x = runif(n), y= runif(n))               # Generate random locs
  coordinates(zdf) = ~x+y                                    # Turn into sp object
  
  datadf_mean = matrix(NA,n,p+1+q1+q2)
  datadf_mean[,(p+1)] = 1
  datadf_mean[,(p+2):(p+1+q1)] = matrix(rnorm(n*q1), nrow = n)
  datadf_mean[,(p+q1+2):(p+1+q1+q2)] = matrix(rnorm(n*q2), nrow = n)
  coeff = matrix(rnorm((1+q1+q2)*p),
                 nrow = 1+q1+q2)
  
  G = auto_basis(manifold = plane(), data = zdf, regular = 1, 
                 nres = 1, tunit = "hours", type = 'bisquare')
  # show_basis(G)
  Psimat = eval_basis(G, zdf@coords)
  Psimat = as.matrix(Psimat)
  
  # plotcoord = zdf@coords
  # plotcoord = cbind(plotcoord, Psimat)
  # plotcoord = as.data.frame(plotcoord)
  # ggplot(plotcoord, aes(x = x, y = y)) + geom_point(aes(color = V3))
  
  A_raw = matrix(rnorm(k*p), nrow = k)
  
  datadf_mean = cbind(datadf_mean, Psimat)
  coeff = rbind(coeff, A_raw)
  
  obs_num = n
  ss <- sample(factor(rep(1:10, length.out=obs_num,)))
  table(ss)

  lmax=max(llist)

  nl = n*lmax
  noise_full = matrix(rnorm(nl*p,mean = 0, sd = sigma), nrow = nl) 
  noise_z_full = matrix(rnorm(nl*q2,mean = 0, sd = sigma_z), nrow = nl) 
  datadf_full = datadf_mean[rep(seq_len(nrow(datadf_mean)), each = lmax), ]
  datadf_full[,(p+q1+2):(p+1+q1+q2)] = datadf_full[,(p+q1+2):(p+1+q1+q2)]+noise_z_full
  datadf_full[,1:p] = datadf_full[,(p+1):(p+1+q1+q2+k)]%*%coeff + noise_full
  
  datadf = datadf_full[seq(1,nrow(datadf_full),lmax),]
  datadf = as.data.frame(datadf)
  
  # plotcoord = zdf@coords
  # plotcoord = cbind(plotcoord, datadf)
  # plotcoord = as.data.frame(plotcoord)
  # ggplot(plotcoord, aes(x = x, y = y)) + geom_point(aes(color = V5))
  
  datadf_bar_list = list()
  
  for (i in 1:length(llist)){
    l = llist[i]
    selectrows = rep(seq(1,nrow(datadf_full),lmax),each = l) + rep(c(0:(l-1)),times = n)
    datadf_bar = aggregate(datadf_full[selectrows,], list(rep(1:(n + 1), 
                                                 each = l, len = n*l)), mean)[-1]
    datadf_bar_list[[i]] = datadf_bar
  }
  
  colnames(datadf)[p+1]='intercept'
  yname = colnames(datadf)[1:p]
  xname = colnames(datadf)[(p+2):(p+1+q1)]
  zname = colnames(datadf)[(p+q1+2):(p+1+q1+q2)]
  
  MSEGLS = rep(0,length(llist))
  biasGLS = rep(0,length(llist))

  MSEtrueinverse = rep(0,length(llist))
  biastrueinverse = rep(0,length(llist))

  MSEbestinverse = rep(0,length(llist))
  biasbestinverse = rep(0,length(llist))

  MSERF = rep(0,length(llist))
  biasRF = rep(0,length(llist))

  
  for (val_index in c(1:10)){
    train_index = which(ss!=val_index)
    test_index = which(ss==val_index)
    datatrain = datadf[train_index,]
    # datatest = datadf[test_index,]
    # datatest_bar = datadf_bar[test_index,]
    meanadj = colMeans(datatrain[(p+2):(p+1+q1+q2)])
    stdadj = sqrt(diag(var(datatrain[(p+2):(p+1+q1+q2)])))
    datatrain[(p+2):(p+1+q1+q2)] = sweep(sweep(datatrain[(p+2):(p+1+q1+q2)],2,meanadj),2,stdadj,"/")
    # datatest[,c(1:p,(p+q1+1):(p+q1+q2))] = sweep(sweep(datatest[,c(1:p,(p+q1+1):(p+q1+q2))],2,
    #                                                    meanadj[c(1:p,(p+q1+1):(p+q1+q2))]),2,stdadj[c(1:p,(p+q1+1):(p+q1+q2))],"/")
    # datatest_bar[,c(1:p,(p+q1+1):(p+q1+q2))] = sweep(sweep(datatest_bar[,c(1:p,(p+q1+1):(p+q1+q2))],2,
    #                                                        meanadj[c(1:p,(p+q1+1):(p+q1+q2))]),2,stdadj[c(1:p,(p+q1+1):(p+q1+q2))],"/")
    Psitrain = Psimat[train_index,]
    Psitest = Psimat[test_index,]
    
    
    X = as.matrix(datatrain[,xname])
    Z = as.matrix(datatrain[,zname])
    Y = as.matrix(datatrain[,yname])
    Psitrain_tPsitrain = t(Psitrain)%*%Psitrain
    
    ntrain = dim(Y)[1]
    onematrix = matrix(1,nrow = ntrain, ncol = 1)
    
    mathcal_X = cbind(onematrix,X,Z)
    
    Gamma_1 = diag(rep(1,p))
    Gamma_2 = diag(rep(1,p))
    
    mathcal_B = matrix(0, nrow = 1+q1+q2, ncol = p)
    
    EMiter = 1
    loglikelihood_old = Inf
    deltaloglikelihood = Inf
    
    while(EMiter<=50&deltaloglikelihood>0.1){
      #print(EMiter)
      
      Gamma_2inv = solve(Gamma_2)
      Gamma_1Gamma_2inv = Gamma_1%*%Gamma_2inv
      midinvmat = solve(diag(rep(1,k*p))+kronecker(Psitrain_tPsitrain,Gamma_1Gamma_2inv))
      
      
      midmathcal_B = solve(kronecker(t(mathcal_X)%*%mathcal_X,Gamma_2inv)-
                             kronecker(t(mathcal_X)%*%Psitrain,Gamma_2inv)%*%midinvmat%*%
                             kronecker(t(Psitrain)%*%mathcal_X,Gamma_1Gamma_2inv))
      mathcal_B = midmathcal_B%*%(kronecker(t(mathcal_X),Gamma_2inv)%*%as.matrix(c(t(Y)))-
                                    kronecker(t(mathcal_X)%*%Psitrain,Gamma_2inv)%*%midinvmat%*%
                                    kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Y))))
      mathcal_B = t(matrix(c(mathcal_B),nrow = p))
      
      Yres = Y-mathcal_X%*%mathcal_B
      
      A = kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Yres)))
      A = A - kronecker(Psitrain_tPsitrain,Gamma_1Gamma_2inv)%*%
        midinvmat%*%kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Yres)))
      A = t(matrix(c(A),nrow = p))
      
      midGamma_1 = kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv%*%Gamma_1)-
        kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
        midinvmat%*%kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv%*%Gamma_1)
      midGamma_1_ = kronecker(t(Psitrain)%*%mathcal_X, Gamma_1Gamma_2inv)-
        kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
        midinvmat%*%kronecker(t(Psitrain)%*%mathcal_X, Gamma_1Gamma_2inv)
      midGamma_1 = midGamma_1-midGamma_1_%*%midmathcal_B%*%t(midGamma_1_)
      
      midGamma_1_new = matrix(0,nrow = p,ncol = p)
      for (j in c(1:k)){
        midGamma_1_new = midGamma_1_new + midGamma_1[(j*p-p+1):(j*p),(j*p-p+1):(j*p)]/k
      }
      
      Gamma_1 = Gamma_1+1/k*t(A)%*%A-midGamma_1_new
      
      Yhat = mathcal_X%*%mathcal_B+Psitrain%*%A
      Gamma_2 = 1/ntrain*t(Y-Yhat)%*%(Y-Yhat)
      
      EMiter = EMiter+1
      #rm(Mz,Mx)
      #gc()
      loglikelihood = ntrain*log(det(Gamma_2)) + k*log(det(Gamma_1)) + 
        sum(diag(A%*%solve(Gamma_1)%*%t(A))) + 
        sum(diag((Yres-Psitrain%*%A)%*%solve(Gamma_2)%*%t(Yres-Psitrain%*%A)))
      #print(loglikelihood)
      deltaloglikelihood = loglikelihood_old - loglikelihood
      loglikelihood_old = loglikelihood
      #print(deltaloglikelihood)
    }
    
    alpha_true = as.matrix(mathcal_B[1,])
    B_true = mathcal_B[2:(1+q1),]
    D_true = mathcal_B[(2+q1):(1+q1+q2),]
    A_true = A
    Gamma_1_true = Gamma_1
    Gamma_2_true = Gamma_2
    
    #reversed modeling
    
    mathcal_X = cbind(onematrix,Y,Z)
    
    Gamma_1 = diag(rep(1,q1))
    Gamma_2 = diag(rep(1,q1))
    
    mathcal_B = matrix(0, nrow = 1+p+q2, ncol = q1)
    
    EMiter = 1
    loglikelihood_old = Inf
    deltaloglikelihood = Inf
    
    while(EMiter<=50&deltaloglikelihood>0.1){
      #print(EMiter)
      
      Gamma_2inv = solve(Gamma_2)
      Gamma_1Gamma_2inv = Gamma_1%*%Gamma_2inv
      midinvmat = solve(diag(rep(1,k*q1))+kronecker(t(Psitrain)%*%Psitrain,Gamma_1Gamma_2inv))
      
      
      midmathcal_B = solve(kronecker(t(mathcal_X)%*%mathcal_X,Gamma_2inv)-
                             kronecker(t(mathcal_X)%*%Psitrain,Gamma_2inv)%*%midinvmat%*%
                             kronecker(t(Psitrain)%*%mathcal_X,Gamma_1Gamma_2inv))
      mathcal_B = midmathcal_B%*%(kronecker(t(mathcal_X),Gamma_2inv)%*%as.matrix(c(t(X)))-
                                    kronecker(t(mathcal_X)%*%Psitrain,Gamma_2inv)%*%midinvmat%*%
                                    kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(X))))
      mathcal_B = t(matrix(c(mathcal_B),nrow = q1))
      
      Xres = X-mathcal_X%*%mathcal_B
      
      A = kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Xres)))
      A = A - kronecker(t(Psitrain)%*%Psitrain,Gamma_1Gamma_2inv)%*%
        midinvmat%*%kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Xres)))
      A = t(matrix(c(A),nrow = q1))
      
      midGamma_1 = kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv%*%Gamma_1)-
        kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
        midinvmat%*%kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv%*%Gamma_1)
      midGamma_1_ = kronecker(t(Psitrain)%*%mathcal_X, Gamma_1Gamma_2inv)-
        kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
        midinvmat%*%kronecker(t(Psitrain)%*%mathcal_X, Gamma_1Gamma_2inv)
      midGamma_1 = midGamma_1-midGamma_1_%*%midmathcal_B%*%t(midGamma_1_)
      
      midGamma_1_new = matrix(0,nrow = q1,ncol = q1)
      for (j in c(1:k)){
        midGamma_1_new = midGamma_1_new + midGamma_1[(j*q1-q1+1):(j*q1),(j*q1-q1+1):(j*q1)]/k
      }
      
      Gamma_1 = Gamma_1+1/k*t(A)%*%A-midGamma_1_new
      
      Xhat = mathcal_X%*%mathcal_B+Psitrain%*%A
      Gamma_2 = 1/ntrain*t(X-Xhat)%*%(X-Xhat)+diag(diag(rep(1e-3,q1)))
      
      EMiter = EMiter+1
      #rm(Mz,Mx)
      #gc()
      loglikelihood = ntrain*log(det(Gamma_2)) + k*log(det(Gamma_1)) + 
        sum(diag(A%*%solve(Gamma_1)%*%t(A))) + 
        sum(diag((Xres-Psitrain%*%A)%*%solve(Gamma_2)%*%t(Xres-Psitrain%*%A)))
      #print(loglikelihood)
      deltaloglikelihood = loglikelihood_old - loglikelihood
      loglikelihood_old = loglikelihood
      #print(deltaloglikelihood)
    }
    
    alpha_wrong = as.matrix(mathcal_B[1,])
    B_wrong = mathcal_B[2:(1+p),]
    D_wrong = mathcal_B[(2+p):(1+p+q2),]
    A_wrong = A
    Gamma_1_wrong = Gamma_1
    Gamma_2_wrong = Gamma_2
    
    learner_list = list()
    for (j in c(1:length(xname))){
      xname_i = xname[j]
      task = as_task_regr(datatrain[,c(yname, xname_i, zname)],target = xname_i)
      learner = lrn("regr.ranger")
      learner$train(task)
      learner_list[[j]] = learner
    }

    
    for (i in 1:length(llist)){
      l = llist[i]
      datatest = datadf_bar_list[[i]][test_index,]
      datatest[,c((p+q1+2):(p+1+q1+q2))] = sweep(sweep(datatest[,c((p+q1+2):(p+1+q1+q2))],2,
                                                         meanadj[c((q1+1):(q1+q2))]),2,stdadj[c((q1+1):(q1+q2))],"/")
      
      Zprime = as.matrix(datatest[,(p+q1+2):(p+1+q1+q2)])
      Yprime = as.matrix(datatest[,c(1:p)])
      Psitrain_tPsitrain = t(Psitrain)%*%Psitrain
      Gamma_1Gamma_2inv = Gamma_1_true%*%solve(Gamma_2_true)
      midinvmat = solve(diag(rep(1,k*p))+kronecker(Psitrain_tPsitrain,Gamma_1Gamma_2inv))
      XTX_inv = solve(t(X)%*%X)*n
      
      xhat_GLS = matrix(NA,nrow = nrow(datatest), ncol = q1)
      xhat_OPT = matrix(NA,nrow = nrow(datatest), ncol = q1)
      for (j in c(1:nrow(datatest))){
        psi_test_j = as.matrix(Psitest[j,])
        S_j = kronecker(t(psi_test_j)%*%Psitrain_tPsitrain%*%psi_test_j, Gamma_1Gamma_2inv%*%Gamma_1_true)-
          kronecker(t(psi_test_j)%*%Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
          midinvmat%*%kronecker(Psitrain_tPsitrain%*%psi_test_j, Gamma_1Gamma_2inv%*%Gamma_1_true)
        S_j = Gamma_2_true/l+kronecker(t(psi_test_j)%*%psi_test_j,Gamma_1_true)-S_j
        S_j = solve(S_j)
        Yres_j = Yprime[j,]-alpha_true-t(D_true)%*%as.matrix(Zprime[j,])-t(A_true)%*%psi_test_j
        xhat_j = solve(B_true%*%S_j%*%t(B_true))%*%B_true%*%S_j%*%as.matrix(Yres_j)
        xhat_GLS[j,] = t(xhat_j)
        xhat_j = solve(B_true%*%S_j%*%t(B_true)+XTX_inv)%*%B_true%*%S_j%*%as.matrix(Yres_j)
        xhat_OPT[j,] = t(xhat_j)
      }

      xhat = t(t(xhat_GLS)*stdadj[(1):(q1)]+meanadj[(1):(q1)])
      MSEGLS[i] = MSEGLS[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat)^2)
      biasGLS[i] = biasGLS[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat))
      
      xhat = matrix(1,nrow = length(test_index), ncol = 1)%*%t(alpha_wrong) + 
        Yprime%*%B_wrong + Zprime%*%D_wrong + Psitest%*%A
      xhat = t(t(xhat)*stdadj[(1):(q1)]+meanadj[(1):(q1)])
      MSEtrueinverse[i] = MSEtrueinverse[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat)^2)
      biastrueinverse[i] = biastrueinverse[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat))
      
      xhat = t(t(xhat_OPT)*stdadj[(1):(q1)]+meanadj[(1):(q1)])
      MSEbestinverse[i] = MSEbestinverse[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat)^2)
      biasbestinverse[i] = biasbestinverse[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat))
      
      for (j in c(1:length(xname))){
        xhat[,j] = learner_list[[j]]$predict_newdata(datatest[,c(yname, zname)])$response
      }
      xhat = t(t(xhat)*stdadj[(1):(q1)]+meanadj[(1):(q1)])
      MSERF[i] = MSERF[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat)^2)
      biasRF[i] = biasRF[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat))
    }
  }

  mserecord[2,] = mserecord[2,] + MSEGLS/10
  biasrecord[2,] = biasrecord[2,] + biasGLS/10
  
  mserecord[3,] = mserecord[3,] + MSEtrueinverse/10
  biasrecord[3,] = biasrecord[3,] + biastrueinverse/10
  
  mserecord[4,] = mserecord[4,] + MSEbestinverse/10
  biasrecord[4,] = biasrecord[4,] + biasbestinverse/10
  
  mserecord[5,] = mserecord[5,] + MSERF/10
  biasrecord[5,] = biasrecord[5,] + biasRF/10
  
  #mserecord_list[[trial]] = mserecord
  #biasrecord_list[[trial]] = biasrecord
  mserecord
}
stopCluster(cl)
stopImplicitCluster()


mserecord_final = do.call(rbind, mserecord_list)
biasrecord_final = do.call(rbind, biasrecord_list)

#save(mserecord_list,file='STmserecord_list_l.RData')

#write.csv(biasrecord_final,'STbias.csv')
#write.csv(mserecord_final,'STmse.csv')

#load('STmserecord_list_l.RData')

mserecord_mean = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), mean))
mserecord_90quantile = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), 
                                        c(1,2), function(x){quantile(x,0.90)}))
mserecord_10quantile = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), 
                                        c(1,2), function(x){quantile(x,0.10)}))
mserecord_sd = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), sd))
rownames(mserecord_mean)[4] = 'OPT'
rownames(mserecord_sd)[4] = 'OPT'

mserecord_mean = mserecord_mean[2:5,]
colnames(mserecord_mean) = llist
mserecord_mean

################################################################################
# we then study the performance of different noise ratio                       #
################################################################################


n = 2500
sigmalist = c(0.5,1,2,4,8,16)
sigma_z = 0.1

l = 3

#repeat_trial = 10

mserecord_list = list()
biasrecord_list = list()

cl=makeCluster(2)
registerDoParallel(cl)
mserecord_list = foreach (trial = 1:repeat_trial, 
                          .packages=c('mlr3','mlr3learners','ranger','FRK','sp')) %dopar% {
  
  mserecord = as.data.frame(matrix(0,5,length(sigmalist)))
  biasrecord = as.data.frame(matrix(0,5,length(sigmalist)))
  colnames(mserecord) = paste0('sigma=',sigmalist)
  colnames(biasrecord) = paste0('sigma=',sigmalist)
  rownames(mserecord) = c('IR-lm','GLS','IR','opt','RF')
  rownames(biasrecord) = c('IR-lm','GLS','IR','opt','RF')
  
  print(trial)
  zdf <- data.frame(x = runif(n), y= runif(n))               # Generate random locs
  coordinates(zdf) = ~x+y                                    # Turn into sp object
  
  datadf_mean = matrix(NA,n,p+1+q1+q2)
  datadf_mean[,(p+1)] = 1
  datadf_mean[,(p+2):(p+1+q1)] = matrix(rnorm(n*q1), nrow = n)
  datadf_mean[,(p+q1+2):(p+1+q1+q2)] = matrix(rnorm(n*q2), nrow = n)
  coeff = matrix(rnorm((1+q1+q2)*p),
                 nrow = 1+q1+q2)
  
  G = auto_basis(manifold = plane(), data = zdf, regular = 1, 
                 nres = 1, tunit = "hours", type = 'bisquare')
  # show_basis(G)
  Psimat = eval_basis(G, zdf@coords)
  Psimat = as.matrix(Psimat)
  
  # plotcoord = zdf@coords
  # plotcoord = cbind(plotcoord, Psimat)
  # plotcoord = as.data.frame(plotcoord)
  # ggplot(plotcoord, aes(x = x, y = y)) + geom_point(aes(color = V3))
  
  A_raw = matrix(rnorm(k*p), nrow = k)
  
  datadf_mean = cbind(datadf_mean, Psimat)
  coeff = rbind(coeff, A_raw)
  
  obs_num = n
  ss <- sample(factor(rep(1:10, length.out=obs_num,)))
  table(ss)
  
  lmax=l
  
  nl = n*lmax
  
  noise_z_full = matrix(rnorm(nl*q2,mean = 0, sd = sigma_z), nrow = nl) 
  datadf_full = datadf_mean[rep(seq_len(nrow(datadf_mean)), each = lmax), ]
  datadf_full[,(p+q1+2):(p+1+q1+q2)] = datadf_full[,(p+q1+2):(p+1+q1+q2)]+noise_z_full
  
  datadf_list = list()
  datadf_bar_list = list()
  for (i in 1:length(sigmalist)){
    sigma = sqrt(sigmalist[i])
    noise_full = matrix(rnorm(nl*p,mean = 0, sd = sigma), nrow = nl) 
    datadf_full[,1:p] = datadf_full[,(p+1):(p+1+q1+q2+k)]%*%coeff + noise_full
    datadf = datadf_full[seq(1,nrow(datadf_full),lmax),]
    datadf_list[[i]] = datadf
    
    datadf_bar = aggregate(datadf_full, list(rep(1:(n + 1), each = l, len = n*l)), mean)[-1]
    datadf_bar_list[[i]] = datadf_bar
  }
  
  MSEwronginverse = rep(0,length(sigmalist))
  biaswronginverse = rep(0,length(sigmalist))
  
  MSEGLS = rep(0,length(sigmalist))
  biasGLS = rep(0,length(sigmalist))
  
  MSEtrueinverse = rep(0,length(sigmalist))
  biastrueinverse = rep(0,length(sigmalist))
  
  MSEbestinverse = rep(0,length(sigmalist))
  biasbestinverse = rep(0,length(sigmalist))
  
  MSERF = rep(0,length(sigmalist))
  biasRF = rep(0,length(sigmalist))
  
  for (i in 1:length(sigmalist)){
    print(i)
    datadf = as.data.frame(datadf_list[[i]])
    colnames(datadf)[p+1]='intercept'
    yname = colnames(datadf)[1:p]
    xname = colnames(datadf)[(p+2):(p+1+q1)]
    zname = colnames(datadf)[(p+q1+2):(p+1+q1+q2)]
    
    for (val_index in c(1:10)){
      train_index = which(ss!=val_index)
      test_index = which(ss==val_index)
      datatrain = datadf[train_index,]
      # datatest = datadf[test_index,]
      # datatest_bar = datadf_bar[test_index,]
      meanadj = colMeans(datatrain[(p+2):(p+1+q1+q2)])
      stdadj = sqrt(diag(var(datatrain[(p+2):(p+1+q1+q2)])))
      datatrain[(p+2):(p+1+q1+q2)] = sweep(sweep(datatrain[(p+2):(p+1+q1+q2)],2,meanadj),2,stdadj,"/")
      # datatest[,c(1:p,(p+q1+1):(p+q1+q2))] = sweep(sweep(datatest[,c(1:p,(p+q1+1):(p+q1+q2))],2,
      #                                                    meanadj[c(1:p,(p+q1+1):(p+q1+q2))]),2,stdadj[c(1:p,(p+q1+1):(p+q1+q2))],"/")
      # datatest_bar[,c(1:p,(p+q1+1):(p+q1+q2))] = sweep(sweep(datatest_bar[,c(1:p,(p+q1+1):(p+q1+q2))],2,
      #                                                        meanadj[c(1:p,(p+q1+1):(p+q1+q2))]),2,stdadj[c(1:p,(p+q1+1):(p+q1+q2))],"/")
      Psitrain = Psimat[train_index,]
      Psitest = Psimat[test_index,]
      
      
      X = as.matrix(datatrain[,xname])
      Z = as.matrix(datatrain[,zname])
      Y = as.matrix(datatrain[,yname])
      Psitrain_tPsitrain = t(Psitrain)%*%Psitrain
      
      ntrain = dim(Y)[1]
      onematrix = matrix(1,nrow = ntrain, ncol = 1)
      
      mathcal_X = cbind(onematrix,X,Z)
      
      Gamma_1 = diag(rep(1,p))
      Gamma_2 = diag(rep(1,p))
      
      mathcal_B = matrix(0, nrow = 1+q1+q2, ncol = p)
      
      EMiter = 1
      loglikelihood_old = Inf
      deltaloglikelihood = Inf
      
      while(EMiter<=50&deltaloglikelihood>0.1){
        #print(EMiter)
        
        Gamma_2inv = solve(Gamma_2)
        Gamma_1Gamma_2inv = Gamma_1%*%Gamma_2inv
        midinvmat = solve(diag(rep(1,k*p))+kronecker(Psitrain_tPsitrain,Gamma_1Gamma_2inv))
        
        
        midmathcal_B = solve(kronecker(t(mathcal_X)%*%mathcal_X,Gamma_2inv)-
                            kronecker(t(mathcal_X)%*%Psitrain,Gamma_2inv)%*%midinvmat%*%
                            kronecker(t(Psitrain)%*%mathcal_X,Gamma_1Gamma_2inv))
        mathcal_B = midmathcal_B%*%(kronecker(t(mathcal_X),Gamma_2inv)%*%as.matrix(c(t(Y)))-
                                   kronecker(t(mathcal_X)%*%Psitrain,Gamma_2inv)%*%midinvmat%*%
                                   kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Y))))
        mathcal_B = t(matrix(c(mathcal_B),nrow = p))
        
        Yres = Y-mathcal_X%*%mathcal_B
        
        A = kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Yres)))
        A = A - kronecker(Psitrain_tPsitrain,Gamma_1Gamma_2inv)%*%
          midinvmat%*%kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Yres)))
        A = t(matrix(c(A),nrow = p))
        
        midGamma_1 = kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv%*%Gamma_1)-
          kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
          midinvmat%*%kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv%*%Gamma_1)
        midGamma_1_ = kronecker(t(Psitrain)%*%mathcal_X, Gamma_1Gamma_2inv)-
          kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
          midinvmat%*%kronecker(t(Psitrain)%*%mathcal_X, Gamma_1Gamma_2inv)
        midGamma_1 = midGamma_1-midGamma_1_%*%midmathcal_B%*%t(midGamma_1_)
        
        midGamma_1_new = matrix(0,nrow = p,ncol = p)
        for (j in c(1:k)){
          midGamma_1_new = midGamma_1_new + midGamma_1[(j*p-p+1):(j*p),(j*p-p+1):(j*p)]/k
        }
        
        Gamma_1 = Gamma_1+1/k*t(A)%*%A-midGamma_1_new
        
        Yhat = mathcal_X%*%mathcal_B+Psitrain%*%A
        Gamma_2 = 1/ntrain*t(Y-Yhat)%*%(Y-Yhat)
        
        EMiter = EMiter+1
        #rm(Mz,Mx)
        #gc()
        loglikelihood = ntrain*log(det(Gamma_2)) + k*log(det(Gamma_1)) + 
          sum(diag(A%*%solve(Gamma_1)%*%t(A))) + 
          sum(diag((Yres-Psitrain%*%A)%*%solve(Gamma_2)%*%t(Yres-Psitrain%*%A)))
        #print(loglikelihood)
        deltaloglikelihood = loglikelihood_old - loglikelihood
        loglikelihood_old = loglikelihood
        #print(deltaloglikelihood)
      }
      
      alpha_true = as.matrix(mathcal_B[1,])
      B_true = mathcal_B[2:(1+q1),]
      D_true = mathcal_B[(2+q1):(1+q1+q2),]
      A_true = A
      Gamma_1_true = Gamma_1
      Gamma_2_true = Gamma_2
      
      #reversed modeling
      
      mathcal_X = cbind(onematrix,Y,Z)
      
      Gamma_1 = diag(rep(1,q1))
      Gamma_2 = diag(rep(1,q1))
      
      mathcal_B = matrix(0, nrow = 1+p+q2, ncol = q1)
      
      EMiter = 1
      loglikelihood_old = Inf
      deltaloglikelihood = Inf
      
      while(EMiter<=50&deltaloglikelihood>0.1){
        #print(EMiter)
        
        Gamma_2inv = solve(Gamma_2)
        Gamma_1Gamma_2inv = Gamma_1%*%Gamma_2inv
        midinvmat = solve(diag(rep(1,k*q1))+kronecker(t(Psitrain)%*%Psitrain,Gamma_1Gamma_2inv))
        
        
        midmathcal_B = solve(kronecker(t(mathcal_X)%*%mathcal_X,Gamma_2inv)-
                            kronecker(t(mathcal_X)%*%Psitrain,Gamma_2inv)%*%midinvmat%*%
                            kronecker(t(Psitrain)%*%mathcal_X,Gamma_1Gamma_2inv))
        mathcal_B = midmathcal_B%*%(kronecker(t(mathcal_X),Gamma_2inv)%*%as.matrix(c(t(X)))-
                                   kronecker(t(mathcal_X)%*%Psitrain,Gamma_2inv)%*%midinvmat%*%
                                   kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(X))))
        mathcal_B = t(matrix(c(mathcal_B),nrow = q1))
        
        Xres = X-mathcal_X%*%mathcal_B
        
        A = kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Xres)))
        A = A - kronecker(t(Psitrain)%*%Psitrain,Gamma_1Gamma_2inv)%*%
          midinvmat%*%kronecker(t(Psitrain),Gamma_1Gamma_2inv)%*%as.matrix(c(t(Xres)))
        A = t(matrix(c(A),nrow = q1))
        
        midGamma_1 = kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv%*%Gamma_1)-
          kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
          midinvmat%*%kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv%*%Gamma_1)
        midGamma_1_ = kronecker(t(Psitrain)%*%mathcal_X, Gamma_1Gamma_2inv)-
          kronecker(Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
          midinvmat%*%kronecker(t(Psitrain)%*%mathcal_X, Gamma_1Gamma_2inv)
        midGamma_1 = midGamma_1-midGamma_1_%*%midmathcal_B%*%t(midGamma_1_)
        
        midGamma_1_new = matrix(0,nrow = q1,ncol = q1)
        for (j in c(1:k)){
          midGamma_1_new = midGamma_1_new + midGamma_1[(j*q1-q1+1):(j*q1),(j*q1-q1+1):(j*q1)]/k
        }
        
        Gamma_1 = Gamma_1+1/k*t(A)%*%A-midGamma_1_new
        
        Xhat = mathcal_X%*%mathcal_B+Psitrain%*%A
        Gamma_2 = 1/ntrain*t(X-Xhat)%*%(X-Xhat)+diag(diag(rep(1e-3,q1)))
        
        EMiter = EMiter+1
        #rm(Mz,Mx)
        #gc()
        loglikelihood = ntrain*log(det(Gamma_2)) + k*log(det(Gamma_1)) + 
          sum(diag(A%*%solve(Gamma_1)%*%t(A))) + 
          sum(diag((Xres-Psitrain%*%A)%*%solve(Gamma_2)%*%t(Xres-Psitrain%*%A)))
        #print(loglikelihood)
        deltaloglikelihood = loglikelihood_old - loglikelihood
        loglikelihood_old = loglikelihood
        #print(deltaloglikelihood)
      }
      
      alpha_wrong = as.matrix(mathcal_B[1,])
      B_wrong = mathcal_B[2:(1+p),]
      D_wrong = mathcal_B[(2+p):(1+p+q2),]
      A_wrong = A
      Gamma_1_wrong = Gamma_1
      Gamma_2_wrong = Gamma_2
      
      learner_list = list()
      for (j in c(1:length(xname))){
        xname_i = xname[j]
        task = as_task_regr(datatrain[,c(yname, xname_i, zname)],target = xname_i)
        learner = lrn("regr.ranger")
        learner$train(task)
        learner_list[[j]] = learner
      }
      
      datatest = datadf_bar_list[[i]][test_index,]
      datatest[,c((p+q1+2):(p+1+q1+q2))] = sweep(sweep(datatest[,c((p+q1+2):(p+1+q1+q2))],2,
                                                       meanadj[c((q1+1):(q1+q2))]),2,stdadj[c((q1+1):(q1+q2))],"/")
      
      Zprime = as.matrix(datatest[,(p+q1+2):(p+1+q1+q2)])
      Yprime = as.matrix(datatest[,c(1:p)])
      Gamma_1Gamma_2inv = Gamma_1_true%*%solve(Gamma_2_true)
      midinvmat = solve(diag(rep(1,k*p))+kronecker(Psitrain_tPsitrain,Gamma_1Gamma_2inv))
      XTX_inv = solve(t(X)%*%X)*n
      
      xhat_GLS = matrix(NA,nrow = nrow(datatest), ncol = q1)
      xhat_OPT = matrix(NA,nrow = nrow(datatest), ncol = q1)
      for (j in c(1:nrow(datatest))){
        psi_test_j = as.matrix(Psitest[j,])
        S_j = kronecker(t(psi_test_j)%*%Psitrain_tPsitrain%*%psi_test_j, Gamma_1Gamma_2inv%*%Gamma_1_true)-
          kronecker(t(psi_test_j)%*%Psitrain_tPsitrain, Gamma_1Gamma_2inv)%*%
          midinvmat%*%kronecker(Psitrain_tPsitrain%*%psi_test_j, Gamma_1Gamma_2inv%*%Gamma_1_true)
        S_j = Gamma_2_true/l+kronecker(t(psi_test_j)%*%psi_test_j,Gamma_1_true)-S_j
        S_j = solve(S_j)
        Yres_j = Yprime[j,]-alpha_true-t(D_true)%*%as.matrix(Zprime[j,])-t(A_true)%*%psi_test_j
        xhat_j = solve(B_true%*%S_j%*%t(B_true))%*%B_true%*%S_j%*%as.matrix(Yres_j)
        xhat_GLS[j,] = t(xhat_j)
        xhat_j = solve(B_true%*%S_j%*%t(B_true)+XTX_inv)%*%B_true%*%S_j%*%as.matrix(Yres_j)
        xhat_OPT[j,] = t(xhat_j)
      }
      
      xhat = t(t(xhat_GLS)*stdadj[(1):(q1)]+meanadj[(1):(q1)])
      MSEGLS[i] = MSEGLS[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat)^2)
      biasGLS[i] = biasGLS[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat))
      
      xhat = matrix(1,nrow = length(test_index), ncol = 1)%*%t(alpha_wrong) + 
        Yprime%*%B_wrong + Zprime%*%D_wrong + Psitest%*%A
      xhat = t(t(xhat)*stdadj[(1):(q1)]+meanadj[(1):(q1)])
      MSEtrueinverse[i] = MSEtrueinverse[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat)^2)
      biastrueinverse[i] = biastrueinverse[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat))
      
      xhat = t(t(xhat_OPT)*stdadj[(1):(q1)]+meanadj[(1):(q1)])
      MSEbestinverse[i] = MSEbestinverse[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat)^2)
      biasbestinverse[i] = biasbestinverse[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat))
      
      for (j in c(1:length(xname))){
        xhat[,j] = learner_list[[j]]$predict_newdata(datatest[,c(yname, zname)])$response
      }
      xhat = t(t(xhat)*stdadj[(1):(q1)]+meanadj[(1):(q1)])
      MSERF[i] = MSERF[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat)^2)
      biasRF[i] = biasRF[i]+mean(as.matrix(datatest[,c((p+2):(p+1+q1))]-xhat))
    }
  }
  
  mserecord[2,] = mserecord[2,] + MSEGLS/10
  biasrecord[2,] = biasrecord[2,] + biasGLS/10
  
  mserecord[3,] = mserecord[3,] + MSEtrueinverse/10
  biasrecord[3,] = biasrecord[3,] + biastrueinverse/10
  
  mserecord[4,] = mserecord[4,] + MSEbestinverse/10
  biasrecord[4,] = biasrecord[4,] + biasbestinverse/10
  
  mserecord[5,] = mserecord[5,] + MSERF/10
  biasrecord[5,] = biasrecord[5,] + biasRF/10
  
  mserecord
}
stopCluster(cl)
stopImplicitCluster()

mserecord_final = do.call(rbind, mserecord_list)
#biasrecord_final = do.call(rbind, biasrecord_list)

#save(mserecord_list,file='STmserecord_list_sigma.RData')
#write.csv(biasrecord_final,'bias_sigma.csv')
#write.csv(mserecord_final,'STmse_sigma.csv')

#load('STmserecord_list_sigma.RData')

mserecord_mean = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), mean))
mserecord_sd = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), sd))

mserecord_90quantile = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), 
                                        c(1,2), function(x){quantile(x,0.90)}))
mserecord_10quantile = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), 
                                        c(1,2), function(x){quantile(x,0.10)}))
mserecord_sd = data.frame(apply(sapply(mserecord_list, as.matrix, simplify="array"), c(1,2), sd))
rownames(mserecord_mean)[4] = 'OPT'
rownames(mserecord_sd)[4] = 'OPT'

mserecord_mean = mserecord_mean[2:5,]
colnames(mserecord_mean) = sigmalist
mserecord_mean
