import scipy as sp 
from scipy import stats
import numpy as np
import multiprocessing as mp
import h5py
import random
import math
import sys
import time

## Load data:

"""
with h5py.File("New_try.h5","r") as hf:

	data = hf.get("Chromosomes")
	#data = hf["Chromosomes"]
	Chromosomes = sp.array(data)
	print("Shape of the array Chromosomes:", Chromosomes.shape)
	#print(Chromosomes[...])

	Affections = hf["sample_informations"]["Affections"]
	Y_unnorm = sp.array(Affections)
	print("Shape of the array phenotypes:", Y_unnorm.shape)
	#print(Y_unnorm[...])
	print("Number of cases:", sum(Y_unnorm))
	print("Number of controls:", Y_unnorm.shape[0]- sum(Y_unnorm))

	X_unnorm = sp.array(hf["chr_1"]["snps"])
	for chrom in range(2,26):
		snps = sp.array(hf["chr_%d" % chrom]["snps"])
		X_unnorm = sp.concatenate((X_unnorm,snps),axis=0)

		
	X_unnorm = sp.array(X_unnorm)
	print("Shape of the array genotypes:", X_unnorm.shape)


## Normalize:
print("Normalizing...")
(M,N) = X_unnorm.shape
X_means = sp.mean(X_unnorm,axis=0)
X_std = sp.std(X_unnorm,axis=0)
X_means = X_means.T
X = (X_unnorm-X_means)/X_std
# X=sp.apply_along_axis(lambda x: (x-sp.mean(x))/sp.std(x),0,X_unnorm)

X = X.T

print("Sum of means:",sum(sp.mean(X,axis=1)))
print("Sum of variances:",sum(sp.std(X,axis=1)))

Y = (Y_unnorm-sp.mean(Y_unnorm))/sp.std(Y_unnorm)
#print(sp.mean(Y))
#print(sp.std(Y))

(N,M) = X.shape
#Y.shape=(N,)
#print("Y.shape:",Y.shape)

## Saving the normalized genotype matrix and the 
## normalized phenotype vector in a HDF5 file:
#with h5py.File("Normalized_data.h5","w") as hf:
#	hf.create_dataset('X', data=X)
#	hf.create_dataset('Y', data=Y)
#	hf.create_dataset('Chromosomes', data=Chromosomes)

"""
start_time = time.time()
print("Importing the data...")

with h5py.File("Normalized_data.h5","r") as hf:
	data = hf.get('X')
	X= sp.array(data, dtype= "single")
	print(type(X) is single)
	data = hf.get('Y')
	Y= sp.array(data, dtype= "single")

with h5py.File("New_try.h5","r") as hf:
	data = hf["Chromosomes"]
	Chromosomes = sp.array(data, dtype= "single")

print("Execution time (importing):", round(time.time()-start_time,2),"seconds")

(N,M) = X.shape
print("Shape of the array genotypes:", X.shape)
print("Shape of the array phenotypes:", Y.shape)
print("Shape of the array Chromosomes:", Chromosomes.shape)

## Step 1a : Estimate variance parameters

##---------------------------------------------------------
## Conjugate gradient iteration 
##---------------------------------------------------------
## Name: conjugateGradientSolve
## Purpose: Solving an equation of the form
##              Ax=b
##          by the aid of conjugate gradient iteration
## Input: 
## A = nxn matrix
## b = n-vector
## Output:
## x
##---------------------------------------------------------

def conjugateGradientSolve(A,x0,b):
	x = x0
	r = b-sp.dot(A,x)
	p=r
	rsold = sp.dot(r,r)
	norm = sp.sqrt(rsold)
	while norm>0.0005:
		Ap = sp.dot(A,p)
		## alpha = step size
    	## alpha = t(r_{k-1})*r_{k-1}/(t(p_k)*A*p_k)
		alpha = rsold/(sp.dot(p,Ap))
		## x_k = x_{k-1} + alpha*p_k
		x = x+alpha*p
		## r_k = r_{k-1}-alpha*A*p_k
		r = r-alpha*Ap
		rsnew = sp.dot(r,r)
		## beta = t(r_{k-1})*r_{k-1}/(t(r_{k-2})*r_{k-2})
		## p = search direction
		## p_k = r_{k-1} + beta*p_{k-1}
		p = r+(rsnew/rsold)*p
		norm = sp.sqrt(rsnew)
		rsold = rsnew
	return(x)


##---------------------------------------------------------
## Conjugate gradient iteration ()
##---------------------------------------------------------
## Name: FASTconjugateGradientSolve
## Purpose: Solving an equation of the form
##              Ax=b
## 			(where A is of the form XX'/M + d*I)
##          by the aid of conjugate gradient iteration.
##			Due to the composition of A, we have that
## 			Ax= (XX'/M + d*I)x= (1/M)*XX'x + d*x.
##			This expression can be calculated 
##        	much more efficient as the calculation
## 			needs less CPU time and memory.
## Input: 
## X = NxM matrix
## b = N-vector
## x0 = M-vector, the initial value of x
## c1,c2 = a real scalar
## Output:
## x
##---------------------------------------------------------

def FASTconjugateGradientSolve(X,x0,b,c1=1,c2=1):
	M= X.shape[1]
	x = x0
	r = b-(sp.dot(X,sp.dot(X.T,x))*float(c1)/float(M) + float(c2)*x)
	p=r
	rsold = sp.dot(r,r)
	norm = sp.sqrt(rsold)
	while norm>0.0005:
		Ap = (sp.dot(X,sp.dot(X.T,p))*float(c1)/float(M) + float(c2)*p)
		## alpha = step size
    	## alpha = t(r_{k-1})*r_{k-1}/(t(p_k)*A*p_k)
		alpha = rsold/(sp.dot(p,Ap))
		## x_k = x_{k-1} + alpha*p_k
		x = x+alpha*p
		## r_k = r_{k-1}-alpha*A*p_k
		r = r-alpha*Ap
		rsnew = sp.dot(r,r)
		## beta = t(r_{k-1})*r_{k-1}/(t(r_{k-2})*r_{k-2})
		## p = search direction
		## p_k = r_{k-1} + beta*p_{k-1}
		p = r+(rsnew/rsold)*p
		norm = sp.sqrt(rsnew)
		rsold = rsnew
	return(x)



##---------------------------------------------------------
## Compute f_{REML}(log(delta)) 
##---------------------------------------------------------
## Name: evalfREML
## Purpose: Compute f_{REML}(log(delta)), where
##          f_{REML}(log(delta)) = log((sum(beta.hat_data^2)/
##          sum(e.hat_data^2))/(E[sum(hat.beta^2)]/E[sum(
##              e.hat_data^2)]))
## Input: 
## logDelta = log(delta) = log(sigma.e^2/sigma.g^2) 
## MCtrials = Number of Monte Carlo simulations
## X = NxM matrix - normalized genotypes
## Y = phenotype vector, N-vector
## beta_rand = random SNP effects
## e_rand_unscaled = random environmental effects
## Output:
## The evaluated function, f_{REML}
##---------------------------------------------------------

def evalfREML(logDelta,MCtrials,X,Y,beta_rand,e_rand_unscaled):

	(N,M) = X.shape
	delta = sp.exp(logDelta, dtype= "single")
	y_rand = sp.empty((N,MCtrials), dtype= "single")
	H_inv_y_rand = sp.empty((N,MCtrials), dtype= "single")
	beta_hat_rand = sp.empty((M,MCtrials), dtype= "single")
	e_hat_rand = sp.empty((N,MCtrials), dtype= "single")

	## Defining the initial vector x0
	x0 = sp.zeros(N, dtype= "single")
	for t in range(0,MCtrials):
		## build random phenotypes using pre-generated components
		y_rand[:,t] = sp.dot(X,beta_rand[:,t])+sp.sqrt(delta)*e_rand_unscaled[:,t]
		## compute H^(-1)%*%y.rand[,t] by the aid of conjugate gradient iteration
		H_inv_y_rand[:,t] = FASTconjugateGradientSolve(X=X,x0=x0,b=y_rand[:,t],c2=delta)
		## compute BLUP estimated SNP effect sizes and residuals
		beta_hat_rand[:,t] = sp.dot(X.T,H_inv_y_rand[:,t])
		e_hat_rand[:,t] = H_inv_y_rand[:,t]
		print("In evalfREML: Iteration %d has been completed..." % t)

	## compute BLUP estimated SNP effect sizes and residuals for real phenotypes
	e_hat_data = FASTconjugateGradientSolve(X=X,x0=x0,b=Y,c2=delta)
	beta_hat_data = sp.dot(X.T,e_hat_data )
	
	## evaluate f_REML
	f = sp.log((sp.sum(beta_hat_data**2)/sp.sum(e_hat_data**2))/(sp.sum(beta_hat_rand**2)/sp.sum(e_hat_rand**2)))
	return(f)


print("Step 1a : Estimate variance parameters...")
step = time.time()

## Set the number of Monte Carlo trials
MCtrials = max(min(4e9/(N**2),15),3)
print("The number of MC trials is:", MCtrials)

## generate random SNP effects
beta_rand = stats.norm.rvs(0,1,size=(M,MCtrials))*sp.sqrt(1.0/float(M))
beta_rand.astype(dtype="single")
## generate random environmental effects
e_rand_unscaled = stats.norm.rvs(0,1,size=(N,MCtrials))
e_rand_unscaled.astype(dtype="single")

h12 = 0.25
logDelta = [sp.log((1-h12)/h12)]

## Perform first fREML evaluation
print("Performing the first fREML evaluation...")
start_time = time.time()
f = [evalfREML(logDelta=logDelta[0],MCtrials=MCtrials,X=X,Y=Y,beta_rand=beta_rand,e_rand_unscaled=e_rand_unscaled)]
print("Execution time (first fREML):", round(time.time()-start_time,2),"seconds")

if f[0]<0:
	h22=0.125
else:
	h22=0.5

logDelta.append(sp.log((1-h22)/h22))

## Perform second fREML evaluation
print("Performing the second fREML evaluation...")
start_time = time.time()
f.append(evalfREML(logDelta=logDelta[1],MCtrials=MCtrials,X=X,Y=Y,beta_rand=beta_rand,e_rand_unscaled=e_rand_unscaled))
print("Execution time (second fREML):", round(time.time()-start_time,2),"seconds")

## Perform up to 5 steps of secant iteration
print("Performing up to 5 steps of secant iteration...")
for s in range(2,7):
	logDelta.append((logDelta[s-2]*f[s-1]-logDelta[s-1]*f[s-2])/(f[s-1]-f[s-2]))
	## check convergence
	if abs(logDelta[s]-logDelta[s-1])<0.01:
		break
	f.append(evalfREML(logDelta=logDelta[s],MCtrials=MCtrials,X=X,Y=Y,beta_rand=beta_rand,e_rand_unscaled=e_rand_unscaled))
	print("Iteration %d has been completed successfully." % (s-1))

delta = sp.exp(logDelta[-1])
print("The final delta:",delta) # 0.17609251449105767

x0 = sp.zeros(N, dtype= "single")
H_inv_y_data = FASTconjugateGradientSolve(X=X,x0=x0,b=Y,c2=delta)

sigma_g = sp.dot(Y,H_inv_y_data)/float(N)
sigma_e = delta*sigma_g
print("sigma.g=",sigma_g) # 0.80200006131763968
print("sigma.e=",sigma_e) # 0.14122620741940561

print("Step 1a took", round((time.time()-step)/60,2),"minutes")

"""
sigma_g = 0.80200006131763968 #0.80366656315376572
sigma_e = 0.14122620741940561 # 0.14008623371097639
print("sigma.g=",sigma_g) # 0.80200006131763968
print("sigma.e=",sigma_e) # 0.14122620741940561

x0 = sp.zeros(N)
"""
## Step 1b : Compute and calibrate BOLT-LMM-inf statistics
print("Step 1b : Compute and calibrate BOLT-LMM-inf statistics...")
step = time.time()

## precompute V_{-chr}^{-1}*Y, where 
## V_{-chr}= sigma.g^2*X_{-chr}*t(X_{-chr})/M_{-chr} + sigma.e^2*I_N

V_chr_inv_Y = sp.empty((N,25))

for chrom in range(1,26):
	chrom_index = sp.array(Chromosomes != chrom)
	X_chr = X[:,chrom_index]
	V_chr_inv_Y[:,chrom-1] = FASTconjugateGradientSolve(X=X_chr,x0=x0,b=Y,c1=sigma_g,c2=sigma_e)

## Compute calibration for BOL-LMM-inf statistic using 30 random SNPs

prospectiveStat = sp.zeros(30)
uncalibratedRetrospectiveStat = sp.zeros(30)

for t in range(0,30):

	## random SNP in {1...M}
	m= random.randint(0,M)
	## normalized genotype vector for chosen SNP m
	x = X[:,m]
	## chromosome containing chosen SNP m
	chrom = Chromosomes[m]
	chrom_index = sp.array(Chromosomes != chrom)
	## X_{-chr}
	X_chr = X[:,chrom_index]
	V_chr_inv_x = FASTconjugateGradientSolve(X=X_chr,x0=x0,b=x,c1=sigma_g,c2=sigma_e)

	prospectiveStat[t] = sp.dot(x,V_chr_inv_Y[:,chrom-1])**2/sp.dot(x,V_chr_inv_x)
	uncalibratedRetrospectiveStat[t] = float(N)*sp.dot(x,V_chr_inv_Y[:,chrom-1])**2/(sp.sum(x**2)*sp.sum(V_chr_inv_Y[:,chrom-1]**2))

infStatCalibration = sp.sum(uncalibratedRetrospectiveStat)/sp.sum(prospectiveStat)
print("infStatCalibration:", infStatCalibration)

## Compute BOLT-LMM-inf mixed model statistics at all SNPs

boltLMMinf = sp.zeros(M)

for m in range(0,M):
	x = X[:,m]
	## Chromosome containing SNP m
	chrom = Chromosomes[m]
	boltLMMinf[m] = float(N)*sp.dot(x,V_chr_inv_Y[:,chrom-1])**2/(sp.sum(x**2)*sp.sum(V_chr_inv_Y[:,chrom-1]**2))/infStatCalibration

print("Step 1b took", round((time.time()-step)/60,2),"minutes")

## Step 2a: Estimate Gaussian mixture prior parameters

##---------------------------------------------------------
## Variational Bayes
##---------------------------------------------------------
## Name: fitVariationalBayes
## Purpose: Fitting Variational Bayes to estimate the 
##          effect sizes. Each estimated SNP effect is
##          set to its conditional posterior mean.
## Input:
## X = NxM matrix - normalized genotypes
## Y = phenotype vector, N-vector
## sigma.g, sigma.e = variance parameters
## p = mixture probability, 
##     beta_m ~ N(0,sigma_{beta,2}^2) with prob (1-p)
##     and
##     beta_m ~ N(0,sigma_{beta,1}^2) with prob p
## f2 = the proportion of the total mixture variance within 
##      the second Gaussian (the ""spike component)
##      f2 = (1-p)sigma_{beta,2}^2/(p*sigma_{beta,1}^2 +
##           ((1-p)sigma_{beta,2}^2)
## maxIters = the function performs VB until convergence 
##            or maxIters
## Output:
## The estimated effect sizes
##---------------------------------------------------------

def fitVariationalBayes(X,Y,sigma_g,sigma_e,f2,p,maxIters=250):
	(N,M) = X.shape
	## Set Gaussian variances
	sigma_beta = [sigma_g/float(M) *(1-f2)/p]
	sigma_beta.append(sigma_g/float(M)*f2/(1-p))
	## Initialize SNP effect estimates to 0
	beta_fit = sp.zeros(M)
	## Initialize residual phenotype to Y
	y_resid = Y
	## Initialize approximate log likelihood to infinity
	approxLL = float('inf')
	## Perform VB iterations until convergence or maxIters
	for k in range(maxIters):
		approxLLprev = approxLL
		approxLL = -float(N)/2.0*math.log(2*math.pi*sigma_e)
		## Update SNP effect estimates in turn and accumulate 
		## contributions to approxLL
		for m in range(M):
			## Extract SNP m
			x = X[:,m]
			## Remove effect of SNP m from residual
			y_resid += beta_fit[m]*x
			## Formulas in Section 2.1.4
			beta_hat = sp.dot(x,y_resid)/sp.sum(x**2)

			beta_bar = [ s*beta_hat/(s+(sigma_e/sp.sum(x**2))) for s in sigma_beta]
			tau = [s*sigma_e/sp.sum(x**2)/(s+(sigma_e/sp.sum(x**2))) for s in sigma_beta]
			s = [z+sigma_e/sp.sum(x**2) for z in sigma_beta]
			pm = (p/sp.sqrt(s[0]))*sp.exp(-beta_hat**2/(2*s[0]))/((p/sp.sqrt(s[0]))*sp.exp(-beta_hat**2/(2*s[0]))+((1-p)/sp.sqrt(s[1]))*sp.exp(-beta_hat**2/(2*s[1])))
			DKL = pm*sp.log(pm/p) + (1-pm)*sp.log((1-pm)/(1-p))-pm/2*(1+sp.log(tau[0]/sigma_beta[0])-(tau[0]+beta_bar[0]**2)/sigma_beta[0])- (1-pm)/2*(1+sp.log(tau[1]/sigma_beta[1])-(tau[1]+beta_bar[1]**2)/sigma_beta[1])

			var_beta = pm*(tau[0]+beta_bar[0]**2)+(1-pm)*(tau[1]+beta_bar[1]**2)-(pm*beta_bar[0])**2
			## Set effect size to conditional posterior mean
			beta_fit[m] = pm*beta_bar[0]+(1-pm)*beta_bar[1]
			## Update approxxLL (as in Section 2.1.4)
			approxLL -=(sp.sum(x**2)/(2*sigma_e)*var_beta + DKL)
			## Update residual with new effect of SNP m
			y_resid -= beta_fit[m]*x

		approxLL -= sp.sum(y_resid**2)/(2*sigma_e)

		## Test convergence
		if approxLL-approxLLprev < 0.01:
			break

	l = [beta_fit,y_resid]
	return(l)


print("Step 2a: Estimate Gaussian mixture prior parameters...")
step = time.time()

## Algorithm: Optimize prediction mean-squared error in cross-validation

f2_vec = [0.5,0.3,0.1]
p_vec = [0.5,0.2,0.1,0.05,0.02,0.01]
MSE = sp.empty((3,6))

for f2 in f2_vec:
	for p in p_vec:
		index = np.random.choice(N,size=N,replace=False)
		for CVfold in range(0,5):
			## Subset of samples {1...N} in fold CVfold
			foldInds = index[CVfold::5]
			notInds = list(set(index)-set(foldInds))
			X_train = X[notInds,:]
			Y_train = Y[notInds]
			X_test = X[foldInds,:]
			Y_test = Y[foldInds]

			## Fit VB on training sets
			ans = fitVariationalBayes(X=X_train,Y=Y_train,sigma_g=sigma_g,sigma_e=sigma_e,f2=f2,p=p)
			beta_fit = ans[0]
			y_resid = ans[1]

			## Evaluating the fit
			Y_pred = sp.dot(X_test,beta_fit)

			MSE[f2_vec.index(f2),p_vec.index(p)] += sp.sum((Y_pred-Y_test)**2)

### Change this indexing!!!!!!!!!!!!!!!
#min_MSE_index = sp.where(MSE == MSE.min())
for i in range(3):
	for j in range(6):
		if MSE[i,j]==MSE.min():
			min_index_0 = i
			min_index_1 = j

f2 = f2_vec[min_index_0]
p = p_vec[min_index_1]
print("f2:",f2)
print("p:",p)

print("Step 2a took", round((time.time()-step)/60,2),"minutes")