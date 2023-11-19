import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    elements = np.unique(y)
    means = np.zeros(shape=(X.shape[1], len(elements)))

    for element in elements:
        items = X[y[:, 0] == element]
        means[:, int(element) - 1] = np.mean(items, axis=0)

    covmat = np.cov(X.transpose())
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    elements = np.unique(y)
    means = np.zeros(shape=(X.shape[1], len(elements)))
    covmat = np.zeros(shape=(X.shape[1], X.shape[1]))
    covmat_list = list()
    for element in elements:
        items = X[y[:, 0] == element]
        means[:, int(element) - 1] = np.mean(items, axis=0)
        covmat = np.cov(items.transpose())
        covmat_list.append(covmat)

    covmats = covmat_list
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    covt_invr = np.linalg.inv(covmat)
    items = np.shape(means)[1]
    final_pdf = np.zeros(shape=(Xtest.shape[0], items))
    final_result = 0

    for i in range(0, items):
        for j in range(0, Xtest.shape[0]):
            sub_mean_of_x = Xtest[j, :] - (means[:, i]).transpose()
            pdf_value = np.dot(np.dot(sub_mean_of_x, covt_invr), (sub_mean_of_x))
            final_pdf[j, i] = pdf_value

    Myclasses = np.zeros(shape=(Xtest.shape[0], 1))
    Myclasses = (np.argmin(final_pdf, axis=1)) + 1
    for i in range(0, Xtest.shape[0]):
        if (ytest[i] == Myclasses[i]):
            final_result = final_result + 1
        ypred = Myclasses.reshape(Xtest.shape[0], 1)

    acc = (final_result / len(ytest)) * 100
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    i = 0
    final_result = 0
    ypred = np.zeros(shape=(Xtest.shape[0]))
    final_pdf = np.zeros(means.shape[1])

    for elements in Xtest:
        for j in range(0, means.shape[1]):
            sub_mean_of_x = elements - means.transpose()[j]
            cov_inv = np.linalg.inv(covmats[j])
            denomtr = np.sqrt(np.linalg.det(covmats[j]))
            numertr = np.exp(-0.5 * np.dot(np.dot(sub_mean_of_x, cov_inv).transpose(), sub_mean_of_x))
            final_pdf[j] = numertr / denomtr
            pdf_max_value = np.argmax(final_pdf)
        ypred[i] = pdf_max_value + 1
        i += 1

    for i in range(0, len(Xtest)):
        if ypred[i] == ytest[i][0]:
            final_result = final_result + 1

    acc = (final_result / len(ytest)) * 100
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    print(type(X))
    w = np.dot(np.linalg.inv(np.dot(lambd, np.identity(X.shape[1])) + np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    x = Xtest.shape[0]
    mse = (1 / x) * np.dot(np.transpose(np.subtract(ytest, np.dot(Xtest, w))), np.subtract(ytest, np.dot(Xtest, w)))
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    w1 = w.shape[0]
    w = w.reshape(w1, 1)
    error = (1 / 2) * ((np.sum(np.square(y - np.dot(X, w)))) + lambd * np.dot(w.transpose(), w))

    # error gradient:
    error_grad = ((np.dot((np.dot(X.transpose(), X)), w)) - np.dot(X.transpose(), y)) + (lambd * w)

    error_grad = error_grad.flatten()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    N = x.shape[0]
    Xp = np.empty((N, p + 1))
    Xp[:] = 1
    for i in range(1, p + 1):
        Xp[:, i] = pow(x, i)
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

w = learnOLERegression(X,y)
mle = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,X_i,y)

print('train data: MSE without intercept '+str(mle))
print('train data: MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
l ,optimal = 0,4000
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    if (mses3[i] < optimal):
        optimal = mses3[i]
        l = lambd
    i = i + 1

print('Optimal lambda '+str(l))
print('Optimal error '+str(optimal))

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))

optimal_p_0 = 0
optimal_err_0 = 10000
optimal_p_1 = 0
optimal_err_1 = 10000

for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

    if (mses5[p, 0] < optimal_err_0):
        optimal_err_0 = mses5[p, 0]
        optimal_p_0 = p

    if (mses5[p, 1] < optimal_err_1):
        optimal_err_1 = mses5[p, 1]
        optimal_p_1 = p

print('Optimal p - No regularization '+ str(optimal_p_0))
print('Optimal p  - With regularization '+ str(optimal_p_1))

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
