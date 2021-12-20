#
# Moindre carrées
#
# 

import numpy as np
import numpy.linalg as lin
from matplotlib import pyplot as plt

N = 15
x = np.linspace(0,5,N)
y = pow((x-2), 2) + np.random.randn(N, 1)
print(y)

k = 5 #Polynom d'ordre k

#Ajustement par les moindres carrées
A = []
for i in range(N):
    b=[]
    for j in range(k+1):
        b.insert(0, pow(x[i], j))
    A.append(b)
A = np.array(A)
what = lin.lstsq(A,y,rcond=None)[0]


#Test
xx=np.linspace(0,5,200)

AA = []
for i in range(200):
    b=[]
    for j in range(k+1):
        b.insert(0,pow(xx[i], j))
    AA.append(b)
AA = np.array(AA)

yy = np.dot(AA, what)
#Plot
plt.figure(1)
plt.scatter(x,y,c='b')
plt.plot(xx,yy)

plt.xlabel('x')
plt.ylabel('y')
plt.show

err = lin.norm((y - np.dot(A,what)))

#No cross validation
kmax = 5 

A = []
for i in range(N):
    b = []
    for j in range(k+1):
        b.insert(0, pow(x[i], j))
    A.append(b)
A = np.array(A)

err = np.zeros(kmax)

for i in range(kmax):
    Amat = np.array(A[:,kmax-i-1:])
    what = lin.lstsq(Amat,y,rcond=None)[0]
    err[i] = lin.norm((y - np.dot(Amat, what))) / N / N #compute the error and divide by the number of points


#Using cross validation
T = 12
trials = 1000

A = []

for i in range(N):
    b = []
    for j in range(k+1):
        b.insert(0, pow(x[i], j))
    A.append(b)
A = np.array(A)

errcv = np.zeros((kmax,trials))

for i in range(trials):
    r = np.random.permutation(N)
    train = r[: T]
    test = r[T :]
    for j in range(kmax):
        Atrain = A[train, kmax - j - 1:]
        Atest = A[test, kmax-j-1:]

        ytrain = y[train]
        ytest = y[test]

        what = lin.lstsq(Atrain, ytrain, rcond = None)[0]
        errcv[j][i] = lin.norm((ytest - np.dot(Atest, what))) / (N-T)

        avg_err_cv = np.mean(errcv, axis = 1)

        #Plotting all models
        models = ['Model 1', 'Model 2', "Model 3", "Model 4", "Model 5"]

        x_axis = np.arange(kmax)

        plt.figure(2)
        plt.bar(x_axis-0.2,err,0.4,label = 'No cross validation', color = 'r')
        plt.bar(x_axis+0.2,avg_err_cv,0.4,label = "Using cross validation", color = "b")
        plt.xlabel('Models')
        plt.ylabel('Errors')
        plt.xticks(x_axis,models)
        plt.show
        print(f"The best fit using cross validation is: {models[list(avg_err_cv).index(min(avg_err_cv))]}")


