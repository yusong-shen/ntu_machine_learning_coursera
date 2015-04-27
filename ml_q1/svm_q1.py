# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 19:43:30 2015

@author: Yusong
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import KFold

# data format
# each line :  digit intensity symmetry
features_train = np.loadtxt('features_train.txt')
#print len(features_train[1])
#print features_train[1]
features_test = np.loadtxt('features_test.txt')

x_train = [[digit[1], digit[2]] for digit in features_train]
y_train = [digit[0] for digit in features_train]
#print x_train[1]
#print y_train[1]

x_test = [[digit[1], digit[2]] for digit in features_test]
y_test = [digit[0] for digit in features_test]

# question 15
# 0 vs. 'not 0'
# linear kernel, C = 0.01
def convert_labels(labels, num):
    """
    convert labels with n-categories to 2 categories
    e.g. 0 vs 'not 0'    
    """
    bin_labels = []
    for label in labels:
        if label == num:
            bin_labels.append(1.0)
        else:
            bin_labels.append(-1.0)
    return bin_labels

y_train_0 = convert_labels(y_train, 0.0)
y_test_0 = convert_labels(y_test, 0.0)    
lin_svc = svm.SVC(C=0.01, kernel='linear')
lin_svc.fit(x_train, y_train_0)
print lin_svc.score(x_train, y_train_0)
print lin_svc.coef_
dual_coef = lin_svc.dual_coef_

#print clf1.get_params()
#print clf1.decision_function(x_train)[:10]
print lin_svc.predict(x_test)
print lin_svc.score(x_test, y_test_0)

## create a mesh to plot in
#h = .02
#x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
#y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                     np.arange(y_min, y_max, h))
#                     
#Z = lin_svc.predict(np.c_[xx.ravel(), yy.ravel()])
#
## Put the result into a color plot
#Z = Z.reshape(xx.shape)
#plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#
## Plot also the training points
#plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test_0, cmap=plt.cm.Paired)
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
#plt.xlim(xx.min(), xx.max())
#plt.ylim(yy.min(), yy.max())
#plt.xticks(())
#plt.yticks(())                     
#
#plt.show()                     
                     
# question 16
# polynomial kernel, C = 0.01, Q = 2
#e_ins = []
#alphas = []
#for idx in range(5):
#    y = convert_labels(y_train, 2*idx)
#    poly_svc = svm.SVC(C=0.01, kernel='poly', degree=2).fit(x_train, y)
#    e_in = 1 - poly_svc.score(x_train, y)
#    alpha = sum(sum(np.absolute(poly_svc.dual_coef_)))
#    e_ins.append(e_in)
#    alphas.append(alpha)

# question 19 
# gaussian kernel C=0.1
gammas = [1, 10, 100, 1000, 10000]
#e_outs = []
#for gammai in gammas:
#    rbf_svc = svm.SVC(C=0.1, kernel='rbf', gamma=gammai).fit(x_train,y_train_0)
#    e_out = 1 - rbf_svc.score(x_test, y_test_0)
#    e_outs.append((gammai, e_out))

# question 20
# validation procedure
counter = {gammai:0 for gammai in gammas}
for idx in range(14):
    # generate indices     
    kf = KFold(len(x_train), n_folds = 7)
    for train, test in kf:
        x_tra = [x_train[idx] for idx in train]
        x_tes = [x_train[idx] for idx in test]
        y_tra = [y_train_0[idx] for idx in train]
        y_tes = [y_train_0[idx] for idx in test] 
        e_evals = []
        for gammai in gammas:
            rbf_svc = svm.SVC(C=0.1, kernel='rbf', gamma=gammai).fit(x_tra, y_tra)
            e_eval = 1 - rbf_svc.score(x_tes, y_tes)
            e_evals.append((e_eval, gammai))
        best = min(e_evals)
        counter[best[1]]+=1


                     
                     
                     