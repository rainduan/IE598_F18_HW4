# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 14:52:50 2018

@author: duany
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

#df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
#                 'python-machine-learning-book-2nd-edition'
#                 '/master/code/ch10/housing.data.txt',
#                 header=None,
#                 sep='\s+')
os.getcwd()
#df = pd.read_excel('housing.xlsx')
df=pd.read_csv('concrete.csv')
#print(df)
#df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
#              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
#              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.shape
df.info()
df.head()
cols = ['cement', 'slag', 'ash', 'water', 'superplastic','coarseagg','fineagg','age','strength']

for i in ['cement', 'slag', 'ash', 'water', 'superplastic','coarseagg','fineagg','age','strength']:
    plt.figure()
    sns.boxplot(x=i,data=df)
    plt.savefig(i+'10_02.png', dpi=300)
#print(df)
#cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.savefig('10_03.png', dpi=300)
plt.show()


cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 8},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.savefig('10_04.png', dpi=300)
plt.show()

# # Implementing an ordinary least squares linear regression model

# ...

# ## Solving regression for regression parameters with gradient descent
#
#
#class LinearRegressionGD(object):
#
#    def __init__(self, eta=0.001, n_iter=20):
#        self.eta = eta
#        self.n_iter = n_iter
#
#    def fit(self, X, y):
#        self.w_ = np.zeros(1 + X.shape[1])
#        self.cost_ = []
#
#        for i in range(self.n_iter):
#            output = self.net_input(X)
#            errors = (y - output)
#            self.w_[1:] += self.eta * X.T.dot(errors)
#            self.w_[0] += self.eta * errors.sum()
#            cost = (errors**2).sum() / 2.0
#            self.cost_.append(cost)
#        return self
#
#    def net_input(self, X):
#        return np.dot(X, self.w_[1:]) + self.w_[0]
#
#    def predict(self, X):
#        return self.net_input(X)
#
#
X = df[['cement']].values
y = df['strength'].values
#

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#X_std = sc_x.fit_transform(X)
#y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
#
#lr = LinearRegressionGD()
#lr.fit(X_std, y_std)
#
##sns.reset_orig()
#plt.plot(range(1, lr.n_iter+1), lr.cost_)
#plt.ylabel('SSE')
#plt.xlabel('Epoch')
#plt.savefig('10_05.png', dpi=300)
#plt.show()
#
#
#
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=20)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 
#
#
#lin_regplot(X_std, y_std, lr)
#plt.xlabel('[cement] (standardized)')
#plt.ylabel('[strength] (standardized)')
#plt.tight_layout()
#plt.savefig('10_06.png', dpi=300)
#plt.show()
#
#
#
#print('Slope: %.3f' % lr.w_[1])
#print('Intercept: %.3f' % lr.w_[0])
#
#
#
#
#cement_std = sc_x.transform(np.array([[5.0]]))
#strength_std = lr.predict(cement_std)
#print("Strength: %.3f" % sc_y.inverse_transform(strength_std))


# ## Estimating the coefficient of a regression model via scikit-learn


slr = LinearRegression()
slr.fit(X_train, y_train)
y_pred = slr.predict(X_train)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)
# Print R^2 
print('R^2: %.3f' % slr.score(X_train, y_train))



lin_regplot(X_train, y_train, slr)
plt.xlabel('[cement] ')
plt.ylabel('[strength] ')

plt.savefig('10_07.png', dpi=300)
plt.show()


# **Normal Equations** alternative:



# adding a column vector of "ones"
#Xb = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
#w = np.zeros(X_train.shape[1])
#z = np.linalg.inv(np.dot(Xb.T, Xb))
#w = np.dot(z, np.dot(Xb.T, y_train))
#
#print('Slope: %.3f' % w[1])
#print('Intercept: %.3f' % w[0])



# # Fitting a robust regression model using RANSAC


#
#
#ransac = RANSACRegressor(LinearRegression(), 
#                         max_trials=100, 
#                         min_samples=80, 
#                         loss='absolute_loss', 
#                         residual_threshold=5.0, 
#                         random_state=42)
#
#
#ransac.fit(X, y)
#
#inlier_mask = ransac.inlier_mask_
#outlier_mask = np.logical_not(inlier_mask)
#
#line_X = np.arange(3, 10, 1)
#line_y_ransac = ransac.predict(line_X[:, np.newaxis])
#plt.scatter(X[inlier_mask], y[inlier_mask],
#            c='steelblue', edgecolor='white', 
#            marker='o', label='Inliers')
#plt.scatter(X[outlier_mask], y[outlier_mask],
#            c='limegreen', edgecolor='white', 
#            marker='s', label='Outliers')
#plt.plot(line_X, line_y_ransac, color='black', lw=2)   
#plt.xlabel(' [cement]')
#plt.ylabel(' [strength]')
#plt.legend(loc='upper left')
#
#plt.savefig('10_08.png', dpi=300)
#plt.show()
#
#
#
#
#print('Slope: %.3f' % ransac.estimator_.coef_[0])
#print('Intercept: %.3f' % ransac.estimator_.intercept_)




# # Evaluating the performance of linear regression models




X = df.iloc[:, :-1].values
y = df['strength'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)




slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

np.set_printoptions(precision=3)
print('Slope:' , slr.coef_)
print('Intercept: %.3f' % slr.intercept_)
# Print R^2 
print('R^2: %.3f' % slr.score(X_train, y_train))




ary = np.array(range(100000))




plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
plt.xlim([-10, 90])
plt.tight_layout()

plt.savefig('10_09.png', dpi=300)
plt.show()





print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# # Using regularized methods for regression



def display_plot(cv_scores, cv_scores_std,str):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.savefig(str+'10_10.png', dpi=300)
    plt.show()



print('ridge regression')
# Ridge regression:

alpha_space = np.logspace(-2, 0, 5)
ridge_scores = []
ridge_scores_std = []
# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

for alpha in alpha_space:
# Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
#    plt.plot(range(len(df.columns)-1), ridge.coef_)
#    plt.xticks(range(len(df.columns)-1), df.columns.values, rotation=60)
#    plt.margins(0.02)
#    plt.show()
    print('Slope:' , ridge.coef_)
    print('Intercept: %.3f' % ridge.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

    ary = np.array(range(100000))




    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.tight_layout()
    plt.title('alpha='+str(alpha))
    plt.savefig('ridge alpha='+str(alpha)+'10_09.png', dpi=300)
    plt.show()


alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []


# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)


# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X_train, y_train, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std,'ridge')



#print(ridge.coef_)




print('lasso regression')
# LASSO regression:


alpha_space = np.logspace(-2, 0, 5)
lasso_scores = []
lasso_scores_std = []

# Create a ridge regressor: lasso
lasso = Lasso(normalize=True)


for alpha in alpha_space:
# Specify the alpha value to use: ridge.alpha
    lasso.alpha = alpha
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
#    plt.plot(range(len(df.columns)-1), ridge.coef_)
#    plt.xticks(range(len(df.columns)-1), df.columns.values, rotation=60)
#    plt.margins(0.02)
#    plt.show()
    print('Slope:' , lasso.coef_)
    print('Intercept: %.3f' % lasso.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

    ary = np.array(range(100000))




    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.tight_layout()
    plt.title('alpha='+str(alpha))
    plt.savefig('lasso alpha='+str(alpha)+'10_09.png', dpi=300)
    plt.show()


#ax = plt.gca()
#ax.plot(lasso.coef_)
#ax.set_xscale('log')
#ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
#plt.xlabel('alpha')
#plt.ylabel('weights')
#plt.title('Ridge coefficients as a function of the regularization')
#plt.axis('tight')
#plt.show()


alpha_space = np.logspace(-4, 0, 50)
lasso_scores = []
lasso_scores_std = []


# Create a ridge regressor: lasso
lasso = Lasso(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: lasso.alpha
    lasso.alpha = alpha
    
    # Perform 10-fold CV: lasso_cv_scores
    lasso_cv_scores = cross_val_score(lasso, X_train, y_train, cv=10)
    
    # Append the mean of ridge_cv_scores to lasso_scores
    lasso_scores.append(np.mean(lasso_cv_scores))
    
    # Append the std of ridge_cv_scores to lasso_scores_std
    lasso_scores_std.append(np.std(lasso_cv_scores))

# Display the plot
display_plot(lasso_scores, lasso_scores_std,'lasso')

print('Elastic Net regression')
# Elastic Net regression:


alpha_space = np.logspace(-2, 0, 5)
elanet_scores = []
elanet_scores_std = []

# Create a ridge regressor: lasso
elanet = ElasticNet(alpha=1.0)


for alpha in alpha_space:
# Specify the alpha value to use: ridge.alpha
    elanet.l1_ratio = alpha
    elanet.fit(X_train, y_train)
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)
#    plt.plot(range(len(df.columns)-1), ridge.coef_)
#    plt.xticks(range(len(df.columns)-1), df.columns.values, rotation=60)
#    plt.margins(0.02)
#    plt.show()
    print('Slope:' , elanet.coef_)
    print('Intercept: %.3f' % elanet.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

    ary = np.array(range(100000))




    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.tight_layout()
    plt.title('l1_ratio='+str(alpha))
    plt.savefig('ElasticNet l1_ratio='+str(alpha)+'10_09.png', dpi=300)
    plt.show()

alpha_space = np.logspace(-4, 0, 50)
elanet_scores = []
elanet_scores_std = []


# Create a ridge regressor: ridge
#elanet = ElasticNet(normalize=True)
elanet = ElasticNet(alpha=1.0,l1_ratio=0.5,normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    elanet.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    elanet_cv_scores = cross_val_score(elanet, X_train, y_train, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    elanet_scores.append(np.mean(elanet_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    elanet_scores_std.append(np.std(elanet_cv_scores))

# Display the plot
display_plot(elanet_scores, elanet_scores_std,'elanet')


print("My name is Yuchen Duan")
print("My NetID is: yuchend3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
