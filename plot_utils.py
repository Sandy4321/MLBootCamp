import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import PolynomialFeatures

def plot_data(X,y,xlabel,ylabel):
    fig = plt.figure()
    plt.plot(X,y,'bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_line(X,y,xlabel,ylabel):
    fig = plt.figure()
    plt.plot(X,y,'b-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def make_surface_plot(X,Y,Z,xlabel,ylabel):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z,cmap=cm.jet)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel);


def make_contour_plot(X,Y,Z,levels,xlabel,ylabel,theta):
    plt.figure()
    CS = plt.contour(X, Y, Z, levels = levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot([theta[0]],[theta[1]], marker='x',color='r',markersize=10)


def plot_learning_curve(error_train,error_val,reg):
    plt.figure()
    xvals = np.arange(2,len(error_train)+1)
    plt.plot(xvals,error_train[1:],'b-',xvals,error_val[1:],'g-')
    plt.title('Learning curve for linear regression with lambda = '+str(reg))
    plt.xlabel('Number of training examples')
    plt.ylabel('Training/Validation error')
    plt.legend(["Training error","Validation error"])

def plot_fit(X,y,minx, maxx, mu, sigma, theta, p, xlabel, ylabel, title):

    plt.figure()
    plt.plot(X,y,'bo')

    # plots a learned polynomial regression fit 

    x = np.arange(minx - 5,maxx+15,0.1)
    # map the X values
    poly = sklearn.preprocessing.PolynomialFeatures(p,include_bias=False)
    x_poly = poly.fit_transform(np.reshape(x,(len(x),1)))
    x_poly = (x_poly - mu) / sigma
    # add the column of ones
    xx_poly = np.vstack([np.ones((x_poly.shape[0],)),x_poly.T]).T
    plt.plot(x,np.dot(xx_poly,theta))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plot_lambda_selection(reg_vec,error_train,error_val):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(reg_vec,error_train,'b-',reg_vec,error_val,'g-')
  plt.title('Variation in training/validation error with lambda')
  plt.xlabel('Lambda')
  plt.ylabel('Training/Validation error')
  plt.legend(["Training error","Validation error"])
  ax.set_xscale('log')

def plot_twoclass_data(X,y,xlabel,ylabel,legend):
    fig = plt.figure()
    X0 = X[np.where(y==0)]
    X1 = X[np.where(y==1)]
    plt.scatter(X0[:,0],X0[:,1],c='red', s=40, label = legend[0])
    plt.scatter(X1[:,0],X1[:,1],c='green', s = 40, label=legend[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    
def plot_decision_boundary_sklearn(X,y,sk_logreg,  xlabel, ylabel, legend):
    plot_twoclass_data(X,y,xlabel,ylabel,legend)
    
    # create a mesh to plot in
    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh
    Z = np.array(sk_logreg.predict(np.c_[xx1.ravel(), xx2.ravel()]))

    # Put the result into a color contour plot
    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])

def plot_decision_boundary_sklearn_poly(X,y,sk_logreg,reg,p,  xlabel, ylabel, legend):
    plot_twoclass_data(X,y,xlabel,ylabel,legend)
    
    # create a mesh to plot in
    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh
    poly = sklearn.preprocessing.PolynomialFeatures(degree=p,include_bias=False)
    X_poly = poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()])
       
    
    Z = np.array(sk_logreg.predict(X_poly))

    # Put the result into a color contour plot
    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])
    plt.title("Decision boundary for lambda = " + str(reg))


import sklearn
from sklearn import svm, linear_model
from sklearn.svm import l1_min_c

# From
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#example-linear-model-plot-logistic-path-py

def plot_regularization_path(X,y):
    plt.figure()
    cs = sklearn.svm.l1_min_c(X, y, loss='log') * np.logspace(0, 3)
    print("Computing regularization path ...")
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, y)
        coefs_.append(clf.coef_.ravel().copy())

    coefs_ = np.array(coefs_)
    plt.plot(np.log10(cs), coefs_)
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression Path')

from matplotlib import colors
from matplotlib.colors import ListedColormap

def plot_datasets(X,y,X_train,y_train,X_test,y_test,i):

    h = 0.02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first                                                              
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(3, 4, i)
    # Plot the training points                                                                 
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points                                                                       
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    return  xx, yy

def plot_decision_boundary_classifier(name,model,xx,yy,X_train,y_train,X_test,y_test,score,i):
    # Plot the decision boundary. For that, we will assign a color to each                     
    # point in the mesh [x_min, m_max]x[y_min, y_max].                                         
    ax = plt.subplot(3, 4, i)

    if hasattr(model, "decision_function"):
       Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
       Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Put the result into a color plot                                                         
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    # Plot also the training points                                                            
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points                                                                       
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
    