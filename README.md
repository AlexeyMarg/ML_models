# ML_models
##linear_models:
regressors:
- dataset_generation.py - generation of datasets
- normal_equation.py - solution of linear regression with use of normal equation
- gradient_descent.py - solution of linear regression with use of gradient descent
- stochastic_gradient_descent.py - stochastic gradient search and mini-batch gradient search
- regularization.py - examples of 
    - ridge (add l2 norm of parameters vector to MSE)
    - lasso regularization (add l1 norm of parameters vector to MSE)
    - elastic net regularization - mix of both
    
classifiers:
- logistic_regression.py - logistic regression classifier

##SVM:
classifiers:
- linear_SVM_clf.py - linear SVM classifier for iris dataset
- polynomial_SVM_clf.py - polynomial SVM classifier for nonlinear dataset
- kernel_SVM.py - SVM classifier with kernel trick instead polynomial
- rbf_SVM_clf.py - SVM classifier with Gaussian RBF kernel

regressors:
- linear_SVM_reg.py - SVM linear regressor
- kernel_SVM_reg.py - SVM kernel regressor

##Decision tree:
classifiers:
- decision_tree_clf.py - decision tree for iris classification
- random_forest_clf.py - random forest for iris classification

regressors:
- decision_tree_reg.py - decision tree regressor for noised parabola

##Ensembles
- voting_clf.py - voting classifier with random forrest, SVM and logistic regression
- bagging_clf.py - use of bagging (random choise of train data) for decision tree in iris task
- AdaBoost_clf.py - use of adaptive boosting for decision tree in iris task
- gradient_boost_reg.py - gradient boosting regressor for noised parabola

##Dimensions reduction
- PCA.py - handmade and sklearn principal component reduction for dimension reduction
= batch_PCA.py - incremental PCA (less use of memory) for batch principal component reduction for dimension reduction for MNIST
- kernel_PCA.py - kernel PCA dimension reduction
- LLE.py - Locally linear embedding for dimension reduction

##tensor_flow
- hello_world.py - hello world for tensor flow