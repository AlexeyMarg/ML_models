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