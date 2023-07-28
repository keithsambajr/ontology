from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def tune_logistic_regression(X_train_feature, y_train):
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter (smaller values for more regularization)
    }
    
    lr_grid_search = GridSearchCV(LogisticRegression(), lr_param_grid, scoring='accuracy', cv=5)
    lr_grid_search.fit(X_train_feature, y_train)
    
    best_lr_params = lr_grid_search.best_params_
    best_lr_model = lr_grid_search.best_estimator_
    
    return best_lr_model, best_lr_params

def tune_svm(X_train_feature, y_train):
    svm_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter (smaller values for more regularization)
        'kernel': ['linear', 'rbf', 'poly'],  # Kernel type for SVM
    }
    
    svm_grid_search = GridSearchCV(SVC(), svm_param_grid, scoring='accuracy', cv=5)
    svm_grid_search.fit(X_train_feature, y_train)
    
    best_svm_params = svm_grid_search.best_params_
    best_svm_model = svm_grid_search.best_estimator_
    
    return best_svm_model, best_svm_params
