from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def svm_train_search(X_train, y_train):

  print('\nSVM parameter search is selected.')
                     
  params_grid = [{'decision_function_shape': ['ovo', 'ovr'], 'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                    {'decision_function_shape': ['ovo', 'ovr'],'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                    {'decision_function_shape': ['ovo', 'ovr'], 'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]             

  svm_model = GridSearchCV(SVC(), params_grid, n_jobs = 30, cv = 2)
  
  print('SVM Train...')
  svm_model.fit(X_train, y_train)
  print('SVM Train Finished.')
  
  return svm_model.best_params_, svm_model.best_estimator_

def svm_train(X_train, y_train):
  
  svm_model = SVC(kernel = 'rbf', gamma = 0.01, C = 100, random_state = 1)
  
  print('\nSVM Train...')
  svm_model.fit(X_train, y_train)
  print('SVM Train Finished.')
  
  return svm_model