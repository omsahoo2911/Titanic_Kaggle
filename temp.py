# Load libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split # Import train_test_split function
from sklearn import metrics, svm #Import scikit-learn metrics module for accuracy calculation

import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential

#Sources:

    #Kernel: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py

    #Trees: https://www.datacamp.com/tutorial/decision-tree-classification-python
    #Hyperparamater Tuning       https://www.section.io/engineering-education/hyperparmeter-tuning/
    #K-Fold       https://isheunesu48.medium.com/cross-validation-using-k-fold-with-scikit-learn-cfc44bf1ce6
#

#Pre-Processing:
#Keeping only numerical features which are fully filled out

#Renaming columns
train_col_names = ['id', 'survived', 'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked']
train = pd.read_csv("train.csv", header=None, names=train_col_names)
train = train.drop([0])

test_col_names = ['id', 'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked']
test = pd.read_csv("test.csv", header=None, names=test_col_names)
test = test.drop([0])

#Selecting columns with desired features
feature_cols = ['pclass', 'sibsp', 'parch', 'fare']
X = train[feature_cols]
X.columns.name = None

x_test = test[feature_cols]
x_test.columns.name = None

#Setting the labels to the survived feature
Y = train.survived
Y.index.name = None

#Hyperparameter tuning

#Kernel
def kernel_hyp_tuning():
    # Neural Network nn_model
    C_range = np.logspace(-2, 4, 13)
    gamma_range = np.logspace(-4, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(n_splits=5)
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    grid.fit(X, Y)

    print(
        "The best parameters are %s with a score of %0.7f"
        % (grid.best_params_, grid.best_score_)
    )

    return grid.best_score_, grid.best_params_

#NN
def NN_hyp_tuning():
    # Neural Network nn_model
    param_arr = []
    avg_val_arr = []

    max_alpha = 0.01
    max_lr = 0.1
    max_iters = 1000

    for k in range(3, 7):
        kFold = KFold(n_splits=k, shuffle=False)
        alpha = 0.00001
        while alpha <= max_alpha:
            lr = 0.00001
            while lr <= max_lr:
                val_accs = []
                for train_index, val_index in kFold.split(X):
                    X_train, X_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], Y.iloc[train_index], Y.iloc[val_index]
                    nn_class = MLPClassifier(random_state=1, learning_rate_init=lr, alpha=alpha, max_iter=max_iters)
                    nn_class.fit(X_train, y_train)
                    acc = nn_class.score(X_val, y_val)
                    val_accs.append(acc)

                    #print('The Training Accuracy for max_depth {} is:'.format(max_d), clf.score(X_train, y_train))
                    #print('The Validation Accuracy for max_depth {} is:'.format(max_d), clf.score(X_val, y_val))
                    #print()
                
                avg_val = sum(val_accs) / len(val_accs)
                param_arr.append({'k': k, 'lr': lr, 'alpha': alpha})
                avg_val_arr.append(avg_val)

                print("Curr Iter: ", {'k': k, 'lr': lr, 'alpha': alpha})
                print()

                lr *= 10
            
            alpha *= 10

        
    best_val = max(avg_val_arr)
    print("------------------------------------------")
    print()
    print("Best Average Val: ", best_val)

    best_ind = avg_val_arr.index(best_val)
    best_param = param_arr[best_ind]
    print("Final Parameters: ", best_param)
    print()

    return best_val, best_param

#Tree
def tree_hyp_tuning():
    param_arr = []
    avg_val_arr = []

    for k in range(3, 7):
        kFold = KFold(n_splits=k, shuffle=False)
        for max_leaf in range(2, 50):
            for max_d in range(1,21):
                val_accs = []
                for train_index, val_index in kFold.split(X):
                    X_train, X_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], Y.iloc[train_index], Y.iloc[val_index]
                    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf, max_depth=max_d, random_state=42)
                    clf.fit(X_train, y_train)
                    acc = clf.score(X_val, y_val)
                    val_accs.append(acc)

                    #print('The Training Accuracy for max_depth {} is:'.format(max_d), clf.score(X_train, y_train))
                    #print('The Validation Accuracy for max_depth {} is:'.format(max_d), clf.score(X_val, y_val))
                    #print()
                
                avg_val = sum(val_accs) / len(val_accs)
                param_arr.append({'k': k, 'max_leaf': max_leaf, 'max_d': max_d})
                avg_val_arr.append(avg_val)

                print("Curr Iter: ", {'k': k, 'max_leaf': max_leaf, 'max_d': max_d})
                print()

        
    best_val = max(avg_val_arr)
    print("------------------------------------------")
    print()
    print("Best Average Val: ", best_val)

    best_ind = avg_val_arr.index(best_val)
    best_param = param_arr[best_ind]
    print("Final Parameters: ", best_param)
    print()

    return best_val, best_param

#
# Kernel
#
kernel_best_val, best_kernel_param = kernel_hyp_tuning()
# Best Average Val Acc = 0.7093717

# best_C = best_kernel_param['C']
# best_gamma = best_kernel_param['gamma']

#best_k = 5
best_C = 316.22776601683796
best_gamma = 4.641588833612782

kernel_test_clf = svm.SVC(C=best_C, gamma=best_gamma, kernel='rbf')
kernel_test_clf.fit(X, Y)
kernel_test_out = kernel_test_clf.predict(x_test.fillna(0))

kernel_test_df = pd.DataFrame({'Survived': kernel_test_out})
pass_id = test[['id']]
kernel_test_df['PassengerID'] = pass_id['id'].tolist()

#Swap order of columns
columns_titles = ["PassengerID","Survived"]
kernel_test_df = kernel_test_df.reindex(columns=columns_titles)
kernel_test_df.to_csv(r'/Users/johnschool/Documents/Fall 2023/CS589/Final/kernel_output.csv', index = False, header=True)

#
# Neural Network
#
#nn_best_val, best_nn_param = NN_hyp_tuning()

# Best Average Val Acc = 0.7082731780804721

# best_lr = best_nn_param['lr']
# best_alpha = best_nn_param['alpha']

#best_k = 5
best_lr = 0.1
best_alpha = 0.001

nn_test_clf = MLPClassifier(random_state=1, learning_rate_init=best_lr, alpha=best_alpha, max_iter=1000)
nn_test_clf.fit(X, Y)
nn_test_out = nn_test_clf.predict(x_test.fillna(0))

nn_test_df = pd.DataFrame({'Survived': nn_test_out})
pass_id = test[['id']]
nn_test_df['PassengerID'] = pass_id['id'].tolist()

#Swap order of columns
columns_titles = ["PassengerID","Survived"]
nn_test_df = nn_test_df.reindex(columns=columns_titles)
nn_test_df.to_csv(r'/Users/johnschool/Documents/Fall 2023/CS589/Final/nn_output.csv', index = False, header=True)

#
# Trees
#

#tree_best_val, best_tree_param = tree_hyp_tuning()
# Best Average Val Acc = 0.7351684644285541

# best_leaf = best_tree_param['max_leaf']
# best_depth = best_tree_param['max_d']

#best_k = 4
best_leaf = 17
best_depth = 8

tree_test_clf = DecisionTreeClassifier(max_leaf_nodes=best_leaf, max_depth=best_depth, random_state=42)
tree_test_clf.fit(X, Y)
tree_test_out = tree_test_clf.predict(x_test)

tree_test_df = pd.DataFrame({'Survived': tree_test_out})
pass_id = test[['id']]
tree_test_df['PassengerID'] = pass_id['id'].tolist()

#Swap order of columns
columns_titles = ["PassengerID","Survived"]
tree_test_df = tree_test_df.reindex(columns=columns_titles)

# print(tree_test_df)
# print()

#Save Tree Model Output
tree_test_df.to_csv(r'/Users/johnschool/Documents/Fall 2023/CS589/Final/tree_output.csv', index = False, header=True)






