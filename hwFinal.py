import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import tensorflow as tf
import keras 
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics, svm 
from scikeras.wrappers import KerasClassifier 
from keras.callbacks import EarlyStopping
from matplotlib import pyplot


#The dataset needs to be placed in the same folder as the python file with the code.

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# #nan value bar chart
# plt.figure(figsize = (13,5))
# plt.bar(train_data.columns, train_data.isna().sum())
# plt.xlabel("Columns name")
# plt.ylabel("Number of missing values in training data")
# plt.show()

# #correlation heat map
# corr_train = train_data.corr()
# sns.heatmap(corr_train)
# plt.show()

# Dropping Cabin column due to high number of nan values
train_data.drop('Cabin', axis = 1, inplace = True)
test_data.drop('Cabin', axis = 1, inplace = True)

combined_data = [train_data, test_data]

# Filling nan values of Age and Fare with their means
for data in combined_data:
    data.Age.fillna(data.Age.mean(), inplace = True)
    data.Fare.fillna(data.Fare.mean(), inplace = True)

# Filling nan values with Southhampton since it is the most frequent embarked place
train_data.Embarked.fillna('S', inplace = True)

# Converting categorical features Sex and Embarked
def change_gender(x):
    if x == 'male':
        return 0
    elif x == 'female':
        return 1
train_data.Sex = train_data.Sex.apply(change_gender)
test_data.Sex = test_data.Sex.apply(change_gender)

change = {'S':1,'C':2,'Q':0}
train_data.Embarked = train_data.Embarked.map(change)
test_data.Embarked = test_data.Embarked.map(change)

# Combining Sibsp and Parch columns due to similarity
train_data['Alone'] = train_data.SibSp + train_data.Parch
test_data['Alone'] = test_data.SibSp + test_data.Parch

train_data.Alone = train_data.Alone.apply(lambda x: 1 if x == 0 else 0)
test_data.Alone = test_data.Alone.apply(lambda x: 1 if x == 0 else 0)

train_data.drop(['SibSp','Parch'], axis = 1, inplace = True)
test_data.drop(['SibSp','Parch'], axis = 1, inplace = True )

# Creating Title feature from Name
for data in combined_data:
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand = False)
    data.drop('Name', axis = 1, inplace = True)
least_occuring = [ 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess','Dona',
       'Jonkheer']
for data in combined_data:
    data.Title = data.Title.replace(least_occuring, 'Rare')

# Mapping titles to numbers
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for data in combined_data:
    data['Title'] = data['Title'].map(title_mapping)
    
# Dropping PassengerID and Ticket columns
columns_to_drop = ['PassengerId','Ticket']
train_data.drop(columns_to_drop, axis = 1, inplace = True)
test_data.drop(columns_to_drop[1], axis = 1, inplace = True)

# Binning Age and Fare Columns
for dataset in combined_data:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
for data in combined_data:
    data.loc[data['Fare'] < 30, 'Fare'] = 1
    data.loc[(data['Fare'] >= 30) & (data['Fare'] < 50),'Fare'] = 2
    data.loc[(data['Fare'] >= 50) & (data['Fare'] < 100),'Fare'] = 3
    data.loc[(data['Fare'] >= 100),'Fare'] = 4

# Training and testing data
X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_train = np.asarray(X_train).astype(np.float32)
Y_train = np.asarray(Y_train).astype(np.float32)
X_test = test_data.drop("PassengerId", axis = 1)
print("shape of X_train",X_train.shape)
print("Shape of Y_train",Y_train.shape)
print("Shape of X_test",X_test.shape)

# Split for validation and training
n_train = 600
trainX, testX = X_train[:n_train, :], X_train[n_train:, :]
trainY, testY = Y_train[:n_train], Y_train[n_train:]


# Kernel

# Kernel Hyperparameter Tuning section

# def kernelTuning():
#     Cs,gammas = np.logspace(-2,5,13),np.logspace(-4,4,13)
#     paramGrid = dict(gamma=gammas, C=Cs)
#     cv = StratifiedKFold(n_splits=5)
#     grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=paramGrid, cv=cv)
#     grid.fit(X_train, Y_train)
#     print("Best parameters: %s"% (grid.best_params_))
#     return grid.best_score_, grid.best_params_

# _, kernelBestParams = kernelTuning()

#Running the above tuning code, I got {'C': 31.622776601683793, 'gamma': 0.046415888336127774} as the optimal parameters

best_C, best_gamma = 31.622776601683793, 0.046415888336127774
kernel_model = svm.SVC(C=best_C, gamma=best_gamma, kernel='rbf')
kernel_model.fit(X_train, Y_train)
Y_pred_rand = (kernel_model.predict(testX) > 0.5).astype(int)
print("Kernel Stats:")
print(" ")
print('Accuracy : ', np.round(metrics.accuracy_score(testY, Y_pred_rand)*100,2))
kernel_test_out = kernel_model.predict(X_test)
kernel_test_df = pd.DataFrame({"PassengerId":test_data.PassengerId, 'Survived': [int(a) for a in kernel_test_out]})
kernel_test_df.to_csv(r'/Users/omsah/Downloads/hwFinal/kernel_output.csv', index = False, header=True)

# Neural Network nn_model

nn_model = Sequential()
nn_model.add(Dense(units = 32, input_shape = (7,), activation = 'relu'))
nn_model.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False))
nn_model.add(tf.keras.layers.BatchNormalization())
nn_model.add(Dense(units = 128, activation = 'relu',kernel_initializer = 'he_normal', use_bias = False))
nn_model.add(Dropout(0.1))
nn_model.add(Dense(units = 64, activation = 'relu',kernel_initializer = 'he_normal', use_bias = False))
nn_model.add(Dropout(0.1))
nn_model.add(Dense(units = 32, activation = 'relu'))
nn_model.add(Dropout(0.15))
nn_model.add(Dense(units = 16, activation = 'relu'))
nn_model.add(Dense(units = 8, activation = 'relu',kernel_initializer = 'he_normal', use_bias = False))
nn_model.add(Dense(units =1 , activation = 'sigmoid'))

# Neural Network Model Summary
# nn_model.summary()

# Compiling and training
nn_model.compile(loss = tf.keras.losses.binary_crossentropy, optimizer = tf.keras.optimizers.Adam(), metrics = ['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience = 5)
history = nn_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=4000, verbose=0, callbacks=[es])
_, train_acc = nn_model.evaluate(trainX, trainY, verbose=0)
_, test_acc = nn_model.evaluate(testX, testY, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.title("Training and Validation loss over iterations")
pyplot.ylabel("Loss")
pyplot.xlabel("Iterations")
pyplot.legend()
pyplot.show()
Y_pred_rand = (nn_model.predict(X_train) > 0.5).astype(int)
print("Neural Network Stats:")
print(" ")
print('Accuracy : ', np.round(metrics.accuracy_score(Y_train, Y_pred_rand)*100,2))
nn_test_out = nn_model.predict(X_test)
nn_test_df = pd.DataFrame({"PassengerId":test_data.PassengerId, 'Survived': [int(a) for a in nn_test_out]})
nn_test_df.to_csv(r'/Users/omsah/Downloads/hwFinal/nn_output.csv', index = False, header=True)

#Tree

#Tree Tuning code
# def treeTuner():
#     params, avgVals = [], []
#     for k in range(3, 5):
#         kFold = KFold(n_splits=k, shuffle=False)
#         for leaf in range(2, 30):
#             for d in range(1,10):
#                 accuracies = []
#                 for ti, vi in kFold.split(X_train):
#                     trainx, X_val, trainy, Y_val = X_train[ti],X_train[vi],Y_train[ti],Y_train[vi]
#                     clf = DecisionTreeClassifier(max_leaf_nodes=leaf, max_depth=d, random_state=42)
#                     clf.fit(trainx, trainy)
#                     accuracies.append(clf.score(X_val, Y_val))
#                 params.append({'k': k, 'max_leaf': leaf, 'max_depth': d})
#                 avgVals.append(clf.score(X_val, Y_val))
#     best_param = params[avgVals.index(max(avgVals))]
#     print("Final Parameters: ", best_param)
#     return max(avgVals), best_param

# _, treeBestParams = treeTuner() 

#Using the tuning code above, I got {'k': 4, 'max_leaf': 28, 'max_depth': 8} as the best parameters
best_leaf,best_depth = 28, 8
tree_model = DecisionTreeClassifier(max_leaf_nodes=best_leaf, max_depth=best_depth, random_state=42)
tree_model.fit(trainX, trainY)
Y_pred_rand = (tree_model.predict(testX) > 0.5).astype(int)
print("Tree Stats:")
print(" ")
# print('Precision : ', np.round(metrics.precision_score(testY, Y_pred_rand)*100,2))
print('Accuracy : ', np.round(metrics.accuracy_score(testY, Y_pred_rand)*100,2))
tree_out = tree_model.predict(X_test)
treedf = pd.DataFrame({"PassengerId":test_data.PassengerId, 'Survived': [int(a) for a in tree_out]})
treedf.to_csv(r'/Users/omsah/Downloads/hwFinal/tree_output.csv', index = False, header=True)

