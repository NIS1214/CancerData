##############################################################################
#**Code**
import pandas as pd
df = pd.read_csv('wdbc.data', header=None)

# The cancer data file has 30 features.
# Assign the 30 features to a numpy array x.
# Then using a lable encoder object, transform the class labels
# in column 2 (M for malignent / B for benign) into integers
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

# get 30 features
X = df.loc[:, 2:].values
#get lables
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)


# List the label for each class.
le.classes_

# To determine which value each class mapped into
# mapping: M mapped to 1, B mapped to 0
le.transform(['M','B'])


##############################################################################
# Training/Validation/Test procedure  / template section 2
# divide the data into 80% training and 20% test
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)


# run individually to see split sizes of the data
X.shape
X_train.shape
X_test.shape

#now develop model parameters/hyperparameters with train/validate
# then estimate the modesl generalization performance on the test set

##############################################################################
# Linear Regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Training set results
y_pred = regressor.predict(X_train)
#import matplotlib.pyplot as plt
#plt.figure(0)
#plt.plot(y_pred)

#regression returns a real value, so convert it to binary values.
y_pred=(y_pred > 0.5).astype(int)

TP=np.sum(y_pred==y_train)
accuracy=TP/y_pred.size
print('Train Accuracy Linear Regression: %.3f' % accuracy)
print('Train Accuracy Linear Regression(coeff of determination): %.3f' % regressor.score(X_train, y_train))

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_test)
accuracy=TP/y_pred.size
print('Test Accuracy Linear Regression: %.3f' % accuracy)
print('Test Accuracy Linear Regression(coeff of determination): %.3f' % regressor.score(X_test, y_test))
print('\n')
      



##############################################################################
# K-fold for Linear Regression
# *import numpy as np
from sklearn.model_selection import StratifiedKFold
S_KFold = StratifiedKFold(n_splits=10)
kfold = S_KFold.split(X_train, y_train)
scores = []

accuracies = []

for k, (train, test) in enumerate(kfold):    
    
     regressor.fit(X_train[train], y_train[train])
    
#compute  the regression error    
     score = regressor.score(X_train[test], y_train[test])        
     scores.append(score)                
     print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
     np.bincount(y_train[train]), score))

# compute true classification prediction accuracies    
     y_pred = regressor.predict(X_train[test]) # prediction return reals
     y_pred=(y_pred > 0.5).astype(int) #convert to discrete 0 or 1 for malignenat or benign
     TP=np.sum(y_pred==y_train[test]) #compute number of true positives
     accuracy=TP/y_pred.size
     accuracies.append(accuracy)#
     print('  Fold: Test Accuracy: %.3f' % accuracy)#               
    
print('\nCV accuracy Linear Regression (coeff of determination): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\nCV accuracy Linear Regression: %.3f +/- %.3f' % (np.mean(accuracies), np.std(accuracies))) #use this CV error!

# do the above k-fold algorithm for differnt algorithms.
# Then choose the model that gives the best CV error.
# then use that model, train it on all the train & validation data,
# then use the test data to check generlization performance.
# in the above, use the accuracies to determine performance.

##############################################################################
# confusion matrix for Linear Regression

# using the test data, generate a confusion matrix to view the performance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Training set results
y_pred = regressor.predict(X_test)

#regression returns a real value, so convert it to binary values.
y_pred=(y_pred > 0.5).astype(int)

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

#now use matplot lib to give a nice plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('LR Predicted label')
plt.ylabel('LR True label')

plt.tight_layout()
plt.show()


##############################################################################
# Naive Bayes Classifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

GaussianNB()

y_pred = nb.predict(X_test)

y_pred=(y_pred > 0.5).astype(int)

TP=np.sum(y_pred==y_train)
accuracy=TP/y_pred.size
print('Train Accuracy Naive Bayes: %.3f' % accuracy)
print('Train Accuracy Naive Bayes(coeff of determination): %.3f' % nb.score(X_train, y_train))

# Predicting the Test set results
y_pred = nb.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_test)
accuracy=TP/y_pred.size
print('Test Accuracy Naive Bayes: %.3f' % accuracy)
print('Test Accuracy Naive Bayes(coeff of determination): %.3f' % nb.score(X_test, y_test))
print('\n')


##############################################################################
# K-fold for Naive Bayes
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
S_KFold = StratifiedKFold(n_splits=10)
kfold = S_KFold.split(X_train, y_train)
scores = []

accuracies = []

for k, (train, test) in enumerate(kfold):    
    
     nb.fit(X_train[train], y_train[train])
    
#compute the error    
     score = nb.score(X_train[test], y_train[test])        
     scores.append(score)                
     print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
     np.bincount(y_train[train]), score))

# compute true classification prediction accuracies    
     y_pred = nb.predict(X_train[test]) # prediction return reals
     y_pred=(y_pred > 0.5).astype(int) #convert to discrete 0 or 1 for malignenat or benign
     TP=np.sum(y_pred==y_train[test]) #compute number of true positives
     accuracy=TP/y_pred.size
     accuracies.append(accuracy)#
     print('  Fold: Test Accuracy: %.3f' % accuracy)#               
    
print('\nCV accuracy Naive Bayes (coeff of determination): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\nCV accuracy Naive Bayes: %.3f +/- %.3f' % (np.mean(accuracies), np.std(accuracies))) #use this CV error!

##############################################################################
# Confusion matrix for Naive Bayes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

nb = GaussianNB()
nb.fit(X_train, y_train)

# Predicting the Training set results
y_pred = nb.predict(X_test)

# returns a real value, so convert it to binary values.
y_pred=(y_pred > 0.5).astype(int)

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

#now use matplot lib to give a nice plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('NB Predicted label')
plt.ylabel('NB True label')

plt.tight_layout()
plt.show()


##############################################################################
# K-nearest-neighbor classifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_pred=(y_pred > 0.5).astype(int)

TP=np.sum(y_pred==y_train)
accuracy=TP/y_pred.size
print('\n')
print('Train Accuracy KNN: %.3f' % accuracy)
print('Train Accuracy KNN(coeff of determination): %.3f' % classifier.score(X_train, y_train))

# Predicting the Test set results
y_pred = nb.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_test)
accuracy=TP/y_pred.size
print('Test Accuracy KNN: %.3f' % accuracy)
print('Test Accuracy KNN(coeff of determination): %.3f' % classifier.score(X_test, y_test))
print('\n')


##############################################################################
# K-fold for K-Nearest-Neighbor
# *import numpy as np
from sklearn.model_selection import StratifiedKFold
S_KFold = StratifiedKFold(n_splits=10)
kfold = S_KFold.split(X_train, y_train)
scores = []

accuracies = []

for k, (train, test) in enumerate(kfold):    
    
     classifier.fit(X_train[train], y_train[train])
    
#compute the error    
     score = classifier.score(X_train[test], y_train[test])        
     scores.append(score)                
     print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
     np.bincount(y_train[train]), score))

# compute true classification prediction accuracies    
     y_pred = classifier.predict(X_train[test]) # prediction return reals
     y_pred=(y_pred > 0.5).astype(int) #convert to discrete 0 or 1 for malignenat or benign
     TP=np.sum(y_pred==y_train[test]) #compute number of true positives
     accuracy=TP/y_pred.size
     accuracies.append(accuracy)#
     print('  Fold: Test Accuracy: %.3f' % accuracy)#               
    
print('\nCV accuracy KNN (coeff of determination): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\nCV accuracy KNN: %.3f +/- %.3f' % (np.mean(accuracies), np.std(accuracies)))


##############################################################################
# Confusion matrix for K-Nearest-Neighbor
# using the test data, generate a confusion matrix to view the performance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = LinearRegression()
classifier.fit(X_train, y_train)

# Predicting the Training set results
y_pred = classifier.predict(X_test)

#returns a real value, so convert it to binary values.
y_pred=(y_pred > 0.5).astype(int)

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

#now use matplot lib to give a nice plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('KNN Predicted label')
plt.ylabel('KNN True label')

plt.tight_layout()
plt.show()
