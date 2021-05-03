#Building ANN

##-------------------------PART 1 = DATA PREPROCESSING---------------------
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Churn_Modelling.csv")
     #Set matrix of idependent variable
X=dataset.iloc[:,3:13].values
    #Set matrix of dependent variable
y=dataset.iloc[:,13].values

"""
#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
"""

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
  #Encoding categorical variable of index[1] country
labelencoder_X_1= LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
  #Encoding categorical variable of index[2] gender
labelencoder_X_2= LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
  #converting index[1] variable to dummy variable for avoiding comparison
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [("oh",OneHotEncoder(),[1])],remainder="passthrough")
X = ct.fit_transform(X)
  #Removing first column to avoid dummy variable trap
X = X[:,1:]

#Spliting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

#Feature Scalimg
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

##-------------------------PART 2 = MAKE THE ANN----------------------------
#Import KERAS Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
"""from keras.layers import Dropout
 """
#Initialising the ANN
classifier = Sequential()
#Adding the input layer and first hidden layer
classifier.add(Dense(6,kernel_initializer ='uniform',activation = 'relu',input_dim = 11))
"""classifier.add(Dropout(p=0.1))# we can try for p=0.2,0.3,0.4 and0.5
 """
#Adding the second hidden layer
classifier.add(Dense(6,kernel_initializer ='uniform',activation = 'relu'))
"""classifier.add(Dropout(p=0.1))# we can try for p=0.2,0.3,0.4 and0.5
 """
#Adding the output layer
classifier.add(Dense(1,kernel_initializer ='uniform',activation = 'sigmoid'))
#Compiling the ANN
classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
""" for updating version #Compiling the ANN
    classifier.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    """
#Fitting the ANN to the Training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

##-------PART 3 = MAKING THE PREDICTIONS & EVALUATING THE MODEL----------------------
#Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Predicting a single new observation
new_pred = classifier.predict(sc_X.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred=(new_pred>0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

##-------PART 4 = Evaluating,Improving and Tuning the ANN----------------------
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
  #Building function
def built_classifier():
    classifier = Sequential()
    classifier.add(Dense(6,kernel_initializer ='uniform',activation = 'relu',input_dim = 11))
    classifier.add(Dense(6,kernel_initializer ='uniform',activation = 'relu'))
    classifier.add(Dense(1,kernel_initializer ='uniform',activation = 'sigmoid'))
    classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    """ for updating version #Compiling the ANN
    classifier.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    """
    return classifier

  #Wrapping the model
  classifier =KerasClassifier(build_fn = built_classifier,batch_size=10,epochs=100)
  #Applying cross validation
  accuracies =cross_val_score(estimator=classifier,X=X_train,y=y_train,scoring="accuracy",cv=10)
  mean=accuracies.mean()
  variance=accuracies.std()
  print(mean)
  print(variance)
  
#Improving the ANN
  #Dropout regularization to reduce overfitting if needed
  
#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV #from sklearn.grid_search import GridSearchCV 
from keras.models import Sequential
from keras.layers import Dense
  #Building function
def built_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6,kernel_initializer ='uniform',activation = 'relu',input_dim = 11))
    classifier.add(Dense(6,kernel_initializer ='uniform',activation = 'relu'))
    classifier.add(Dense(1,kernel_initializer ='uniform',activation = 'sigmoid'))
    classifier.compile(optimizer='optimizer',loss='binary_crossentropy',metrics=['accuracy'])
    """ for updating version #Compiling the ANN
    classifier.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    """
    return classifier

  classifier =KerasClassifier(build_fn = built_classifier)
  parameters={'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']}
  grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
  grid_search=grid_search.fit(X_train,y_train)
  best_parameters = grid_search.best_params_
  best_accuracy = grid_search.best_score_
  
  























