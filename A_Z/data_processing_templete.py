#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read the dataset
dataset=pd.read_csv('Data.csv');
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values
#print(x)

#taking care of missing values
from sklearn.impute import SimpleImputer as sp
imputer =sp(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
'''np.set_printoptions(suppress=True)
print(x)
np.set_printoptions(suppress=True)
print(y)
'''

#spliting data into training data or testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
'''np.set_printoptions(suppress=True)
print(x_train)
print("---------------------test------------------------")
print(x_test)
print("----------------------traning-----------------------")
print(y_train)
print("--------------------------test-------------------")
print(y_test)
'''

#scaling data for the future
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
np.set_printoptions(suppress=True)
print(x_train)
print(x_test)