##x_test,x_train,y_train,y_test are subject to personal preference 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

dataset = pd.read_csv('file.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2) ##(random state = 0) and (test_size <.30) 

from sklearn.preprocessing import StandardScaler  #scaling is expected in sklearn.deomposition.KernelPCA
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.decomposition import KernelPCA   
kernel = KernelPCA(n_components=2,kernel='rbf') ##kernel and components are personal prefernce! refer the documentation for more methods of this class
x_train = kernel.fit_transform(x_train)
x_test = kernel.transform(x_test)

from sklearn.linear_model import LogisticRegression   ##training model over LogisticRegression 
classifier = LogisticRegression(random_state=0)
y_pred = classifier.fit(x_train,y_train).predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix ##prediction-actual comparision/test
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
