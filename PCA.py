##x_train,y_train,x_test,y_test are subject to personal perference

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
##incase dataset is already splited
##from sklearn.model_selection import train_test_split
##x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.preprocessing import StandardScaler  #scaling is expected in sklearn.decomposition.PCA
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)   #modification is applicable 
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


y_pred = classifier.predict(x_test)
##actual-prediction analysis with modified componenets 
from sklearn.metrics import accuracy_score, confusion_matrix    
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred,))




