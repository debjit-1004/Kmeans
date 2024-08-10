from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd 

bc=load_breast_cancer()


X=bc.data

Y=bc.target

X_train,X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2)

model=KMeans(n_clusters=2,random_state=0)

model.fit(X_train)

predictions=model.predict(X_test)

labels=model.labels_

print("labels:", labels)
print("predictions:", predictions)
print('accuracy:', accuracy_score(Y_test, predictions))
print('actual:', Y_test)