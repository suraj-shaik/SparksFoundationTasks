"""#task 2 Prediction using Unsupervised ML

## **for this task there are 6 sections as follows :
## 1)data importing with pandas
## 2) checking for outliers
## 3) making dependent and independent values
## 4) creating and fitting KMEANS clusters model
## 5) coloring the predicted clusters
## 6) 3d graphical represtation of the clusters

### **the following are the neccessary packages for the task
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans,Birch,DBSCAN,AffinityPropagation,OPTICS
import matplotlib.pyplot as plt
import numpy as np

"""## 1)data importing with pandas"""

data2=pd.read_csv('Iris.csv')
data2

"""## 2) checking for outliers

"""

figr,axr =  plt.subplots(1,2)
axr[0].boxplot(data2['SepalLengthCm'])
axr[1].boxplot(data2['SepalWidthCm'])
axr[0].set_title('SepalLengthCm')
axr[1].set_title('SepalWidthCm')
plt.show()

"""***outliers exists in SepalWidthCm***"""

figr,axr =  plt.subplots(1,2)
axr[0].boxplot(data2['PetalLengthCm'])
axr[1].boxplot(data2['PetalWidthCm'])
axr[0].set_title('PetalLengthCm')
axr[1].set_title('PetalWidthCm')
plt.show()

"""***removing outliers***"""

q1cl=data2['SepalWidthCm'].quantile(0.25)
q3cl=data2['SepalWidthCm'].quantile(0.75)
iqrcl=q3cl-q1cl
mincl=q1cl-(1*iqrcl)
maxcl=q3cl+(1*iqrcl)
print(mincl,maxcl)

data2=data2[data2['SepalWidthCm']>=mincl]
data2=data2[data2['SepalWidthCm']<=maxcl]
data2

"""***after removing outliers***"""

figr,axr =  plt.subplots(1,2)
axr[0].boxplot(data2['SepalLengthCm'])
axr[1].boxplot(data2['SepalWidthCm'])
axr[0].set_title('SepalLengthCm')
axr[1].set_title('SepalWidthCm')
plt.show()

figr,axr =  plt.subplots(1,2)
axr[0].boxplot(data2['PetalLengthCm'])
axr[1].boxplot(data2['PetalWidthCm'])
axr[0].set_title('PetalLengthCm')
axr[1].set_title('PetalWidthCm')
plt.show()

data2['Species'].unique() #all unique cluster names

"""## 3) making dependent and independent values

"""

xcl=data2.iloc[:,1:-1].values
ycl=data2.iloc[:,-1:].values
xcl

"""## 4) creating and fitting KMEANS clusters model"""

clus1=KMeans(n_clusters=3,random_state=1).fit(xcl)
clus1

"""***the optimal number of clusters are 3***

## 5) coloring the predicted clusters
"""

data2['clus']=clus1.predict(xcl)
data2['color']=data2.clus.map({0:'red',1:'yellow',2:'blue'})
data2

"""## 6) 3d graphical represtation of the clusters

### graph for SepalLengthCm SepalWidthCm PetalLengthCm
"""

figcl=plt.figure(figsize=(21,15))
axcl=plt.axes(projection='3d')
axcl.scatter(data2['SepalLengthCm'],data2['SepalWidthCm'],data2['PetalLengthCm'],c=data2.color)
axcl.set_xlabel('SepalLengthCm')
axcl.set_ylabel('SepalWidthCm')
axcl.set_zlabel('PetalLengthCm')
plt.show()

"""### graph for SepalLengthCm SepalWidthCm PetalWidthCm

"""

figcl=plt.figure(figsize=(21,15))
axcl=plt.axes(projection='3d')
axcl.scatter(data2['SepalLengthCm'],data2['SepalWidthCm'],data2['PetalWidthCm'],c=data2.color)
axcl.set_xlabel('SepalLengthCm')
axcl.set_ylabel('SepalWidthCm')
axcl.set_zlabel('PetalWidthCm')
plt.show()

"""### graph for PetalWidthCm SepalWidthCm PetalLengthCm  

"""

figcl=plt.figure(figsize=(21,15))
axcl=plt.axes(projection='3d')
axcl.scatter(data2['PetalWidthCm'],data2['SepalWidthCm'],data2['PetalLengthCm'],c=data2.color)
axcl.set_xlabel('PetalWidthCm')
axcl.set_ylabel('SepalWidthCm')
axcl.set_zlabel('PetalLengthCm')
plt.show()
