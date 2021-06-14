"""#task 6 Prediction using Decision Tree Algorithm

## **for this task there are 6 sections as follows :
## 1)data importing with pandas
## 2) checking for outliers
## 3) making dependent and independent values
## 4) creating and fitting Decision tree Classifier model
## 5) evaluation of the test
## 6) tree building
### **the following are the neccessary packages for the task
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

"""## 1)data importing with pandas

"""

data3=pd.read_csv('Iris.csv')
data3

"""## 2) checking for outliers

"""

figcls,axcls =  plt.subplots(1,2)
axcls[0].boxplot(data3['SepalLengthCm'])
axcls[1].boxplot(data3['SepalWidthCm'])
axcls[0].set_title('SepalLengthCm')
axcls[1].set_title('SepalWidthCm')
plt.show()

"""***outliers exists***"""

figcls,axcls =  plt.subplots(1,2)
axcls[0].boxplot(data3['PetalLengthCm'])
axcls[1].boxplot(data3['PetalWidthCm'])
axcls[0].set_title('PetalLengthCm')
axcls[1].set_title('PetalWidthCm')
plt.show()

"""***removing outliers***"""

q1cls=data3['SepalWidthCm'].quantile(0.25)
q3cls=data3['SepalWidthCm'].quantile(0.75)
iqrcls=q3cls-q1cls
mincls=q1cls-(1*iqrcls)
maxcls=q3cls+(1*iqrcls)
print(mincls,maxcls)

data3=data3[data3['SepalWidthCm']>=mincls]
data3=data3[data3['SepalWidthCm']<=maxcls]
#data3

"""***data after removing outliers***"""

figcls,axcls =  plt.subplots(1,2)
axcls[0].boxplot(data3['SepalLengthCm'])
axcls[1].boxplot(data3['SepalWidthCm'])
axcls[0].set_title('SepalLengthCm')
axcls[1].set_title('SepalWidthCm')
plt.show()

figcls,axcls =  plt.subplots(1,2)
axcls[0].boxplot(data3['PetalLengthCm'])
axcls[1].boxplot(data3['PetalWidthCm'])
axcls[0].set_title('PetalLengthCm')
axcls[1].set_title('PetalWidthCm')
plt.show()

"""## 3) spliting the data into train and test

"""

clsxtrain,clsxtest,clsytrain,clsytest=train_test_split(data3.iloc[:,1:-1].values,
                                                      data3.iloc[:,-1:],test_size=0.3,random_state=1)

"""## 4) creating and fitting Decision tree Classifier model

"""

cls1=DecisionTreeClassifier().fit(clsxtrain,clsytrain)
cls1

"""## 5) evaluation of the test

"""

predcls=cls1.predict(clsxtest)

acc1 = round(metrics.accuracy_score(clsytest, predcls),3)
print('decision tree pred : ',acc1)

"""***result gave 100% accuracy***

## 6) tree building

### text tree
"""

sda=tree.export_text(cls1,feature_names=['SepalLengthCm'	,'SepalWidthCm'	,'PetalLengthCm'	,'PetalWidthCm'	])
print(sda)

"""### plot tree"""

figcls = plt.figure(figsize=(21,16))
tree.plot_tree(cls1,feature_names=['SepalLengthCm'	,'SepalWidthCm'	,'PetalLengthCm'	,'PetalWidthCm'	],filled=True)
plt.show()