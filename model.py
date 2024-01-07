import numpy as np
import pandas as pd
import os, sys
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#DataFlair - Read the data
df=pd.read_csv("C:/Users/vetri/Downloads/parkinsons/parkinsons.data")
df.head()
#DataFlair - Get the features and labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values
#DataFlair - Get the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])
#DataFlair - Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
#DataFlair - Train the model
model=XGBClassifier(base_score=0.5,
    booster='gbtree',
    colsample_bylevel=1,
    colsample_bynode=1,
    colsample_bytree=1,
    grow_policy='depthwise',
    importance_type='gain',
    learning_rate=0.1,
    max_bin=256,
    max_cat_threshold=32,
    max_cat_to_onehot=4,
    max_delta_step=0,
    max_depth=3,
    max_leaves=0,
    min_child_weight=1,
    n_estimators=100,
    num_parallel_tree=1,
    random_state=42,)
model.fit(x_train,y_train)
# DataFlair - Calculate the accuracy
y_pred=model.predict(x_test)
print("ACCURACY:",accuracy_score(y_test, y_pred)*100)
pd.DataFrame(

    confusion_matrix(y_test, y_pred),

    columns=['Predicted Healthy', 'Predicted Parkinsons'],

    index=['True Healthy', 'True Parkinsons']

)