# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## DESIGN STEPS

### STEP 1:
Load the csv file and then use the preprocessing steps to clean the data

### STEP 2:
Split the data to training and testing

### STEP 3:
Train the data and then predict using Tensorflow

## PROGRAM

```python3
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt
df=pd.read_csv("customers.csv")

df.head(10)
df_processed=df.dropna(axis=0)


categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)
enc

data = df_processed.copy()
data[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(data[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])
     
     
le = LabelEncoder()
data['Segmentation'] = le.fit_transform(data['Segmentation'])
     
data= data.drop('ID',axis=1)
data = data.drop('Var_1',axis=1)

# Calculate the correlation matrix
corr = data.corr()

# Plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
     
sns.pairplot(data)

sns.distplot(data['Age'])


plt.figure(figsize=(2,6))
sns.countplot(data['Family_Size'])

plt.figure(figsize=(10,6))
sns.boxplot(x='Family_Size',y='Age',data=data)
     
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=data)

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=data)

data.describe()

X=data[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y1 = data[['Segmentation']].values

one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)

y = one_hot_enc.transform(y1).toarray()

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               random_state=50)
     
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

ai_brain = tf.keras.Sequential([tf.keras.layers.Input(shape=(8,)),
                           tf.keras.layers.Dense(256,activation="relu"),
                           tf.keras.layers.Dense(128,activation="relu"),
                           tf.keras.layers.Dense(64,activation="relu"),
                           tf.keras.layers.Dense(32,activation="relu"),
                           tf.keras.layers.Dense(16,activation="relu"),
                           tf.keras.layers.Dense(8,activation="relu"),
                           tf.keras.layers.Dense(4,activation="softmax")])
                          
ai_brain.compile(optimizer='adagrad',
                 loss='categorical_crossentropy',
                 metrics=['accuracy']) 

ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=200,batch_size=32,
             validation_data=(X_test_scaled,y_test),
             #callbacks=[early_stop]
             )
             
metrics = pd.DataFrame(ai_brain.history.history)
metrics.head()

metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
 
y_test_truevalue = np.argmax(y_test,axis=1)

print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

```

## Dataset Information

![image](https://user-images.githubusercontent.com/75234588/228124706-41438108-5600-4198-92f4-934ad1bd4fc3.png)

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75234588/228128835-4454c334-e210-4dad-93f8-7035c2069e70.png)

![image](https://user-images.githubusercontent.com/75234588/228128951-0eb6d9e0-71b7-46df-96e7-ff53b8d16451.png)

### Classification Report

![image](https://user-images.githubusercontent.com/75234588/228129058-6a8e4a24-b7ba-42b0-812a-eb302faad1df.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/75234588/228129102-c8cbce28-3c24-4e43-8cfa-6fc0db8c491e.png)

## RESULT

Thus a Neural Network Classification Model is created and executed successfully
