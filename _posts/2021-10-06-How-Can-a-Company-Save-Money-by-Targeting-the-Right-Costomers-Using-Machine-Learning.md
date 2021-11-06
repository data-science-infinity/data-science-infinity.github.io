---
layout: post
title: How Can a Company Save Money by Targeting the Right Customers Using Machine Learning
image: "/posts/markus_winkler_unsplash.jpg"
tags: [Python, Random Forests, Classification Model]
---

In this example we are using Machine Learning algorithm called Random Forests to find out which customers have high probability to sign up for the Delivery Club so the company targets only those customers and reduces the costs of the future campaigns. 

---

Data first! In order to be able to use a ML argorithm we should have collected data on cutromer's behavour for at least some customers for a curtain ammount of time. In our example we have the data for 870 customers collected through 3 months period. The software used is Python (Anaconda, Spider). The sample data is already saved as pickle file.

##### Step 1: Import required packages

```python
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
```
##### Step 2: Import Data

```python
data_for_model=pickle.load(open('Data/abc_classification_modelling.p', 'rb'))
```
##### Step 3: Drop Uneccessary columns

```python
data_for_model.drop(['customer_id'], axis=1, inplace=True)
```
##### Step 3: Shuffle Data

```python
data_for_model=shuffle(data_for_model, random_state=42)
```
##### Step 4: Check Class Balance

```python
data_for_model['signup_flag'].value_counts()

OUTPUT:
0    593
1    267

# In order to see the % use (normalize=True)

```python
data_for_model['signup_flag'].value_counts(normalize=True)

OUTPUT:
0    0.689535
1    0.310465
```

###### OUTPUT:
![output](/img/posts/outpu.png "output")

#### We will be answering 7 questions
##### Question 1: Is there a relationship between advertising and sales?
