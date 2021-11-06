---
layout: post
title: How Can a Company Save Money by Targeting the Right Customers Using Machine Learning
image: "/posts/markus_winkler_unsplash.jpg"
tags: [Python, Random Forests, Classification Model]
---

In this example we are using Machine Learning algorithm called Random Forests to find out which customers have high probability to sign up for the Delivery Club so the company targets only those customers and reduces the costs of the future campaigns. 

---

Data first! In order to be able to use a ML argorithm we should have collected data on cutromer's behavour for at least some customers for a curtain ammount of time. In our example we have the data for 870 customers collected through 3 months period.  

##### Step 1: Import required packages
The software used is Python (Anaconda, Spider) so we don't need to install anything, just import.

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
The sample data is already saved as pickle file.

```python
data_for_model=pickle.load(open('Data/abc_classification_modelling.p', 'rb'))
```
##### Step 3: Drop Uneccessary columns
We don't need the customer_id column.

```python
data_for_model.drop(['customer_id'], axis=1, inplace=True)
```
##### Step 3: Shuffle Data

```python
data_for_model=shuffle(data_for_model, random_state=42)
```
##### Step 4: Check Class Balance
As we can see on the output below, 69% of our data belongs to class 0 and 31% to class 1. This is not perfectly balanced data, but we should not have any problems.

```python
data_for_model['signup_flag'].value_counts()

OUTPUT:
0    593
1    267

# In order to see the % use (normalize=True)

data_for_model['signup_flag'].value_counts(normalize=True)

OUTPUT:
0    0.689535
1    0.310465
```
##### Step 5: Dealing with Missing Values

```python
data_for_model.isna().sum()

OUTPUT:
signup_flag             0
distance_from_store     5
gender                  5
credit_score            8
total_sales             0
total_items             0
transaction_count       0
product_area_count      0
average_basket_value    0

# We have only 18 missing values in the data set with the sample size 870, we shouldn't have a problem if we drop them all. We are now left with 847 rows.

data_for_model.dropna(how='any', inplace=True)
```


###### OUTPUT:
![output](/img/posts/outpu.png "output")

#### We will be answering 7 questions
##### Question 1: Is there a relationship between advertising and sales?
