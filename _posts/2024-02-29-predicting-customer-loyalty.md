---
layout: post
title: Predicting Customer Loyalty Using ML
image: "/posts/regression-title-img.png"
tags: [Customer Loyalty, Machine Learning, Regression, Python]
---

Our client, a grocery retailer, hired a market research consultancy to append market level customer loyalty information to the database.  However, only around 50% of the client's customer base could be tagged, thus the other half did not have this information present.  Let's use ML to solve this!

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
    - [Key Definition](#overview-definition)
- [01. Data Overview](#data-overview)
- [02. Modeling Overview](#modeling-overview)
- [03. Linear Regression](#linreg-title)
- [04. Decision Tree](#regtree-title)
- [05. Random Forest](#rf-title)
- [06. Modeling Summary](#modeling-summary)
- [07. Predicting Missing Loyalty Scores](#modeling-predictions)
- [08. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client, a grocery retailer, hired a market research consultancy to append market level customer loyalty information to the database.  However, only around 50% of the client's customer base could be tagged, thus the other half did not have this information present.

The overall aim of this work is to accurately predict the *loyalty score* for those customers who could not be tagged, enabling our client a clear understanding of true customer loyalty, regardless of total spend volume - and allowing for more accurate and relevant customer tracking, targeting, and comms.

To achieve this, we looked to build out a predictive model that will find relationships between customer metrics and *loyalty score* for those customers who were tagged, and use this to predict the loyalty score metric for those who were not.
<br>
<br>
### Actions <a name="overview-actions"></a>

We firstly needed to compile the necessary data from tables in the database, gathering key customer metrics that may help predict *loyalty score*, appending on the dependent variable, and separating out those who did and did not have this dependent variable present.

As we are predicting a numeric output, we tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest
<br>
<br>

### Results <a name="overview-results"></a>

Our testing found that the Random Forest had the highest predictive accuracy.

<br>
**Metric 1: Adjusted R-Squared (Test Set)**

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.754

<br>
**Metric 2: R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853

As the most important outcome for this project was predictive accuracy, rather than explicitly understanding weighted drivers of prediction, we chose the Random Forest as the model to use for making predictions on the customers who were missing the *loyalty score* metric.
<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high - other modeling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty
<br>
<br>
### Key Definition  <a name="overview-definition"></a>

The *loyalty score* metric measures the % of grocery spend (market level) that each customer allocates to the client vs. all of the competitors.  

Example 1: Customer X has a total grocery spend of $100 and all of this is spent with our client. Customer X has a *loyalty score* of 1.0

Example 2: Customer Y has a total grocery spend of $200 but only 20% is spent with our client.  The remaining 80% is spend with competitors.  Customer Y has a *customer loyalty score* of 0.2
<br>
<br>
___

# Data Overview  <a name="data-overview"></a>

We will be predicting the *loyalty_score* metric.  This metric exists (for half of the customer base) in the *loyalty_scores* table of the client database.

The key variables hypothesized to predict the missing loyalty scores will come from the client database, namely the *transactions* table, the *customer_details* table, and the *product_areas* table.

Using pandas in Python, we merged these tables together for all customers, creating a single dataset that we can use for modeling.

```python

# import requried packages
import pandas as pd
import pickle

# import the data
loyalty_scores = pd.read_excel('../data/ABC_Grocery_Data/grocery_database.xlsx', sheet_name='loyalty_scores')
customer_details = pd.read_excel('../data/ABC_Grocery_Data/grocery_database.xlsx', sheet_name='customer_details')
transactions = pd.read_excel('../data/ABC_Grocery_Data/grocery_database.xlsx', sheet_name='transactions')

# merge datasets
df = pd.merge(customer_details, loyalty_scores, on='customer_id', how='left')

# column customer_loyalty_score is the target

# aggregate the transactions data
sales_summary = transactions.groupby('customer_id').agg({'sales_cost': 'sum', 
                                                         'num_items': 'sum',
                                                         'transaction_id': 'count',
                                                         'product_area_id': 'nunique'}).reset_index()

# rename the aggregated columns
sales_summary.columns = ['customer_id', 'total_sales', 'total_items', 'transaction_count', 'product_area_count']

# create average basket amount
sales_summary['average_basket_value'] = sales_summary['total_sales'] / sales_summary['transaction_count']

df = pd.merge(df, sales_summary, on='customer_id', how='inner')

# split the dataframes into rows with loyalty scores and rows without
df_with_loyalty = df[df['customer_loyalty_score'].notnull()]
df_without_loyalty = df[df['customer_loyalty_score'].isnull()]
# drop the target column from the dataframe without loyalty scores
df_without_loyalty = df_without_loyalty.drop('customer_loyalty_score', axis=1)


# save the dataframes to pickle file
with open('./data/df_with_loyalty.pkl', 'wb') as f:
    pickle.dump(df_with_loyalty, f)
    
with open('./data/df_without_loyalty.pkl', 'wb') as f:
    pickle.dump(df_without_loyalty, f)


```
<br>
After this data pre-processing in Python, we have a dataset for modeling that contains the following fields...
<br>
<br>

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| loyalty_score | Dependent | The % of total grocery spend that each customer allocates to ABC Grocery vs. competitors |
| distance_from_store | Independent | "The distance in miles from the customers home address, and the store" |
| gender | Independent | The gender provided by the customer |
| credit_score | Independent | The customers most recent credit score |
| total_sales | Independent | Total spend by the customer in ABC Grocery within the latest 6 months |
| total_items | Independent | Total products purchased by the customer in ABC Grocery within the latest 6 months |
| transaction_count | Independent | Total unique transactions made by the customer in ABC Grocery within the latest 6 months |
| product_area_count | Independent | The number of product areas within ABC Grocery the customers has shopped into within the latest 6 months |
| average_basket_value | Independent | The average spend per transaction for the customer in ABC Grocery within the latest 6 months |

___
<br>
# Modeling Overview

We will build a model that looks to accurately predict the “loyalty_score” metric for those customers that were able to be tagged, based upon the customer metrics listed above.

If that can be achieved, we can use this model to predict the customer loyalty score for the customers that were unable to be tagged by the agency.

As we are predicting a numeric output, we tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest

___
<br>
# Linear Regression <a name="linreg-title"></a>

We utlize the scikit-learn library within Python to model our data using Linear Regression. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="linreg-import"></a>

Since we saved our modeling data as a pickle file, we import it.  We ensure we remove the id column, and we also ensure our data is shuffled.

```python

# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle  
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV


# load the data for modeling
with open('./data/df_with_loyalty.pkl', 'rb') as f:
    df_with_loyalty = pickle.load(f)
    
# drop the customer_id column
df_with_loyalty = df_with_loyalty.drop('customer_id', axis=1)

# shuffle the data
df_with_loyalty = shuffle(df_with_loyalty, random_state=42)

```
<br>
### Data Preprocessing <a name="linreg-preprocessing"></a>

For Linear Regression we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* The effect of outliers
* Encoding categorical variables to numeric form
* Multicollinearity & Feature Selection

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows

```python

# drop missing values since only a few are present
df_with_loyalty = df_with_loyalty.dropna()

```

<br>
##### Outliers

The ability for a Linear Regression model to generalize well across *all* data can be hampered if there are outliers present.  There is no right or wrong way to deal with outliers, but it is always something worth very careful consideration - just because a value is high or low, does not necessarily mean it should not be there!

In this code section, we use **.describe()** from Pandas to investigate the spread of values for each of our predictors.  The results of this can be seen in the table below.

<br>

| **metric** | **distance_from_store** | **credit_score** | **total_sales** | **total_items** | **transaction_count** | **product_area_count** | **average_basket_value** |
|---|---|---|---|---|---|---|---|
| mean | 2.02 | 0.60 | 1846.50 | 278.30 | 44.93 | 4.31 | 36.78 |
| std | 2.57 | 0.10 | 1767.83 | 214.24 | 21.25 | 0.73 | 19.34 |
| min | 0.00 | 0.26 | 45.95 | 10.00 | 4.00 | 2.00 | 9.34 |
| 25% | 0.71 | 0.53 | 942.07 | 201.00 | 41.00 | 4.00 | 22.41 |
| 50% | 1.65 | 0.59 | 1471.49 | 258.50 | 50.00 | 4.00 | 30.37 |
| 75% | 2.91 | 0.66 | 2104.73 | 318.50 | 53.00 | 5.00 | 47.21 |
| max | 44.37 | 0.88 | 9878.76 | 1187.00 | 109.00 | 5.00 | 102.34 |

<br>
Based on this investigation, we see some *max* column values for several variables to be much higher than the *median* value.

This is for columns *distance_from_store*, *total_sales*, and *total_items*

For example, the median *distance_to_store* is 1.645 miles, but the maximum is over 44 miles!

Because of this, we apply some outlier removal in order to facilitate generalization across the full dataset.

We do this using the "boxplot approach" where we remove any rows where the values within those columns are outside of the interquartile range multiplied by 2.

<br>
```python

# create function to remove outliers based on 2.0 IQR
def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 2 * iqr
    upper_bound = q3 + 2 * iqr
    outliers_df = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    df = df.drop(outliers_df.index, axis=0).reset_index(drop=True)
    return df

# drop oultiers
for col in ['distance_from_store', 'total_sales', 'total_items']:
    df_with_loyalty = remove_outliers(df_with_loyalty, col)

```

<br>
##### Split Out Data For Modeling

In the next code block we do two things, we firstly split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training.  In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

<br>
```python

# create X and y
X = df_with_loyalty.drop('customer_loyalty_score', axis=1)
y = df_with_loyalty['customer_loyalty_score']

# split the data from trainig and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

<br>
##### Categorical Predictor Variables

In our dataset, we have one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

The Linear Regression algorithm can't deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *gender* doesn't have any explicit *order* to it, in other words, Male isn't higher or lower than Female and vice versa - one appropriate approach is to apply One Hot Encoding to the categorical column.

One Hot Encoding can be thought of as a way to represent categorical variables as binary vectors, in other words, a set of *new* columns for each categorical value with either a 1 or a 0 saying whether that value is true or not for that observation.  These new columns would go into our model as input variables, and the original column is discarded.

We also drop one of the new columns using the parameter *drop = "first"*.  We do this to avoid the *dummy variable trap* where our newly created encoded columns perfectly predict each other - and we run the risk of breaking the assumption that there is no multicollinearity, a requirement or at least an important consideration for some models, Linear Regression being one of them! Multicollinearity occurs when two or more input variables are *highly* correlated with each other, it is a scenario we attempt to avoid as in short, while it won't neccessarily affect the predictive accuracy of our model, it can make it difficult to trust the statistics around how well the model is performing, and how much each input variable is truly having.

In the code, we also make sure to apply *fit_transform* to the training set, but only *transform* to the test set.  This means the One Hot Encoding logic will *learn and apply* the "rules" from the training data, but only *apply* them to the test data.  This is important in order to avoid *data leakage* where the test set *learns* information about the training data, and means we can't fully trust model performance metrics!

For ease, after we have applied One Hot Encoding, we turn our training and test objects back into Pandas Dataframes, with the column names applied.

<br>
```python

# one hot encode the categorical columns
cat_cols = ['gender']
one_hot = OneHotEncoder(drop='first', sparse_output=False)
one_hot.fit(X_train[cat_cols])
X_train_encoded = one_hot.transform(X_train[cat_cols])
encoded_feat_names = one_hot.get_feature_names_out(cat_cols)
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_feat_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(cat_cols, axis=1, inplace=True)

# encode the test data
X_test_encoded = one_hot.transform(X_test[cat_cols])
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_feat_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(cat_cols, axis=1, inplace=True)

```

<br>
##### Feature Selection

Feature Selection is the process used to select the input variables that are most important to your Machine Learning task.  It can be a very important addition or at least, consideration, in certain scenarios.  The potential benefits of Feature Selection are:

* **Improved Model Accuracy** - eliminating noise can help true relationships stand out
* **Lower Computational Cost** - our model becomes faster to train, and faster to make predictions
* **Explainability** - understanding & explaining outputs for stakeholder & customers becomes much easier

There are many, many ways to apply Feature Selection.  These range from simple methods such as a *Correlation Matrix* showing variable relationships, to *Univariate Testing* which helps us understand statistical relationships between variables, and then to even more powerful approaches like *Recursive Feature Elimination (RFE)* which is an approach that starts with all input variables, and then iteratively removes those with the weakest relationships with the output variable.

For our task we applied a variation of Reursive Feature Elimination called *Recursive Feature Elimination With Cross Validation (RFECV)* where we split the data into many "chunks" and iteratively trains & validates models on each "chunk" seperately.  This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of those models was.  From the suite of model scenarios that are created, the algorithm can determine which provided the best accuracy, and thus can infer the best set of input variables to use!

<br>
```python

# create the Linear Regression model
model = LinearRegression(random_state=42)

# use the feature selection of recursive feature elimination with cross validation
rfecv = RFECV(estimator=model)
fit = rfecv.fit(X_train, y_train)

optimal_feature_count = rfecv.n_features_
print(f'Total number of features: {X_train.shape[1]}')
print(f'Optimal number of features: {optimal_feature_count}')
print(f'Optimal features: {X_train.columns[fit.support_]}')

# create X_train and X_test with the optimal features
X_train = X_train.iloc[:, fit.support_]
X_test = X_test.iloc[:, fit.support_]

```

<br>
The below code then produces a plot that visualizes the cross-validated accuracy with each potential number of features

```python

# get the mean cross validation score
grid_scores = [np.mean(vals) for vals in fit.grid_scores_]


# plot the feature selection
plt.plot(range(1, len(fit.grid_scores_) + 1), grid_scores, marker='o')
plt.ylabel('R2 Score')
plt.xlabel('Number of Features')
plt.title(f'Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(rfecv.grid_scores_.max(), 2)})')
plt.tight_layout()
plt.show()

```

<br>
This creates the below plot, which shows us that the highest cross-validated accuracy (0.8635) is actually when we include all eight of our original input variables.  This is marginally higher than 6 included variables, and 7 included variables.  We will continue on with all 8!

<br>
![alt text](/img/posts/lin-reg-feature-selection-plot.png "Linear Regression Feature Selection Plot")

<br>
### Model Training <a name="linreg-model-training"></a>

Instantiating and training our Linear Regression model is done using the below code

```python

# train the model
model = LinearRegression()
model.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="linreg-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting on new data - we use the trained model object (here called *regressor*) and ask it to predict the *loyalty_score* variable for the test set

```python

# predict the test set
y_pred = model.predict(X_test)

```

<br>
##### Calculate R-Squared

R-Squared is a metric that shows the percentage of variance in our output variable *y* that is being explained by our input variable(s) *x*.  It is a value that ranges between 0 and 1, with a higher value showing a higher level of explained variance.  Another way of explaining this would be to say that, if we had an r-squared score of 0.8 it would suggest that 80% of the variation of our output variable is being explained by our input variables - and something else, or some other variables must account for the other 20%

To calculate r-squared, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python

# use r2 score to evaluate the model
r2 = r2_score(y_test, y_pred)
print(f'R2 score: {r2}')

```

The resulting r-squared score from this is **0.78**

<br>
##### Calculate Cross Validated R-Squared

An even more powerful and reliable way to assess model performance is to utilize Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we break our data into a number of "chunks" and then iteratively train the model on all but one of the "chunks", test the model on the remaining "chunk" until each has had a chance to be the test set.

The result of this is that we are provided a number of test set validation results - and we can take the average of these to give a much more robust & reliable view of how our model will perform on new, un-seen data!

In the code below, we put this into place.  We first specify that we want 4 "chunks" and then we pass in our regressor object, training set, and test set.  We also specify the metric we want to assess with, in this case, we stick with r-squared.

Finally, we take a mean of all four test set results.

```python

# cross validate the model
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
print('Cross Validation Scores:', cv_scores.mean())

```

The mean cross-validated r-squared score from this is **0.853**

<br>
##### Calculate Adjusted R-Squared

When applying Linear Regression with *multiple* input variables, the r-squared metric on it's own *can* end up being an overinflated view of goodness of fit.  This is because each input variable will have an *additive* effect on the overall r-squared score.  In other words, every input variable added to the model *increases* the r-squared value, and *never decreases* it, even if the relationship is by chance.  

**Adjusted R-Squared** is a metric that compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.  It is best practice to use Adjusted R-Squared when assessing the results of a Linear Regression with multiple input variables, as it gives a fairer perception the fit of the data.

```python

# use adjusted r2 score to evaluate the model
n = X_test.shape[0]
p = X_test.shape[1]

# create the adjusted r2 score function
def adjusted_r2_score(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'Adjusted R2 score: {adjusted_r2_score(r2, n, p)}')

```

The resulting *adjusted* r-squared score from this is **0.754** which as expected, is slightly lower than the score we got for r-squared on it's own.

<br>
### Model Summary Statistics <a name="linreg-model-summary"></a>

Although our overall goal for this project is predictive accuracy, rather than an explcit understanding of the relationships of each of the input variables and the output variable, it is always interesting to look at the summary statistics for these.
<br>
```python

# view the coefficients
coefs = pd.DataFrame({'variable': X_train.columns, 'coef': model.coef_})
coefs = coefs.sort_values(by='coef', ascending=False)
print(coefs)

```
<br>
The information from that code block can be found in the table below:
<br>

| **input_variable** | **coefficient** |
|---|---|
| intercept | 0.516 |
| distance_from_store | -0.201 |
| credit_score | -0.028 |
| total_sales | 0.000 |
| total_items | 0.001 |
| transaction_count | -0.005 |
| product_area_count | 0.062 |
| average_basket_value | -0.004 |
| gender_M | -0.013 |

<br>
The coefficient value for each of the input variables, along with that of the intercept would make up the equation for the line of best fit for this particular model (or more accurately, in this case it would be the plane of best fit, as we have multiple input variables).

For each input variable, the coefficient value we see above tells us, with *everything else staying constant* how many units the output variable (loyalty score) would change with a *one unit change* in this particular input variable.

To provide an example of this - in the table above, we can see that the *distance_from_store* input variable has a coefficient value of -0.201.  This is saying that *loyalty_score* decreases by 0.201 (or 20% as loyalty score is a percentage, or at least a decimal value between 0 and 1) for *every additional mile* that a customer lives from the store.  This makes intuitive sense, as customers who live a long way from this store, most likely live near *another* store where they might do some of their shopping as well, whereas customers who live near this store, probably do a greater proportion of their shopping at this store...and hence have a higher loyalty score!

___
<br>
# Decision Tree <a name="regtree-title"></a>

We will again utlize the scikit-learn library within Python to model our data using a Decision Tree. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="regtree-import"></a>

Since we saved our modeling data as a pickle file, we import it.  We ensure we remove the id column, and we also ensure our data is shuffled.

```python

# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle  
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder


# load the data for modeling
with open('./data/df_with_loyalty.pkl', 'rb') as f:
    df_with_loyalty = pickle.load(f)
    
    
# drop the customer_id column
df_with_loyalty = df_with_loyalty.drop('customer_id', axis=1)

# shuffle the data
df_with_loyalty = shuffle(df_with_loyalty, random_state=42)

```
<br>
### Data Preprocessing <a name="regtree-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables - Decision Trees are not, so the required preprocessing here is lighter. We still however will put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows

```python

# drop missing values since only a few are present
df_with_loyalty = df_with_loyalty.dropna()

```

<br>
##### Split Out Data For Modeling

In exactly the same way we did for Linear Regression, in the next code block we do two things, we firstly split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training.  In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

<br>
```python

# create X and y
X = df_with_loyalty.drop('customer_loyalty_score', axis=1)
y = df_with_loyalty['customer_loyalty_score']

# split the data from trainig and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

<br>
##### Categorical Predictor Variables

In our dataset, we have one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

Just like the Linear Regression algorithm, the Decision Tree cannot deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *gender* doesn't have any explicit *order* to it, in other words, Male isn't higher or lower than Female and vice versa - we would again apply One Hot Encoding to the categorical column.

<br>
```python

# one hot encode the categorical columns
cat_cols = ['gender']
one_hot = OneHotEncoder(drop='first', sparse_output=False)
one_hot.fit(X_train[cat_cols])
X_train_encoded = one_hot.transform(X_train[cat_cols])
encoded_feat_names = one_hot.get_feature_names_out(cat_cols)
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_feat_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(cat_cols, axis=1, inplace=True)

# encode the test data
X_test_encoded = one_hot.transform(X_test[cat_cols])
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_feat_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(cat_cols, axis=1, inplace=True)


```

<br>
### Model Training <a name="regtree-model-training"></a>

Instantiating and training our Decision Tree model is done using the below code.  We use the *random_state* parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

```python

# create the Linear Regression model
model = DecisionTreeRegressor(random_state=42)

# train the model
model.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="regtree-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting on new data - we use the trained model object (here called *regressor*) and ask it to predict the *loyalty_score* variable for the test set

```python

# predict the test set
y_pred = model.predict(X_test)

```

<br>
##### Calculate R-Squared

To calculate r-squared, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python

# use r2 score to evaluate the model
r2 = r2_score(y_test, y_pred)
print(f'R2 score: {r2}')

```

The resulting r-squared score from this is **0.898**

<br>
##### Calculate Cross Validated R-Squared

As we did when testing Linear Regression, we will again utilize Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we break our data into a number of "chunks" and then iteratively train the model on all but one of the "chunks", test the model on the remaining "chunk" until each has had a chance to be the test set.

The result of this is that we are provided a number of test set validation results - and we can take the average of these to give a much more robust & reliable view of how our model will perform on new, un-seen data!

In the code below, we put this into place.  We again specify that we want 4 "chunks" and then we pass in our regressor object, training set, and test set.  We also specify the metric we want to assess with, in this case, we stick with r-squared.

Finally, we take a mean of all four test set results.

```python

# cross validate the model
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
print('Cross Validation Scores:', cv_scores.mean())

```

The mean cross-validated r-squared score from this is **0.871** which is slighter higher than we saw for Linear Regression.

<br>
##### Calculate Adjusted R-Squared

Just like we did with Linear Regression, we will also calculate the *Adjusted R-Squared* which compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.

```python

# use adjusted r2 score to evaluate the model
n = X_test.shape[0]
p = X_test.shape[1]

# create the adjusted r2 score function
def adjusted_r2_score(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'Adjusted R2 score: {adjusted_r2_score(r2, n, p)}')

```

The resulting *adjusted* r-squared score from this is **0.887** which as expected, is slightly lower than the score we got for r-squared on it's own.

<br>
### Decision Tree Regularization <a name="regtree-model-regularization"></a>

Decision Tree's can be prone to over-fitting, in other words, without any limits on their splitting, they will end up learning the training data perfectly.  We would much prefer our model to have a more *generalized* set of rules, as this will be more robust & reliable when making predictions on *new* data.

One effective method of avoiding this over-fitting, is to apply a *max depth* to the Decision Tree, meaning we only allow it to split the data a certain number of times before it is required to stop.

Unfortunately, we don't necessarily know the *best* number of splits to use for this - so below we will loop over a variety of values and assess which gives us the best predictive performance!

<br>
```python

# find best max depth
max_depths = range(1, 9)
r2_scores = []
for depth in max_depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    
# find the best max depth
best_r2_score = max(r2_scores) 
best_max_depth = max_depths[r2_scores.index(best_r2_score)]   
    
# plot the max depth vs r2 score
plt.plot(max_depths, r2_scores)
plt.scatter(best_max_depth, best_r2_score, color='red', marker='x', label='Best Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('R2 Score')
plt.title(f'Max Depth vs R2 Score \nBest Max Depth: {best_max_depth} \nBest R2 Score: {round(best_r2_score, 3)}')
plt.tight_layout()
plt.show()

```
<br>
That code gives us the below plot - which visualizes the results!

<br>
![alt text](/img/posts/regression-tree-max-depth-plot.png "Decision Tree Max Depth Plot")

<br>
In the plot we can see that the *maximum* classification accuracy on the test set is found when applying a *max_depth* value of 7.  However, we lose very little accuracy back to a value of 4, but this would result in a simpler model, that generalized even better on new data.  We make the executive decision to re-train our Decision Tree with a maximum depth of 4!

<br>
### Visualize Our Decision Tree <a name="regtree-visualize"></a>

To see the decisions that have been made in the (re-fitted) tree, we can use the plot_tree functionality that we imported from scikit-learn.  To do this, we use the below code:

<br>
```python

## while the max depth of 8 gives the best r-squared, the score is farily flat starting at max depth of 4
# use max depth of 4
model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R2 score: {r2}') # performance barely changes with less chance of overfitting

# plot the model
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, rounded=True, feature_names=X_train.columns, fontsize=16)

```
<br>
That code gives us the below plot:

<br>
![alt text](/img/posts/regression-tree-nodes-plot.png "Decision Tree Max Depth Plot")

<br>
This is a very powerful visual, and one that can be shown to stakeholders in the business to ensure they understand exactly what is driving the predictions.

One interesting thing to note is that the *very first split* appears to be using the variable *distance from store* so it would seem that this is a very important variable when it comes to predicting loyalty!

___
<br>
# Random Forest <a name="rf-title"></a>

We will again utlize the scikit-learn library within Python to model our data using a Random Forest. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="rf-import"></a>

Again, since we saved our modeling data as a pickle file, we import it.  We ensure we remove the id column, and we also ensure our data is shuffled.

```python

# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle  
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# load the data for modeling
with open('./data/df_with_loyalty.pkl', 'rb') as f:
    df_with_loyalty = pickle.load(f)
    
    
# drop the customer_id column
df_with_loyalty = df_with_loyalty.drop('customer_id', axis=1)

# shuffle the data
df_with_loyalty = shuffle(df_with_loyalty, random_state=42)

```
<br>
### Data Preprocessing <a name="rf-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables - Random Forests, just like Decision Trees, are not, so the required preprocessing here is lighter. We still however will put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows

```python

# drop missing values since only a few are present
df_with_loyalty = df_with_loyalty.dropna()

```

<br>
##### Split Out Data For Modeling

In exactly the same way we did for Linear Regression, in the next code block we do two things, we firstly split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training.  In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

<br>
```python

# create X and y
X = df_with_loyalty.drop('customer_loyalty_score', axis=1)
y = df_with_loyalty['customer_loyalty_score']

# split the data from trainig and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

<br>
##### Categorical Predictor Variables

In our dataset, we have one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

Just like the Linear Regression algorithm, Random Forests cannot deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *gender* doesn't have any explicit *order* to it, in other words, Male isn't higher or lower than Female and vice versa - we would again apply One Hot Encoding to the categorical column.

<br>
```python

# one hot encode the categorical columns
cat_cols = ['gender']
one_hot = OneHotEncoder(drop='first', sparse_output=False)
one_hot.fit(X_train[cat_cols])
X_train_encoded = one_hot.transform(X_train[cat_cols])
encoded_feat_names = one_hot.get_feature_names_out(cat_cols)
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_feat_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(cat_cols, axis=1, inplace=True)

# encode the test data
X_test_encoded = one_hot.transform(X_test[cat_cols])
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_feat_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(cat_cols, axis=1, inplace=True)

```

<br>
### Model Training <a name="rf-model-training"></a>

Instantiating and training our Random Forest model is done using the below code.  We use the *random_state* parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

We leave the other parameters at their default values, meaning that we will just be building 100 Decision Trees in this Random Forest.

```python

# create the Linear Regression model
model = RandomForestRegressor(random_state=42)

# train the model
model.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="rf-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting on new data - we use the trained model object (here called *regressor*) and ask it to predict the *loyalty_score* variable for the test set

```python

# predict the test set
y_pred = model.predict(X_test)

```

<br>
##### Calculate R-Squared

To calculate r-squared, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python

# use r2 score to evaluate the model
r2 = r2_score(y_test, y_pred)
print(f'R2 score: {r2}')

```

The resulting r-squared score from this is **0.957** - higher than both Linear Regression & the Decision Tree.

<br>
##### Calculate Cross Validated R-Squared

As we did when testing Linear Regression & our Decision Tree, we will again utilize Cross Validation (for more info on how this works, please refer to the Linear Regression section above)

```python

# cross validate the model
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
print('Cross Validation Scores:', cv_scores.mean())

```

The mean cross-validated r-squared score from this is **0.923** which agian is higher than we saw for both Linear Regression & our Decision Tree.

<br>
##### Calculate Adjusted R-Squared

Just like we did with Linear Regression & our Decision Tree, we will also calculate the *Adjusted R-Squared* which compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.

```python

# use adjusted r2 score to evaluate the model
n = X_test.shape[0]
p = X_test.shape[1]

# create the adjusted r2 score function
def adjusted_r2_score(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'Adjusted R2 score: {adjusted_r2_score(r2, n, p)}')

```

The resulting *adjusted* r-squared score from this is **0.955** which as expected, is slightly lower than the score we got for r-squared on it's own - but again higher than for our other models.

<br>
### Feature Importance <a name="rf-model-feature-importance"></a>

In our Linear Regression model, to understand the relationships between input variables and our ouput variable, loyalty score, we examined the coefficients.  With our Decision Tree we looked at what the earlier splits were.  These allowed us some insight into which input variables were having the most impact.

Random Forests are an ensemble model, made up of many, many Decision Trees, each of which is different due to the randomness of the data being provided, and the random selection of input variables available at each potential split point.

Because of this, we end up with a powerful and robust model, but because of the random or different nature of all these Decision trees - the model gives us a unique insight into how important each of our input variables are to the overall model.  

As we’re using random samples of data, and input variables for each Decision Tree - there are many scenarios where certain input variables are being held back and this enables us a way to compare how accurate the models predictions are if that variable is or isn’t present.

So, at a high level, in a Random Forest we can measure *importance* by asking *How much would accuracy decrease if a specific input variable was removed or randomized?*

If this decrease in performance, or accuracy, is large, then we’d deem that input variable to be quite important, and if we see only a small decrease in accuracy, then we’d conclude that the variable is of less importance.

At a high level, there are two common ways to tackle this.  The first, often just called **Feature Importance** is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the Mean Squared Error (for a Regression problem) was before the split was made, and compare this to the Mean Squared Error after the split was made.  We can take the *average* of these improvements across all Decision Trees in the Random Forest to get a score that tells us *how much better* we’re making the model by using that input variable.

If we do this for *each* of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model!

The other approach, often called **Permutation Importance** cleverly uses some data that has gone *unused* at when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping")

These observations that were not randomly selected for each Decision Tree are known as *Out of Bag* observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the *Out of Bag* observations are gathered and then passed through.  Once all of these observations have been run through the Decision Tree, we obtain an accuracy score for these predictions, which in the case of a regression problem could be Mean Squared Error or r-squared.

In order to understand the *importance*, we *randomize* the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score.  The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

*Permutation Importance* is often preferred over *Feature Importance* which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.

Let's put them both in place, and plot the results...

<br>
```python

# view feature importance
feature_importance = model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)

# plot the feature importance
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest Regression')
plt.tight_layout()
plt.show()

# view the permutation importance
perm_importance = permutation_importance(model, X_test, y_test, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Permutation Importance - Random Forest Regression')
plt.tight_layout()
plt.show()

```
<br>
That code gives us the below plots - the first being for *Feature Importance* and the second for *Permutation Importance*!

<br>
![alt text](/img/posts/rf-regression-feature-importance.png "Random Forest Feature Importance Plot")
<br>
<br>
![alt text](/img/posts/rf-regression-permutation-importance.png "Random Forest Permutation Importance Plot")

<br>
The overall story from both approaches is very similar, in that by far, the most important or impactful input variable is *distance_from_store* which is the same insights we derived when assessing our Linear Regression & Decision Tree models.

There are slight differences in the order or "importance" for the remaining variables but overall they have provided similar findings.

___
<br>
# Modeling Summary  <a name="modeling-summary"></a>

The most important outcome for this project was predictive accuracy, rather than explicitly understanding the drivers of prediction. Based upon this, we chose the model that performed the best when predicted on the test set - the Random Forest.

<br>
**Metric 1: Adjusted R-Squared (Test Set)**

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.754

<br>
**Metric 2: R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853

<br>
Even though we were not specifically interested in the drivers of prediction, it was interesting to see across all three modeling approaches, that the input variable with the biggest impact on the prediction was *distance_from_store* rather than variables such as *total sales*.  This is interesting information for the business, so discovering this as we went was worthwhile.

<br>
# Predicting Missing Loyalty Scores <a name="modeling-predictions"></a>

We have selected the model to use (Random Forest) and now we need to make the *loyalty_score* predictions for those customers that the market research consultancy were unable to tag.

We cannot just pass the data for these customers into the model, as is - we need to ensure the data is in exactly the same format as what was used when training the model.

In the following code, we will

* Import the required packages for preprocessing
* Import the data for those customers who are missing a *loyalty_score* value
* Import our model object & any preprocessing artifacts
* Drop columns that were not used when training the model (customer_id)
* Drop rows with missing values
* Apply One Hot Encoding to the gender column (using transform)
* Make the predictions using .predict()

<br>
```python

# import required packages
import pandas as pd
import pickle

# load in the data to be predicted
with open('./data/df_without_loyalty.pkl', 'rb') as f:
    df_to_predict = pickle.load(f)
    
# import the model and one hot encoder
with open('./models/random_forest_regression_loyalty.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('./models/one_hot_encoder_loyalty.pkl', 'rb') as f:
    one_hot = pickle.load(f)
    
# drop the customer_id column
df_to_predict = df_to_predict.drop('customer_id', axis=1)

# drop any missing values rows
df_to_predict = df_to_predict.dropna()

# one hot encode the categorical columns
cat_cols = ['gender']
transformed_array = one_hot.transform(df_to_predict[cat_cols])
encoded_feat_names = one_hot.get_feature_names_out(cat_cols)
trans_df = pd.DataFrame(transformed_array, columns=encoded_feat_names)
df_to_predict = pd.concat([df_to_predict.reset_index(drop=True), trans_df.reset_index(drop=True)], axis=1)
df_to_predict.drop(cat_cols, axis=1, inplace=True)

# make predictions
predictions = model.predict(df_to_predict)

```
<br>
Just like that, we have made our *loyalty_score* predictions for these missing customers.  Due to the impressive metrics on the test set, we can be reasonably confident with these scores.  This extra customer information will ensure our client can undertake more accurate and relevant customer tracking, targeting, and comms.

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

While predictive accuracy was relatively high - other modeling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

We could even look to tune the hyperparameters of the Random Forest, notably regularization parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty