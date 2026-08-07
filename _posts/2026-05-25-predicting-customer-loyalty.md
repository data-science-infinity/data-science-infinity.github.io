---
layout: post
title: Predicting Customer Loyalty Using ML
image: "/posts/regression-title-img.png"
tags: [Customer Loyalty, Machine Learning, Regression, Python]
---

My client, a grocery retailer, hired a market research consultancy to append market level customer loyalty information to the database.  However, only around 50% of the client's customer base could be tagged, thus the other half did not have this information present.  I'll use ML to solve this!

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
    - [Key Definition](#overview-definition)
- [01. Data Overview](#data-overview)
- [02. Modeling Overview](#modelling-overview)
- [03. Linear Regression](#linreg-title)
- [04. Decision Tree](#regtree-title)
- [05. Random Forest](#rf-title)
- [06. Modeling Summary](#modelling-summary)
- [07. Predicting Missing Loyalty Scores](#modeling-predictions)
- [08. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

My client, a grocery retailer, hired a market research consultancy to append market level customer loyalty information to the database.  However, only around 50% of the client's customer base could be tagged, thus the other half did not have this information present.

The overall aim of this work is to accurately predict the *loyalty score* for those customers who could not be tagged, enabling my client a clear understanding of true customer loyalty, regardless of total spend volume - and allowing for more accurate and relevant customer tracking, targeting, and comms.

To achieve this, I looked to build out a predictive model that will find relationships between customer metrics and *loyalty score* for those customers who were tagged, and use this to predict the loyalty score metric for those who were not.
<br>
<br>
### Actions <a name="overview-actions"></a>

I firstly needed to compile the necessary data from tables in the database, gathering key customer metrics that may help predict *loyalty score*, appending on the dependent variable, and separating out those who did and did not have this dependent variable present.

As I am predicting a numeric output, I tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest
<br>
<br>

### Results <a name="overview-results"></a>

My testing found that the Random Forest had the highest predictive accuracy.

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

As the most important outcome for this project was predictive accuracy, rather than explicitly understanding weighted drivers of prediction, I chose the Random Forest as the model to use for making predictions on the customers who were missing the *loyalty score* metric.
<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

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

We will be predicting the *loyalty_score* metric.  This metric exists (for approximately half of the customer base) in the *loyalty_scores* table of the client database.

The key variables hypothesized to predict the missing loyalty scores will come from the client database, namely the *transactions* table, the *customer_details* table, and the *product_areas* table.

Using pandas in Python, I merged these tables together for all customers, creating a single dataset that we can use for modeling.

```python

# import required packages
import pandas as pd
import pickle

# import required data tables
loyalty_scores = pd.read_excel("data/grocery_data.xlsx", sheet_name = "loyalty_scores")
customer_details = pd.read_excel("data/grocery_data.xlsx", sheet_name = "customer_details")
transactions = pd.read_excel("data/grocery_data.xlsx", sheet_name = "transactions")
# loyalty_scores returns 400 rows, customer_details returns 870 rows

# merge loyalty score data and customer details data, at customer level.  customer_details is the base.
# Later we'll split this into 2 datasets for training and testing
data_for_regression = pd.merge(customer_details, loyalty_scores, how = "left", on = "customer_id")

# aggregate sales data from transactions table.  Create 5 sales metrics.
sales_summary = transactions.groupby("customer_id").agg({"sales_cost" : "sum",
                                                         "num_items" : "sum",
                                                         "transaction_id" : "count",
                                                         "product_area_id" : "nunique"}).reset_index()

# rename columns for clarity
sales_summary.columns = ["customer_id", "total_sales", "total_items", "transaction_count", "product_area_count"]

# engineer an average basket value column for each customer
sales_summary["average_basket_value"] = sales_summary["total_sales"] / sales_summary["transaction_count"]

# Overwrite data_for_regression and merge the sales summary with the overall customer data
data_for_regression = pd.merge(data_for_regression, sales_summary, how = "inner", on = "customer_id")
# Our new "data_for_regression" has 10 columns including customer_id

# split out data for modeling (loyalty score is present or not)
regression_modeling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]

# split out data for scoring post-modeling (loyalty score is missing)
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]

# for scoring set, drop the loyalty score column (as it is blank/redundant)
regression_scoring.drop(["customer_loyalty_score"], axis = 1, inplace = True)

# save our datasets for future use
pickle.dump(regression_modeling, open("data/customer_loyalty_modeling.p", "wb"))
pickle.dump(regression_scoring, open("data/customer_loyalty_scoring.p", "wb"))

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

I will build a model that looks to accurately predict the “loyalty_score” metric for those customers that were able to be tagged, based upon the customer metrics listed above.

If that can be achieved, I can use this model to predict the customer loyalty score for the customers that were unable to be tagged by the agency.

As I am predicting a numeric output, we tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest

___
<br>
# Linear Regression <a name="linreg-title"></a>

I utilized the scikit-learn library within Python to model our data using Linear Regression. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="linreg-import"></a>

Since I saved our modeling data as a pickle file, I'll import it.  I will ensure the id column is removed, and I'll also ensure the data is shuffled.

```python

# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

# import modeling data
data_for_model = pickle.load(open("data/customer_loyalty_modeling.p", "rb"))

# drop uneccessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```
<br>
### Data Preprocessing <a name="linreg-preprocessing"></a>

For Linear Regression there are certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* The effect of outliers (for Linear)
* Encoding categorical variables to numeric form
* Multicollinearity & Feature Selection

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) I'll just drop those rows

```python

# remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)

```

<br>
##### Outliers

The ability for a Linear Regression model to generalize well across *all* data can be hampered if there are outliers present.  There is no right or wrong way to deal with outliers, but it is always something worth very careful consideration - just because a value is high or low, does not necessarily mean it should not be there!

In this code section, I'll use **.describe()** from Pandas to investigate the spread of values for each of our predictors.  The results of this can be seen in the table below.

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
Based on this investigation, some *max* column values for several variables are much higher than the *median* value.

This is for columns *distance_from_store*, *total_sales*, and *total_items*

For example, the median *distance_to_store* is 1.645 miles, but the maximum is over 44 miles!

Because of this, I'll apply some outlier removal in order to facilitate generalization across the full dataset.

I'll do this using the "boxplot approach" where I'll remove any rows where the values within those columns are outside of the interquartile range multiplied by 2.

<br>
```python

outlier_investigation = data_for_model.describe()
outlier_columns = ["distance_from_store", "total_sales", "total_items"]

# boxplot approach
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)

```

<br>
##### Split Out Data For Modeling

In the next code block I'll do two things, firstly split the data into an **X** object which contains only the predictor variables, and a **y** object that contains only the dependent variable.

Once I have done this, I split the data into "training" and "test" sets to ensure I can fairly validate the accuracy of the predictions on data that was not used in training.  In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

<br>
```python

# split data into X and y objects for modeling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>
##### Categorical Predictor Variables

In the dataset, there is one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

The Linear Regression algorithm can't deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *gender* doesn't have any explicit *order* to it, in other words, Male isn't higher or lower than Female and vice versa - one appropriate approach is to apply "One Hot Encoding" to the categorical column.

One Hot Encoding can be thought of as a way to represent categorical variables as binary vectors, in other words, a set of *new* columns for each categorical value with either a 1 or a 0 saying whether that value is true or not for that observation.  These new columns would go into our model as input variables, and the original column is discarded.

I also drop one of the new columns using the parameter *drop = "first"*.  This is done to avoid the *dummy variable trap* where the newly created encoded columns perfectly predict each other, running the risk of breaking the assumption that there is no multicollinearity - a requirement or at least an important consideration for some models, Linear Regression being one of them! Multicollinearity occurs when two or more input variables are *highly* correlated with each other.  It is a scenario to attempt to avoid.  While it won't necessarily affect the predictive accuracy of our model, it can make it difficult to trust the statistics around how well the model is performing, and how much each input variable is truly having.

In the code, I also make sure to apply *fit_transform* to the training set, but only *transform* to the test set.  This means the One Hot Encoding logic will *learn and apply* the "rules" from the training data, but only *apply* them to the test data.  This is important in order to avoid *data leakage* where the test set *learns* information about the training data, and means we can't fully trust model performance metrics!

For ease, after applying One Hot Encoding, training and test objects can be turned back into Pandas Dataframes, with the column names applied.

<br>
```python

# list of categorical variables that need encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns (array)
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to Pandas DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)

# Create a new DataFrame by concatonating the original to the encoded DataFrame
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
# Drop original input 2 and input 3 variables
X_train.drop(categorical_vars, axis = 1, inplace = True)

# Test data
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
# Drop old "gender"
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>
##### Feature Selection using RFE (Recursive Feature Elimination)

Feature Selection is the process used to select the input variables that are most important to your Machine Learning task.  It can be a very important addition or at least, consideration, in certain scenarios.  The potential benefits of Feature Selection are:

* **Improved Model Accuracy** - eliminating noise can help true relationships stand out
* **Lower Computational Cost** - our model becomes faster to train, and faster to make predictions
* **Explainability** - understanding & explaining outputs for stakeholders & customers becomes much easier

There are multiple ways to apply Feature Selection.  These range from simple methods such as a *Correlation Matrix* showing variable relationships, to *Univariate Testing* which helps us understand statistical relationships between variables, and then to even more powerful approaches like *Recursive Feature Elimination (RFE)* which is an approach that starts with all input variables and then iteratively removes those with the weakest relationships to the output variable.

For this task, I applied a variation of Recursive Feature Elimination called *Recursive Feature Elimination With Cross Validation (RFECV)* where the data is split into many "chunks" and iteratively trains & validates models on each "chunk" separately.  This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of those models was.  From the suite of model scenarios that are created, the algorithm can determine which provided the best accuracy, and thus can infer the best set of input variables to use!

<br>
```python

# instantiate RFECV & the model type to be utilized
regressor = LinearRegression()
feature_selector = RFECV(regressor)

# fit RFECV onto our training & test data
fit = feature_selector.fit(X_train,y_train)

# extract & print the optimal number of features
optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

# limit our training & test sets to only include the selected variables
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

```

<br>
The below code then produces a plot that visualizes the cross-validated accuracy with each potential number of features

```python

plt.style.use('seaborn-poster')
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()

```

<br>
This creates the below plot, which shows that the highest cross-validated accuracy (0.8635) is actually when all eight of the original input variables are included.  This is marginally higher than 6 included variables, and 7 included variables.  We will continue on with all 8!

<br>
![alt text](/img/posts/lin-reg-feature-selection-plot.png "Linear Regression Feature Selection Plot")

<br>
### Model Training <a name="linreg-model-training"></a>

Instantiating and training the Linear Regression model is done using the below code

```python

# instantiate our model object
regressor = LinearRegression()

# fit our model using our training & test sets
regressor.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="linreg-model-assessment"></a>

##### Predict On The Test Set

To assess how well the model is predicting on new data - the trained model object (here called *regressor*) is used and asked to predict the *loyalty_score* variable for the test set

```python

# predict on the test set
y_pred = regressor.predict(X_test)

```

<br>
##### Calculate R-Squared

R-Squared is a metric that shows the percentage of variance in our output variable *y* that is being explained by our input variable(s) *x*.  It is a value that ranges between 0 and 1, with a higher value showing a higher level of explained variance.  Another way of explaining this would be to say that, if we had an r-squared score of 0.8 it would suggest that 80% of the variation of the output variable is being explained by the input variables - and something else, or some other variables must account for the other 20%

To calculate r-squared, I use the following code where I pass in our *predicted* outputs for the test set (y_pred) as well as the *actual* outputs for the test set (y_test)

```python

# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)

```

The resulting r-squared score from this is **0.78**

<br>
##### Calculate Cross Validated R-Squared

An even more powerful and reliable way to assess model performance is to utilize Cross Validation.

Instead of simply dividing the data into a single training set, and a single test set, with Cross Validation, the data can be broken into a number of "chunks" and then iteratively train the model on all but one of the "chunks", test the model on the remaining "chunk" until each has had a chance to be the test set.

The result of this is that a number of test set validation results is provided - and the average of these is calculated to give a much more robust & reliable view of how the model will perform on new, un-seen data!

In the code below, this is put into place.  First, 4 "chunks" is specified, and then we pass in the regressor object, the training set, and the test set.  Also specified is the metric with which to assess, in this case, r-squared.

Finally, a mean of all four test set results is calculated.

```python

# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()

```

The mean cross-validated r-squared score from this is **0.853**

<br>
##### Calculate Adjusted R-Squared

When applying Linear Regression with *multiple* input variables, the r-squared metric on it's own *can* end up being an overinflated view of goodness of fit.  This is because each input variable will have an *additive* effect on the overall r-squared score.  In other words, every input variable added to the model *increases* the r-squared value, and *never decreases* it, even if the relationship is by chance.  

**Adjusted R-Squared** is a metric that compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.  It is best practice to use Adjusted R-Squared when assessing the results of a Linear Regression with multiple input variables, as it gives a more fair perception the fit of the data.

```python

# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

```

The resulting *adjusted* r-squared score from this is **0.754** which as expected, is slightly lower than the score we got for r-squared on it's own.

<br>
### Model Summary Statistics <a name="linreg-model-summary"></a>

Although the overall goal for this project is predictive accuracy, rather than an explicit understanding of the relationships of each of the input variables and the output variable, it is always interesting to look at the summary statistics for these.
<br>
```python

# extract model coefficients and create a DataFrame
coefficients = pd.DataFrame(regressor.coef_)
# Make DataFrame more useful by adding names
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names,coefficients], axis = 1)
summary_stats.columns = ["input_variable", "coefficient"]

# Values in the DataFrame will make up the values going into the equation for the "line of best fit" or
# technically the "plane of best fit"

# extract model intercept
regressor.intercept_

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

For each input variable, the coefficient value above tells us, with *everything else staying constant* , how many units the output variable (loyalty score) would change with a *one unit change* in this particular input variable.

To provide an example of this using the table above, we can see that the *distance_from_store* input variable has a coefficient value of -0.201.  This is saying that *loyalty_score* decreases by 0.201 (or 20% as loyalty score is a percentage, or at least a decimal value between 0 and 1) for *every additional mile* that a customer lives from the store.  This makes intuitive sense, as customers who live a long way from this store most likely live near *another* store where they might do some of their shopping as well.  Whereas, customers who live near this store, probably do a greater proportion of their shopping at "this" store and hence have a higher loyalty score!

___
<br>
# Decision Tree <a name="regtree-title"></a>

We will again utilize the scikit-learn library within Python to model our data using a Decision Tree. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="regtree-import"></a>

Since the modeling data was saved as a pickle file, it is imported.  Next, the id column is removed and our data is shuffled.

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

# import modeling data
data_for_model = pickle.load(open("data/customer_loyalty_modeling.p", "rb"))

# drop uneccessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```
<br>
### Data Preprocessing <a name="regtree-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers and highly correlated input variables, Decision Trees are not so the required preprocessing here is lighter. We still; however, will put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows

```python

# remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)

```

<br>
##### Split Out Data For Modeling

In exactly the same way we did for Linear Regression, in the next code block two things are done, firstly the data is split into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once this is complete, the data is split into training and test sets to ensure fairl validation and the accuracy of the predictions on data that was not used in training.  In this case, 80% of the data is allocated for training, and the remaining 20% for validation.

<br>
```python

# split data into X and y objects for modeling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>
##### Categorical Predictor Variables

In our dataset, we have one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

Just like the Linear Regression algorithm, the Decision Tree cannot deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *gender* doesn't have any explicit *order* to it; Male isn't higher or lower than Female and vice versa, One Hot Encoding is again applied to the categorical column.

<br>
```python

# list of categorical variables that need encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to Pandas DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>
### Model Training <a name="regtree-model-training"></a>

Instantiating and training the Decision Tree model is done using the below code.  The *random_state* parameter is used to ensure reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

```python

# instantiate our model object
regressor = DecisionTreeRegressor(random_state = 42)

# fit our model using our training & test sets
regressor.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="regtree-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting on new data, we use the trained model object, *regressor*, and ask it to predict the *loyalty_score* variable for the test set

```python

# predict on the test set
y_pred = regressor.predict(X_test)

```

<br>
##### Calculate R-Squared

To calculate r-squared, the following code is used where the *predicted* outputs are passed in for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python

# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)

```

The resulting r-squared score from this is **0.898**

<br>
##### Calculate Cross Validated R-Squared

As we did when testing Linear Regression, Cross Validation is again used.

Instead of simply dividing the data into a single training set and a single test set, with Cross Validation the data is broken into a number of "chunks" and then iteratively used to train the model on all but one of the "chunks".  The model isn't tested on the remaining "chunk" until each has had a chance to be the test set.

The result of this is that a number of test set validation results is provided and the average of these can be calculated to give a much more robust & reliable view of how our model will perform on new, un-seen data!

In the code below, this into place.  Again, 4 "chunks" is specified in the KFold parameter "n_splits" and then the regressor object, training set, and test set are passed in to it.  Also specified is the metric with which we want to assess the model, in this case, r-squared is again used.

Finally, a mean of all four test set results is calculated.

```python

# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()

```

The mean cross-validated r-squared score from this is **0.871** which is slighter higher than calculated for Linear Regression.

<br>
##### Calculate Adjusted R-Squared

Just like Linear Regression, the *Adjusted R-Squared* is calculated compensating for the addition of input variables, and only increaseing if the variable improves the model above what would be obtained by probability.

```python

# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

```

The resulting *adjusted* r-squared score from this is **0.887**, which as expected is slightly lower than the score we got for r-squared on it's own.

<br>
### Decision Tree Regularization <a name="regtree-model-regularisation"></a>

Decision Tree's can be prone to over-fitting, in other words, without any limits on their splitting, they will end up learning the training data perfectly.  A more *generalized* set of rules is preferred as this will be more robust & reliable when making predictions on "new" data.

One effective method of avoiding this over-fitting, is to apply a *max depth* to the Decision Tree, meaning it only allows it to split the data a certain number of times before it is required to stop.

Unfortunately, the *best* number of splits to use for this is unknown, so below we will loop over a variety of values and assess which yields the best predictive performance!

<br>
```python

# finding the best "max_depth"

# set up the range for search and an empty list to which to append the accuracy scores
max_depth_list = list(range(1,9))
accuracy_scores = []

# loop through each possible depth, train and validate model, append test set accuracy
for depth in max_depth_list:
    
    regressor = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    accuracy_scores.append(accuracy)
    
# store "max accuracy" and "optimal depth"    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# plot "accuracy" by "max depth"
plt.plot(max_depth_list,accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

```
<br>
That code gives yields the below plot helping to visualize the results!

<br>
![alt text](/img/posts/regression-tree-max-depth-plot.png "Decision Tree Max Depth Plot")

<br>
In the plot, *maximum* classification accuracy on the test set is found when applying a *max_depth* value of 7.  However, we lose very little accuracy using a value of 4 and this results in a simpler model that would generalize even better on new data.  Considering how little accuracy is gained using "7", a "max_depth" value of 4 will be used to re-train our Decision Tree.

<br>
### Visualize Our Decision Tree <a name="regtree-visualise"></a>

To see the decisions that have been made in the (re-fitted) tree, the "plot_tree" functionality that we imported from scikit-learn can be used.  To do this, the below code is used:

<br>
```python

# re-fit the model using max depth of 4
regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)
regressor.fit(X_train, y_train)

# plot the nodes of the decision tree
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = list(X.columns),
                 filled = True,
                 rounded = True,
                 fontsize = 16)

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

Again, the scikit-learn library within Python is used to model our data using a Random Forest. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="rf-import"></a>

Again, since the modeling data was saved as a pickle file, it will be imported.  The id column is removed  and the data is shuffled.

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

# import modeling data
data_for_model = pickle.load(open("data/customer_loyalty_modeling.p", "rb"))

# drop unnecessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```
<br>
### Data Preprocessing <a name="rf-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers and highly correlated input variables, Random Forests, just like Decision Trees, are not.  So, the required preprocessing here is lighter. 
However, the logic for the following is still necessary:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) those rows will just be removed

```python

# remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)

```

<br>
##### Split Out Data For Modeling

In exactly the same way done for Linear Regression, in the next code block two things are accomplished.  Firstly, the data is split into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once this is done, the data is split into training and test sets to ensure fair validation of the predictive accuracy on data that was not used in training.  In this case, we have allocated 80% of the data for training and the remaining 20% for validation.

<br>
```python

# split data into X and y objects for modeling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>
##### Categorical Predictor Variables

In the dataset, there is one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

Just like the Linear Regression algorithm, Random Forests cannot deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *gender* doesn't have any explicit *order* to it; Male isn't higher or lower than Female and vice versa, One Hot Encoding is applied to the categorical column.

<br>
```python

# list of categorical variables that need encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to Pandas DataFrame and drop original categorical data column
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>
### Model Training <a name="rf-model-training"></a>

Instantiating and training our Random Forest model is done using the below code.  The *random_state* parameter is used to ensure reproducible results, helping to understand any improvements in performance with changes to model hyperparameters.

Other parameters retain their default values, specifically meaning that the default of 100 Decision Trees will be built in this Random Forest.

```python

# instantiate our model object
regressor = RandomForestRegressor(random_state = 42)

# fit our model using our training & test sets
regressor.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="rf-model-assessment"></a>

##### Predict On The Test Set

To assess how well the model is predicting on new data, the trained model object, *regressor*, is used and asked to predict the *loyalty_score* variable for the test set

```python

# predict on the test set
y_pred = regressor.predict(X_test)

```

<br>
##### Calculate R-Squared

To calculate r-squared, the following code is used where *predicted* outputs for the test set (y_pred) is passed into it as well as the *actual* outputs for the test set (y_test)

```python

# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)

```

The resulting r-squared score from this is **0.957** - higher than both Linear Regression & the Decision Tree.

<br>
##### Calculate Cross Validated R-Squared

As done when testing Linear Regression & our Decision Tree, Cross Validation is utilized (for more info on how this works, please refer to the Linear Regression section above).

```python

# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()

```

The mean cross-validated r-squared score from this is **0.923** which agian is higher than calculated for both Linear Regression & the Decision Tree models.

<br>
##### Calculate Adjusted R-Squared

Just like we did with Linear Regression & Decision Tree, the *Adjusted R-Squared* is calculated compensating for the addition of input variables, and only increasing if the variable improves the model above what would be obtained by probability.

```python

# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

```

The resulting *adjusted* r-squared score from this is **0.955**, which as expected is slightly lower than the score obtained for r-squared on it's own but again higher than the other models.

<br>
### Feature Importance <a name="rf-model-feature-importance"></a>

To aid in understanding the relationships between input variables and our output variable "loyalty score", in the Linear Regression model, the coefficients were examined.  With the Decision Tree, where data was split was analzed allowing us some insight into which input variables were having the most impact.

Random Forests is an "ensemble model" made up of many Decision Trees, each of which is different due to the randomness of the data being provided and the random selection of input variables available at each potential split point.

Because of this, a much more powerful and robust model is created.  And, because of the random or different nature of all these Decision trees, the model gives us a unique insight into how important each of our input variables are to the overall model.  

As random samples of data and input variables for each Decision Tree is utilized, there are many scenarios where certain input variables are being held back enabling a way to compare how accurate the models predictions are if that variable is or isn’t present.

So, at a high level in a Random Forest, we can measure *importance* by asking *How much would accuracy decrease if a specific input variable was removed or randomized?*

If this decrease in performance or accuracy is large, then we’d deem that input variable to be quite important.  And, if we see only a small decrease in accuracy, then we’d conclude that the variable is of less importance.

At a high level, there are two common ways to tackle this.  The first, often just called **Feature Importance**, is where all nodes in the Decision Trees of the forest where a particular input variable is used to split the data are found.  Subsequently. they are assessed as to what the Mean Squared Error (for a Regression problem) was before the split was made and compared to the Mean Squared Error after the split was made.  The *average* of these improvements is calculated across all Decision Trees in the Random Forest to obtain a score that indicates *how much better* the model functions by using that particular input variable.

If this is done for *each* input variable, these scores can be compared to help understand which is adding the most value to the predictive power of the model.

The other approach, often called **Permutation Importance**, cleverly uses some data that has gone *unused* when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping").

These observations (data) that were not randomly selected for each Decision Tree are known as *Out of Bag* observations and can be used for "testing" the accuracy of each particular Decision Tree.

For each Decision Tree, all of the *Out of Bag* observations are gathered and then passed through.  Once all of these observations have been run through the Decision Tree, we obtain an accuracy score for these predictions.  In the case of a regression problem this score could be Mean Squared Error or r-squared.

In order to understand the *importance*, the values are *randomized* within one of the input variables.  This is a process that essentially destroys any relationship that might exist between that input variable and the output variable. The updated data is run through the Decision Tree again, obtaining a second accuracy score.  The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

*Permutation Importance* is often preferred over *Feature Importance* which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.

Put both in place, and plot the results...

<br>
```python

# calculate feature importance
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names,feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

# plot feature importance
plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# calculate permutation importance
result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)
permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names,permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable","permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

# plot permutation importance
plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

```
<br>
The above code creates the "plots" below.  The first is *Feature Importance* and the second is *Permutation Importance*.

<br>
![alt text](/img/posts/rf-regression-feature-importance.png "Random Forest Feature Importance Plot")
<br>
<br>
![alt text](/img/posts/rf-regression-permutation-importance.png "Random Forest Permutation Importance Plot")

<br>
The overall story from both approaches is very similar, in that by far the most important or impactful input variable is *distance_from_store* which is the same insight derived when assessing the Linear Regression & Decision Tree models.

There are slight differences in the order or "importance" for the remaining variables but overall they have provided similar findings.

___
<br>
# Modeling Summary  <a name="modelling-summary"></a>

The most important outcome for this project was predictive accuracy rather than explicitly understanding the drivers of prediction. Based upon this, the model that performed the best when predicted on the test set was the Random Forest model.

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
Even though the drivers of prediction were not of specific interest in this project, it was interesting to see across all three modeling approaches that the input variable with the biggest impact on the prediction was *distance_from_store* rather than variables such as *total sales*.  This is potentially valuable information for the business, so discovering this was worthwhile.

<br>
# Predicting Missing Loyalty Scores <a name="modelling-predictions"></a>

Finally in completeling our task for ABC Grocery, the Random Forest model was selected to ultimately predict the *loyalty_score* for those customers that the market research consultancy were unable to tag.

As before, we cannot just pass the data for these customers into the model "as is".  The data needs to be preparred to be in the "exact" the same format as what was used when training the model.

Tasks to be performed in the following code:

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

# import customers for scoring
to_be_scored = pickle.load("data/abc_regression_scoring.p", "rb")

# import model and model objects
regressor = pickle.load(open("data/random_forest_regression_model.p", "rb"))
one_hot_encoder = pickle.load(open("data/random_forest_regression_ohe.p", "rb"))

# drop unused columns
to_be_scored.drop(["customer_id"], axis = 1, inplace = True)

# drop missing values
to_be_scored.dropna(how = "any", inplace = True)

# apply one hot encoding (transform only)
categorical_vars = ["gender"]
encoder_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars])
encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)
encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)
to_be_scored = pd.concat([to_be_scored.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis = 1)
to_be_scored.drop(categorical_vars, axis = 1, inplace = True)

# make our predictions!
loyalty_predictions = regressor.predict(to_be_scored)

```
<br>
*loyalty_score* predictions have now been made for these missing customers.  Considering the impressive metrics on the test set, there is reasonable confidence in these scores.  This extra customer information will ensure our client can undertake more accurate and relevant customer tracking, targeting, and communications.

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

While predictive accuracy was relatively high, other modeling approaches could be tested to see if even more accuracy could be gained, especially those somewhat similar to Random Forest (i.e. XGBoost, LightGBM).

Additionally, the hyperparameters of the Random Forest could also be tuned.  Notably, regularization parameters such as "tree depth" as well as potentially training on a higher number of Decision Trees (n_estimators) in the Random Forest could be used.

From a data point of view, further variables could be collected and further feature engineering could be undertaken ensuring as much useful information is available for predicting customer loyalty.
