---
layout: post
title: You Are What You Eat - Customer Segmentation
image: "/posts/WCSS_vs_k.jpg"
tags: [K-Means Clustering, WCSS, Matplotlib, Python]
---

In this post, I use K-Means Clustering to segment a basic customer transactions database by how each customer's food spending is divided across categories. To do this, I used Within-Cluster Sum of Squares (WCSS) to determine an optimal `k` value of clusters and then took customer data to fit the model and segment the clusters.

___

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Data Overview & Preparation](#data-overview)
- [02. Applying WCSS to Determine k](#wcss-application)
- [03. Applying K-Means Clustering](#k-means-clustering)
- [04. Analyzing The Results](#test-results)
- [05. Discussion](#discussion)

___

# Project Overview  <a name="overview-main"></a>
<br>

### Context <a name="overview-context"></a>

In this scenario, a management team is looking to segment their customer grocery database to enable better targeted marketing strategies, personalized promotions, and relevant offers.

Without preliminary analysis, the team holds conflicting views regarding customer purchasing behavior.

* *Volume-Based View:* Some team members believe customers buy across all food departments, with purchase volumes primarily driven by the number of occupants per household.

* *Preference-Based View:* Other team members believe distinct diets and lifestyle preferences drive purchasing behavior, causing customers to selectively shop or avoid specific product categories.

Due to missing relevant data, only the preference-based view will be tested here, with a K-Means clustering approach to identify distinct customer segments and behavioral profiles based on shopping behavior.

<br>

### Actions <a name="overview-actions"></a>

* **Data Ingestion and Filtering**: Import libraries and required `transactions` and `product_areas` table data. Merge the tables and filter out non-food product_area data.
* **Feature Engineering**: Aggregate total sales data by product area for each customer. Pivot the data to have each food category be in its own column. Replace absolute dollar spend with category proportions to segment customers by category preference rather than overall spend.
* **Data Validation Check and Rescaling**: Verify that there are no missing values in the aggregated proportional DataFrame. Apply `MinMaxScaler` across spending proportions to ensure equal weighting across product categories regardless of baseline category item costs.
* **Determine `k`**: Perform Within-Cluster Sum of Squares (WCSS) calculations to use the Elbow Method to determine an appropriate number of clusters, `k`. Plot the results in `Matplotlib` to select `k`
* **Cluster Segmentation and Profiling**: Run `KMeans` algorithm to acquire cluster labels corresponding to each unique customer. Acquire a count of the number of customers in each defined cluster. Create the overall profiling summary for each cluster, showing the mean percentage spend in each category.

<br>

### Results & Discussion <a name="overview-results"></a>

Applying K-Means with k=3 split the customer database into three behavioral groups based on how each customer's spending is divided across food categories:

* **Cluster 1, Balanced Omnivores (73.6% of customers):** Spread spending across all food categories, with the largest share on Meat (37.65%).
* **Cluster 0, Vegetarian Buyers (14.6%):** Spend very little on Meat (2.92%), with most spending on Fruit (39.42%) and Dairy (36.39%).
* **Cluster 2, Plant-Based Vegan Buyers (11.8%):** Spend almost nothing on Meat (0.37%) or Dairy (0.24%), with nearly all spending on Fruit (63.78%) and Vegetables (35.61%).

These results support the **Preference-Based View**. Category spending differs sharply across segments, and the clearest differences come from Meat and Dairy. *Note: The Volume-Based View could not be evaluated due to missing household data.*

**<u>Business Impact & Strategy:</u>**:  
- *Reduce mismatched promotions:* Cut back meat promotions for Clusters 0 and 2, and dairy promotions for Cluster 2 only. Cluster 0 still spends heavily on dairy so its dairy offers are still relevant.
- *Personalize offers:* Send plant-based alternatives to Clusters 0 and 2, and specialty cheese or produce deals to Cluster 0.
- *Rethink meat promotions for Cluster 1:* Meat has the lowest profit margin in the `product_areas` data at 11%. Testing whether meat bundles or promotions add profit may be insightful as to whether or not there's value added in these ads.

___

# Data Overview & Preparation  <a name="data-overview"></a>

First, the relevant company data must be imported as well as the python libraries needed for analysis.

```python
# Import required Python packages
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# import tables
transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
product_areas = pd.read_excel("data/grocery_database.xlsx", sheet_name = "product_areas")
```
<br>
Below is a 5 row sample of the imported **transactions** DataFrame. This DataFrame contains months of customer transaction data, broken down by purchase date and product area:
<br>

| **customer_id** | **transaction_date** | **transaction_id** | **product_area_id** | **num_items** | **sales_cost** |
| --- | --- | --- | --- | --- | --- |
| 4 | 2020-04-01 00:00:00 | 435561454899 | 4 | 6 | 52.18 |
| 4 | 2020-04-01 00:00:00 | 435561454899 | 3 | 1 | 3.45 |
| 4 | 2020-04-01 00:00:00 | 435561454899 | 5 | 1 | 8.57 |
| 4 | 2020-04-01 00:00:00 | 435561454899 | 1 | 5 | 129.76 |
| 5 | 2020-04-01 00:00:00 | 435561670722 | 4 | 15 | 76.74 |

In the **transactions** DataFrame we have the following columns:

-- `customer_id`: Unique identifier for each customer. Non-unique across transactions  
-- `transaction_date`: Date of purchase    
-- `transaction_id`: Unique transaction identifier  
-- `product_area_id`: Categorical identifier for the product department  
-- `num_items`: Quantity of items purchased within that product area  
-- `sales_cost`: Total dollar spend for the product area line item 

<br>
The table below contains the data of the imported **product_areas** DataFrame:

| **product_area_id** | **product_area_name** | **profit_margin** |
| --- | --- | --- |
| 1 | Non-Food | 0.25 |
| 2 | Vegetables | 0.18 |
| 3 | Fruit | 0.14 |
| 4 | Dairy | 0.19 |
| 5 | Meat | 0.11 |

<br>
To get the `product_area_name`s into the **transactions** DataFrame, the two imported datasets are to be merged on the `product_area_id` column. To focus strictly on core grocery product purchasing patterns, this analysis evaluates food department activity only and excludes non-food merchandise, so the "Non-Food" category will be removed from the dataset.
<br>

```python
# merge on product area name
transactions = pd.merge(transactions, product_areas, how = "inner", on = "product_area_id")

# drop the non-food category
transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)
```
<br>
Since we are profiling customers by the product areas they buy from, `sales_cost` must be aggregated for a transaction summary at this level. The table below shows a sample from the data set.
<br>

```python
# aggregate sales at customer level (by product area)
transaction_summary = transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()
```

| **customer_id** | **product_area_name** | **sales_cost** |
| --- | --- | --- |
| 1 | Dairy | 367.74 |
| 1 | Fruit | 276 |
| 1 | Meat | 543.38 |
| 1 | Vegetables | 167.12 |
| 2 | Dairy | 150.63 |

To segment each unique customer into clusters, each customer must have a unique row with the related product area data translated to columns. 

The total spent for each customer across all food product areas is also needed to later calculate the percentage allocation each customer dedicates to each food category. This transformation can be achieved with a pivot table.

```python
# pivot data to place product areas as columns
transaction_summary_pivot = transactions.pivot_table(index = "customer_id",
                                columns = "product_area_name",
                                values = "sales_cost",
                                aggfunc = "sum",
                                fill_value = 0,
                                margins = True,
                                margins_name = "Total").rename_axis(None, axis = 1)
```

<br>

| **customer_id** | **Dairy** | **Fruit** | **Meat** | **Vegetables** | **Total** |
| --- | --- | --- | --- | --- | --- |
| 1 | 367.74 | 276 | 543.38 | 167.12 | 1354.24 |
| 2 | 150.63 | 120.93 | 241.21 | 99.05 | 611.82 |
| 3 | 46.2 | 75.39 | 171.13 | 31.5 | 324.22 |
| 4 | 341.91 | 245.36 | 272.79 | 142.35 | 1002.41 |
| 5 | 694.71 | 815.32 | 1405.19 | 350.1 | 3265.32 |

Each product area's spend is then divided by the customer's total expenses to derive the proportion spent in each food category. This is done to isolate the customer behavior from the total volume of purchased goods. Without this transformation, a customer who spends $3,000 total and a customer who spends $300 total will end up in completely different clusters purely because of their overall spend, even if their spending habits were to be completely identical.

With relative proportions established, the `Total` column is removed prior to clustering.

```python
# Turn sales into proportional of total sales
transaction_summary_pivot = transaction_summary_pivot.div(transaction_summary_pivot["Total"], axis = 0)

# drop the "total" column 
data_for_clustering = transaction_summary_pivot.drop(["Total"], axis = 1)
```

| **customer_id** | **Dairy** | **Fruit** | **Meat** | **Vegetables** |
| --- | --- | --- | --- | --- |
| 1 | 0.271547 | 0.203804 | 0.401244 | 0.123405 |
| 2 | 0.246200 | 0.197656 | 0.394250 | 0.161894 |
| 3 | 0.142496 | 0.232527 | 0.527821 | 0.097156 |
| 4 | 0.341088 | 0.244770 | 0.272134 | 0.142008 |
| 5 | 0.212754 | 0.249691 | 0.430338 | 0.107218 |

Before performing WCSS to optimize K-Means Clustering, a quick check for missing values is critical. Missing values present issues for distance based clustering algorithms because centroids cannot be calculated along dimensions where data is missing.

```python
# check for missing values
print(data_for_clustering.isna().sum())

>>> Dairy         0
    Fruit         0
    Meat          0
    Vegetables    0
```

While our proportioned features are already bounded between 0 and 1, applying `MinMaxScaler` to normalize data across each category is still necessary. 

Because expensive category items like *Meat* account for a larger share of overall spend, customer proportions on Meat vary widely, whereas lower-cost categories like *Fruit* vary within a much tighter range. Without `MinMaxScaler`, Meat's larger numerical spread would pull more weight in Euclidean distance calculations than Fruit.

Rescaling each category's observed range strictly between 0 and 1 prevents categories with larger natural variances from dominating the clustering space. This ensures a customer who is a heavy "Fruit enthusiast" relative to other shoppers receives equal weighting to a "Meat enthusiast."

```python
# normalize data
scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns = data_for_clustering.columns)
```

___

# Applying WCSS to Determine k <a name="wcss-application"></a>

To select the optimal number of clusters 'k', we evaluate the Within-Cluster Sum of Squares (WCSS), also known as inertia. WCSS measures how spread out the points are within each cluster, using the squared Euclidean distance from each point to its cluster centroid.

WCSS decreases as 'k' grows, reaching zero only when every datapoint is in its own cluster. WCSS can be calculated across a range of k values and then plotted against those k values to create a curve. The Elbow Method then looks for the bend in that curve, where the drops in WCSS get much smaller.

To do this, `KMeans` imported from `sklearn.cluster` is used for deriving a list of WCSS values. These values are then plotted with `Matplotlib`.


```python
k_values = list(range(1,10))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_) 

plt.plot(k_values, wcss_list)
plt.title("Within-Cluster Sum of Squares - by k")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()
```
<br>

We specify `n_init`=10 to run the algorithm across 10 random initializations and set `random_state`=42 to ensure reproducible results across runs.

![WCSS by k](/img/posts/WCSS_vs_k.jpg)

<br>
The plot shows a distinct elbow at k=3, where the sharp drop in WCSS flattens significantly. We select k=3 as our optimal cluster count and refit the model.
<br>

___

# Applying K-Means Clustering <a name="k-means-clustering"></a>

The K-Means algorithm is then applied to the scaled data but with k as 3. 

```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(data_for_clustering_scaled)
```
<br>
Once fitted, the `kmeans` object stores the resulting cluster assignments in its `.labels_` attribute.

`kmeans.labels_` functionality can be used to add the series of cluster assignments to the appropriate indices of the original `data_for_clustering` DataFrame. By attaching these labels back to our unscaled proportion data rather than the normalized matrix, we can evaluate spending behavior at a tangible business level.

```python
# add cluster labels to our data
data_for_clustering["cluster"] = kmeans.labels_
```

| **customer_id** | **Dairy** | **Fruit** | **Meat** | **Vegetables** | **cluster** |
| --- | --- | --- | --- | --- | --- |
| 7 | 0 | 0.517055 | 0 | 0.482945 | 2 |
| 8 | 0.293497 | 0.322045 | 0.342859 | 0.0415987 | 1 |
| 9 | 0.283465 | 0.434141 | 0.187015 | 0.0953796 | 0 |
| 10 | 0.277192 | 0.251425 | 0.328992 | 0.142391 | 1 |
| 11 | 0.271369 | 0.35314 | 0.227198 | 0.148293 | 1 |

<br> 

We can then use pandas `.value_counts()` method to see the number of customers assigned to each cluster. We can then calculate the percentage of customers within each cluster.

```python
# check cluster sizes
custs_per_cluster = data_for_clustering["cluster"].value_counts()
>>> 1    641
    0    127
    2    103

# get the percentage of customers in each cluster
cust_count_percentage = round(data_for_clustering["cluster"].value_counts() / data_for_clustering["cluster"].value_counts().sum()*100, 2)
>>> 1    73.59
    0    14.58
    2    11.83
```

Lastly, we'll profile our clusters by grouping the data together at the assigned cluster level and finding the mean percentage of each category spent for each cluster. 

```python
# get the mean proportion of each category at the cluster level
cluster_summary = data_for_clustering.groupby("cluster")[["Dairy", "Fruit", "Meat", "Vegetables"]].mean().reset_index()

# convert proportions to percentages
percent_cols = cluster_summary.columns.tolist()[1:5]
cluster_summary[percent_cols] = round(cluster_summary[percent_cols]*100, 2)

# append customers per cluster and customer percentage per cluster by mapping clusters with 'cluster' column as the key
cluster_summary['# of customers'] = cluster_summary['cluster'].map(custs_per_cluster)
cluster_summary['% of customers'] = cluster_summary['cluster'].map(cust_count_percentage)
```

___

# Analyzing The Results <a name="test-results"></a>

Through the K-Means Clustering, the database was segmented into 3 distinct behavioral groups. The table below shows the output data containing each of these clusters along with the mean percentage that customers within that cluster allocate to shopping in each product area:

| **cluster** | **Dairy** | **Fruit** | **Meat** | **Vegetables** | **# of customers** | **% of customers** | 
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 36.39 | 39.42 | 2.92 | 21.27 | 127 | 14.58 |
| 1 | 22.09 | 26.46 | 37.65 | 13.80 | 641 | 73.59 |
| 2 | 0.24 | 63.78 | 0.37 | 35.61 | 103 | 11.83 |

* **Cluster 0: Vegetarian Buyers (14.6% of shoppers):** Customers in this cluster only spend an average of 2.9% on "Meat" and are much more balanced on their spending in the "Dairy", "Fruit" and "Vegetables" categories. This spending pattern strongly points toward vegetarian dietary habits.

* **Cluster 1: Balanced Omnivores (73.6% of shoppers):** Customers in this cluster tend to exhibit a more uniformly spread shopping behavior across all food categories, representing a standard household buyer with unrestricted dietary needs or preferences.

* **Cluster 2: Plant-Based Vegan Buyers (11.8% of shoppers):** Customers in this cluster allocate only 0.24% of their spending to "Dairy" and 0.37% to "Meat" on average, allocating nearly all their spending to the "Fruit" and "Vegetables" product areas. This specific profile aligns with vegan or strict plant-based diets.

___

# Discussion <a name="discussion"></a>

Through K-Means Clustering, each customer was grouped into one of three behavioral segments based on how their total spending is split across all food categories. 

The biggest differences in spending allocations show up in the "Meat" and "Dairy" categories, which are strong indicators of diet. Clusters 0 and 2 customers spend very little on Meat, fitting a vegetarian or vegan shopping pattern. Cluster 1 spreads its spending across all categories, suggesting standard diets. However, these are inferences from purchases, not confirmed diets. Cluster 0 or 2 customers could be buying their meat from a butcher shop or farm.

The volume-based view was not tested as testing this would require data on household size, which is not included in this dataset.

**<u>Business Impact:</u>** 
Now with each customer in a segment, promotions can be matched to likely interests instead of being sent to all customers. Sending meat promotions to customers who rarely buy meat wastes ad spend and can distract customer attention away from applicable deals. Matching offers to segments should improve both of these issues.

**<u>Marketing Opportunities</u>**:
* *Reduce mismatched promotions:* Cut back meat promotions for Clusters 0 and 2, and dairy promotions for Cluster 2 only. Cluster 0 still spends heavily on dairy so its dairy offers are still relevant.
* *Personalize offers:* Send plant-based alternatives to Clusters 0 and 2, and specialty cheese or produce deals to Cluster 0.
* *Rethink meat promotions for Cluster 1:* Meat has the lowest profit margin in the `product_areas` data at 11%. Testing whether meat bundles or promotions add profit may be insightful as to whether or not there's value added in these ads.