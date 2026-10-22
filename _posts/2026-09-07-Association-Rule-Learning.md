---
layout: post
title: Understanding Alcohol Product Relationships Using Association Rule Learning
image: "/posts/association-rules.png"
tags: [Association Rule Learning, Python]
---

In this post, I'll be walking us through using Association Rule Learning to analyze the transactional relationships and dependencies between products in the alcohol section of a grocery store.

___

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
- [01. Data Overview](#data-overview)
- [02. Apriori Overview](#apriori-overview)
- [03. Data Preparation](#apriori-data-prep)
- [04. Applying The Apriori Algorithm](#apriori-fit)
- [05. Interpreting The Results](#apriori-results)
- [06. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

<br> 

### Context <a name="overview-context"></a>

Our client is looking to reorganize the alcohol section within their store. Customers are often complaining that they can't find the products they want and are also wanting recommendations about which other products to try. The marketing team also sees this as an opportunity to start running "bundled" promotions, but needs guidance with selecting which products to bundle together.

They have provided us a sample of 3,567 alcohol transactions. Our task is to see if we can find solutions or insights that might help the business address these endeavors.

<br>

### Actions <a name="overview-actions"></a>

Association Rule Learning, specifically *Apriori* will be used to examine and analyze the strength of relationships between different alcohol products within the transactional data.

* Import `pandas` and the `apyori` package in Python, which contains all of the required functionality for this analysis.
* Import the sample data and process it into the *list of lists* format that the Apriori algorithm takes as an input.
* Apply the `Apriori` algorithm to acquire the following relationship metrics: 
    * Support
    * Confidence
    * Expected Confidence
    * Lift
* Interpret the Results

<br>

### Results <a name="overview-results"></a>

The Apriori analysis identified 132 product pairs, which produced 166 directional rules. The strongest associations involved gift products (Wine Gifts with Beer/Lager Gifts and Spirits & Fortified), boxed wine products (Red Wine Bxes & 25Cl with White Boxes), and small-size products (Small Sizes White with Small Sizes Red).

These patterns suggest a few possible actions for the client. Gift products could be grouped together rather than by product type, while boxed and small size products could be considered for bundles or promotions. All of these rule findings provide low Support values (0.003 to 0.025), so they should be treated as hypotheses to verify with the client rather than settled recommendations.

**<u>Next Steps:</u>** Build an Apriori pipeline with K-Fold validation for the business to use in other store areas and across store areas.

___

# Data Overview  <a name="data-overview"></a>

The initial dataset contains 3,567 transactions, each of which shows the unique alcohol products that were purchased in that transaction.

We'll import `Pandas` and the `apriori` algorithm from the `apyori` library, then import the raw data into Python.
<br>
<br>

```python
# import required Python packages
import pandas as pd
from apyori import apriori

# import the sample data
alcohol_transactions = pd.read_csv("data/sample_data_apriori.csv")
```
<br>

A sample of 10 transactions is shown below. Each row is one transaction, and each product<#> column holds one product from that transaction. Rows with fewer products have blank cells for the rightmost columns.
<br>
<br>

| **transaction_id** | **product1** | **product2** | **product3** | **product4** | **product5** | **…** |
|---|---|---|---|---|---|---|
| 1 | Premium Lager | Iberia | … |  |  | ... |
| 2 | Sparkling | Premium Lager | Premium Cider | Own Label | Italy White | … |
| 3 | Small Sizes White | Small Sizes Red | Sherry Spanish | No/Low Alc Cider | Cooking Wine | … |
| 4 | White Uk | Sherry Spanish | Port | Italian White | Italian Red | … |
| 5 | Premium Lager | Over-Ice Cider | French White South | French Rose | Cocktails/Liqueurs | … |
| 6 | Kosher Red | … |  |  |  | ... |
| 7 | Own Label | Italy White | Australian Red | … |  | ... |
| 8 | Brandy/Cognac | … |  |  |  | ... |
| 9 | Small Sizes White | Bottled Ale | … |  |  | ... |
| 10 | White Uk | Spirits Mixers | Sparkling | German | Australian Red | … |

<br>

In the table, *Transaction 1* (where `transaction_id` is 1) contained two products: Premium Lager and Iberia. As there were only two products in this transaction, the remaining columns are blank.

*Transaction 2* (where `transaction_id` is 2) contained nine products, which were not all included in the data snippet above. The first nine columns after the `transaction_id` for this row are populated, followed by blank values in the rest of the product<#> columns.

For the sample data, the maximum number of unique products in a transaction was 45, meaning the table of data had a total of 46 columns (45 for `product<#>`s + `transaction_id`).

The `apriori` method that we are using does not take DataFrames as inputs; it instead expects data passed in as a *list of lists*. The code and logic on how to modify the input can be found in the [03. Data Preparation](#apriori-data-prep) section.

___


# Apriori Overview  <a name="apriori-overview"></a>

*Association Rule Learning* is an approach that discovers the strength of relationships between different data points. It is commonly used to determine which products are frequently or infrequently purchased together. This determination provides useful information that a business can use to optimize marketing strategies such as:

* *Product arrangement/placement*: making the customer journey more efficient
* *Relevant Product Recommendations*: based on the *customers who bought product A also bought product B* principle
* *Bundled Discounts*: determining which products should or should not be put together

A commonly used algorithm for Association Rule Learning is **Apriori**. In Apriori there are four key metrics: *Support*, *Confidence*, *Expected Confidence* and *Lift*. Each of these metrics helps us understand items and their relationship with other items.

<br>

#### Support

**Support** simply tells us the percentage of all transactions that contain *both* Item A and Item B. To calculate this we’d count up the transactions that include both items and divide this by the total number of transactions.

You can think of *Support* as a baseline metric that helps us understand how common or popular purchasing this particular pair of items together is.

<br>

#### Confidence

**Confidence** looks more explicitly at the *relationship* between the two items. It asks *of all transactions that included 'item A', what proportion also included 'item B'?*

Here we are counting up the number of transactions that contained *both items A and B* and then rather than dividing by *all transactions* like we did for Support, we divide by the *total number of transactions that contained item A*.

A high score for Confidence could indicate a strong product relationship, but not always. When one of the items is very popular we can get an inflated score. To help recognize this, we can look at the last two metrics, *Expected Confidence* and *Lift*.

<br>

#### Expected Confidence

**Expected Confidence** is simply the percentage of *all transactions* that *contained item B*.

This is important as it provides an indication of what the Confidence *would be* if there were no relationship between the items. We can use Expected Confidence, along with Confidence to calculate our final and most powerful metric, *Lift*.

<br>

#### Lift

**Lift** is the factor by which the *Confidence exceeds the Expected Confidence*. In other words, Lift tells us *how much more likely item B is to be purchased when item A is purchased than would be expected by chance* after adjusting for how popular item B is.

We calculate Lift by dividing `Confidence` by `Expected Confidence`.

A Lift score *greater than 1* indicates that items A and B appear together *more often* than expected, while a Lift score *less than 1* indicates that items A and B appear together *less often* than expected. A Lift score **equal to 1** indicates that items A and B are independent and have no association.

<br>

#### In Practice

When running all possible combinations of paired items through the `apriori` algorithm, sorting all pairings by lift score can identify which pairs of products have the strongest and weakest relationships. This information can strategically guide business and marketing decisions regarding product layout, recommendations for customers or promotions.

> **Important Consideration**: Item pairs with a *high Lift* and *low Support* should be interpreted with caution. If we sort all pairs by Lift, the top result may look like a very strong relationship, but it could be happening by chance because the pair is rare. Always check the Support metric alongside Lift.

___

# Data Preparation  <a name="apriori-data-prep"></a>

As mentioned in the [01. Data Overview](#data-overview) section above, the `apyori` library requires a *list of lists* as an input, so the tabular data must be modified to accommodate the `apriori` method.  

In the code below, we:
* Remove the `transaction_id` index column
* Iterate over each row in the imported DataFrame, appending each `transaction` to a list, and appending those transactions to a master list, `transactions_list`
* Print out the first 10 lists from the `transactions_list`

<br>

```python
# drop ID column
alcohol_transactions.drop("transaction_id", axis = 1, inplace = True)

# modify data for apriori algorithm
transactions_list = []
for index, row in alcohol_transactions.iterrows():
    transaction = list(row.dropna())
    transactions_list.append(transaction)
    
# print out first 10 lists from transactions_list
print(transactions_list[:10])

[['Premium Lager', 'Iberia'],
 ['Sparkling', 'Premium Lager', 'Premium Cider', 'Own Label', 'Italy White', 'Italian White', 'Italian Red', 'French Red', 'Bottled Ale'],
 ['Small Sizes White', 'Small Sizes Red', 'Sherry Spanish', 'No/Low Alc Cider', 'Cooking Wine', 'Cocktails/Liqueurs', 'Bottled Ale'],
 ['White Uk', 'Sherry Spanish', 'Port', 'Italian White', 'Italian Red'],
 ['Premium Lager', 'Over-Ice Cider', 'French White South', 'French Rose', 'Cocktails/Liqueurs', 'Bottled Ale'],
 ['Kosher Red'],
 ['Own Label', 'Italy White', 'Australian Red'],
 ['Brandy/Cognac'],
 ['Small Sizes White', 'Bottled Ale'],
 ['White Uk', 'Spirits Mixers', 'Sparkling', 'German', 'Australian Red', 'American Red']]
```

<br>

The print statement demonstrates that each transaction row from the original raw DataFrame is now captured as a `transaction` list, contained within a master list, `transactions_list`.

With the data in this format, the Apriori algorithm can be applied.

___

# Applying The Apriori Algorithm <a name="apriori-fit"></a>

In the code below we apply the `apriori` algorithm from the `apyori` library.

This algorithm allows us to specify the association rules for the analysis. For this test, the input rule parameters are set to:

* A minimum *Support* of 0.003 to eliminate very rare product sets
* A minimum *Confidence* of 0.2
* A minimum *Lift* of 3 to ensure we're only focusing on product sets with strong associations
* A maximum length of 2, meaning we're only focusing on product *pairs* rather than larger sets

The output from the Apriori algorithm is in the form of a generator; the code will convert this to a list as this is easier to manipulate and analyze.

```python
# apply the apriori algorithm and specify required parameters
apriori_rules = apriori(transactions_list,
                        min_support = 0.003,
                        min_confidence = 0.2,
                        min_lift = 3,
                        max_length = 2)

# convert the output to a list
apriori_rules = list(apriori_rules)

# check number of rules 
num_pairs = len(apriori_rules)
print(num_pairs)
>> 132
```

Based on the parameters set when applying the algorithm, there were 132 product pairs identified and stored as their own elements in the `apriori_rules` list output. Running `apriori_rules[0]` prints out the first element from the list along with the necessary analysis metrics for that specific item pairing. 

```python
# print out the first element
apriori_rules[0]

>> RelationRecord(items=frozenset({'America White', 'American Rose'}), support=0.020745724698626296, ordered_statistics=[OrderedStatistic(items_base=frozenset({'American Rose'}), items_add=frozenset({'America White'}), confidence=0.5323741007194245, lift=3.997849299507762)])
```

It is extremely important to note that the `ordered_statistics` in each pair may contain more than one direction between products. For example, the rule could contain information on the relationship between item A -> B and item B -> A. For each pair, both directions will have the same support and lift, however, the confidence value will differ. Because of this, we want to loop through each ordered_statistic within each generated pair.

To get this information into a readable table for all elements in the `apriori_rules` list, the following code snippet extracts the key elements and translates the data into a Pandas DataFrame.

```python
rows = []
for rule in apriori_rules:
    for stat in rule.ordered_statistics:
        rows.append({
            "product1": list(stat.items_base)[0],
            "product2": list(stat.items_add)[0],
            "support": rule.support,
            "confidence": stat.confidence,
            "lift": stat.lift,
        })

apriori_rules_df = pd.DataFrame(rows)

# check number of directional pairs
directional_pairs = apriori_rules_df.shape[0]
print(directional_pairs)
>> 166
```

<br>
When considering the directional relationships between all the rules, we end up with 166 directional product pairs. A sample of the created DataFrame can be seen below, where we have the two products in each row along with the three key metrics; *support*, *confidence*, and *lift*.
<br>


| **product1** | **product2** | **support** | **confidence** | **lift** |
|---|---|---|---|---|
| American Rose | America White | 0.021 | 0.532 | 3.998 |
| America White | American White | 0.054 | 0.408 | 3.597 |
| American White | America White | 0.054 | 0.479 | 3.597 |
| Australian Rose | America White | 0.005 | 0.486 | 3.653 |
| Low Alcohol A.C | America White | 0.003 | 0.462 | 3.466 |


___

# Interpreting The Results <a name="apriori-results"></a>

With the Apriori data in a DataFrame, we can now easily identify the product pairs with the *strongest* relationships by sorting our DataFrame entries by the `lift` column in descending order.

```python
# sort pairs by descending Lift
apriori_rules_df.sort_values(by = "lift", ascending = False, inplace = True)
```

<br>
In the table below, we can see the ten strongest product relationships based on highest Lift value:


| **product1** | **product2** | **support** | **confidence** | **lift** |
|---|---|---|---|---|
| Wine Gifts | Beer/Lager Gifts | 0.004 | 0.314 | 10.173 |
| Spirits & Fortified | Beer/Lager Gifts | 0.013 | 0.305 | 9.897 |
| Beer/Lager Gifts | Spirits & Fortified | 0.013 | 0.427 | 9.897 |
| Wine Gifts | Spirits & Fortified | 0.006 | 0.412 | 9.537 |
| Red Wine Bxes & 25Cl | White Boxes | 0.015 | 0.474 | 9.344 |
| White Boxes | Red Wine Bxes & 25Cl | 0.015 | 0.304 | 9.344 |
| French White Rhone | French Red | 0.003 | 0.480 | 8.691 |
| Small Sizeswhite Oth | Small Sizes White | 0.005 | 0.559 | 8.340 |
| Small Sizes White | Small Sizes Red | 0.025 | 0.372 | 7.258 |
| Small Sizes Red | Small Sizes White | 0.025 | 0.486 | 7.258 |

<br>
Some pairs appear in both directions, such as *Red Wine Bxes* and *25Cl and White Boxes*. Both rows share the same support and lift, but their confidence values differ. Lift is the same in both directions because it accounts for the popularity of both items. 

Confidence is calculated from the count of the "if" item, and Red Wine Bxes appear in fewer transactions than White Boxes. As a result, 47% of Red Wine Bxes buyers also bought White Boxes, while only 30% of White Boxes buyers also bought Red Wine Bxes. 

<br>
At a first glance, the following observations can be made:

* ***Gifts:*** The strongest links include *Gift* products. Managers could consider grouping gifts together in the store rather than by product type, but further analysis should be done to consider the seasonal impact of *Gift* sales.
* **Wines:** Wines seem to be commonly bought together in general, with strong notable relationships such as "Small Sizes" Variety wines and "Boxs"/"Boxes" wine items. It may be worth considering doing bundles, promotions, or discounts grouping boxed wines or small sized wines together.

All item pairs in the table have low Support values, between 0.003 and 0.025, meaning each appears in roughly 12 to 89 of the 3,567 transactions. Because Lift is calculated from small transaction counts, a few new purchases could noticeably change the Lift values. For this reason, these results should be treated as hypotheses to verify with the client, not as settled findings. Pairs with more supporting transactions are better evidenced. For example, *Small Sizes Red* and *Small Sizes White* (at 0.025 Support) are better evidenced than *French White Rhone* and *French Red* (at 0.003 Support).

___

# Growth & Next Steps <a name="growth-next-steps"></a>

This analysis found notable associations between alcohol products by looking at which pairs had the highest Lift values. Because the Support values are low, these findings should be treated as hypotheses to test rather than a prescription for action. Even so, the results are still useful for the business. Knowing which relationships exist lets us investigate why they occur, which can inform strategic and impactful marketing decisions.

Expanding this Apriori approach to other areas of the business could uncover more marketing opportunities, such as alcohol paired with soft drinks or snacks. I'd propose integrating this analysis into a pipeline that Category Managers and analysts can use to identify product relationships within and across areas, providing more insight for sales decisions. The pipeline should also test whether the strongest pairs hold up across several subsets of the data using k-fold  and adjust the results accordingly.