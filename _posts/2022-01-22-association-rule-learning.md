---
layout: post
title: Understanding Alcohol Product Relationships Using Association Rule Learning
image: "/posts/association-rules-title-img.png"
tags: [Association Rule Learning, Python]
---

In this project we use Association Rule Learning to analyse the transactional relationships & dependencies between products in the alcohol section of a grocery store.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Apriori Overview](#apriori-overview)
- [03. Data Preparation](#apriori-data-prep)
- [04. Applying The Apriori Algorithm](#apriori-fit)
- [05. Interpreting The Results](#apriori-results)
- [06. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client is looking to re-jig the alcohol section within their store.  Customers are often complaining that they can't find the products they want, and are also wanting recommendations about which other products to try.  On top of this, their marketing team would like to start running "bundled" promotions as this has worked well in other areas of the store - but need guidance with selecting which products to put together.

They have provided us a sample of 3,500 alcohol transactions - our task is fairly open - to see if we can find solutions or insights that might help the business address the aforementioned problems!

<br>
<br>
### Actions <a name="overview-actions"></a>

Based upon the tasks at hand - we apply Association Rule Learning, specifically *Apriori* to examine & analyse the strength of relationship between different products within the transactional data.

We firstly installed the apyori package, which contains all of the required functionality for this task.

We then needed to bring in the sample data, and get it into the right format for the Apriori algorithm to deal with.

From there we apply the Apriori algorithm to provide us with several different relationship metrics, namely:

* Support
* Confidence
* Expected Confidence
* Lift

These metrics examine product relationships in different ways, so we utilise each to put forward ideas that address each of the tasks at hand.  You can read more about these metrics, and the Apriori algorithm in the relevant section below.

<br>
<br>

### Results <a name="overview-results"></a>

Interestingly, the strongest relationship existed between two products labelled as "gifts" - this is useful information for the category managers as they may want to ensure that gift products are available in one section of the aisle, rather than existing in their respective product types.

We also saw some strong relationships between French wines, and other French wines - which again is extremely useful for category managers who are thinking about the best way to lay out the products - having sections by country rather than necessarily by type might make it easier for customers to find what they are after.

Another interesting association is between products labelled "small".  At this point, we don't know exactly what that means - but it is certainly something to take back to the client as they may be able to make more sense of it, and turn it into an actionable insight!

We propose to also build a "search engine" for category managers where they can look-up products by keyword in the product association table.

As an example - we search for any products that associate strongly with "New Zealand" products. There appeared to be *some* relationship between New Zealand wines and other New Zealand wines, but what was also interesting was that New Zealand wines seemed to be more associated with French & South American wines than they were with Australian Wines.

New Zealand & Australia are often grouped together, but in terms of wine this wouldn't make sense - perhaps because of the difference climates the wines are very different and thus it wouldn't make sense to group wines by geographical proximity, but by preference instead.  This is only a hypothesis for now - we will need to take this back to the client and get their category experts to help us interpret it!

<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

As this is first & foremost an exploratory project, we will take back the results to the client Category Managers & discuss the results, our views on how these insights can be actioned best, and any considerations that need to be taken into account when interpreting.

From there we will recommend applying this same logic to all other categories, as well as potentially across the full-product range.

We will also propose the build of the "Keyword Search Engine" which will help Category Managers extract and utilise the insights held within the data.

<br>
<br>

___

# Data Overview  <a name="data-overview"></a>

Our initial dataset contains 3,500 transactions, each of which shows the alcohol products that were present in that transaction.  

In the code below, we import Pandas, as well as the apriori algorithm from the apyori library, and we bring the raw data into Python.
<br>
```python

# import required Python packages
import pandas as pd
from apyori import apriori

# import the sample data
alcohol_transactions = pd.read_csv("data/sample_data_apriori.csv")

```
<br>

A sample of this data (the first 10 transactions) can be seen below:
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
| … | … | … | … | … | … | … |

<br>
To explain this data, *Transaction 1* (the first row) contained two products, Premium Lager, and Iberia.  As there were only two products in this transaction, the remaining columns are blank.

Transaction 2 (the second row) contained nine products (not all shown in the snippet).  The first nine columns for this row are therefore populated, followed by blank values.

For our sample data, the maximum number of unique products was 45, meaning the table of data had a total of 46 columns (45 for products + transaction_id).

The *apyori* library that we are using does not want the data in this format, it instead wants it passed in as a *list of lists* so we will need to modify it.  The code and logic for this can be found in the Data Preparation section below.

___
<br>
# Apriori Overview  <a name="apriori-overview"></a>

Association Rule Learning is an approach that discovers the strength of relationships between different data-points.  It is commonly utilised to understand which products are frequently (or infrequently) purchased together.

In a business sense this can provide some really interesting, and useful information that can help optimise:

* Product Arrangement/Placement (making the customer journey more efficient)
* Product Recommendations (customers who purchased product A also purchased product B)
* Bundled Discounts (which products should/should not be put together)

One powerful, intuitive, and commonly used algorithm for Association Rule Learning is **Apriori**.

In Apriori there are four key metrics, namely:

* Support
* Confidence
* Expected Confidence
* Lift

Each of these metrics help us understand items, and their relationship with other items in their own way.

<br>
##### Support

Support is extremely intuitive, it simply tells us the percentage of all transactions that contain *both* Item A and Item B.  To calculate this we’d just count up the transactions that include both items, and divide this by the total number of transactions.

You can think of Support as a baseline metric that helps us understand how common or popular this particular *pair* of items is.

<br>
##### Confidence

Confidence takes us a little bit further than Support, and looks more explcitly at the *relationship* between the two items.

It asks "of all transactions that *included item A*, what proportion also included item B?"  

In other words, here we are counting up the number of transactions that contained *both items A and B* and then rather than dividing by *all transactions* like we did for Support, we instead divide this by the *total number of transactions that contained item A*.

A high score for Confidence can mean a strong product relationship - but not always!  When one of the items is very popular we can get an inflated score.  To help us regulate this, we can look at two further metrics, Expected Confidence and Lift!

<br>
##### Expected Confidence

Expected Confidence is quite simple, it is the percentage of *all transactions* that *contained item B*.

This is important as it provides indication of what the Confidence *would be* if there were no relationship between the items.  We can use Expected Confidence, along with Confidence to calculate our final (and most powerful) metric - Lift!

<br>
##### Lift

Lift is the factor by which the Confidence, exceeds the Expected Confidence.  In other words, Lift tells us how likely item B is purchased *when item A is purchased*, while *controlling* for how popular item B is.

We calculate Lift by dividing Confidence by Expected Confidence.

A Lift score *greater than 1* indicates that items A & B appear together *more often* than expected, and conversely a Lift score *less then 1* indicates that items A & B appear together *less often* than expected.

<br>
##### In Practice

While above we're just discussing two products (Item A & Item B) - in reality this score would be calculated between *all* pairs of products, and we could then sort these by Lift score (for example) and see exactly what the strongest or weakest relationships were - and this information would guide our decisions regarding product layout, recommendations for customers, or promotions.

<br>
##### An Important Consideration

Something to consider when assessing the results of Apriori is that, Item/Product relationships that have a *high Lift score* but also have a *low Support score* should be interpreted with caution!

In other words, if we sorted all Item relationships by descending Lift score, the one that comes out on top might initially seem very impressive and it may appear that there is a very strong relationship between the two items.  Always take into account the Support metric - it could be that this relationship is only taking place by chance due to the rarity of the item set.

___
<br>
# Data Preparation  <a name="apriori-data-prep"></a>

As mentioned in the Data Overview section above, the *apyori* library that we are using does not want the data in table format, it instead wants it passed in as a *list of lists* so we will need to modify it here.  

In the code below, we:

* Remove the ID column as it is not required
* Iterate over the DataFrame, appending each transaction to a list, and appending those to a master list
* Print out the first 10 lists from the master list

<br>
```python

# drop ID column
alcohol_transactions.drop("transaction_id", axis = 1, inplace = True)

# modify data for apriori algorithm
transactions_list = []
for index, row in alcohol_transactions.iterrows():
    transaction = list(row.dropna())
    transactions_list.append(transaction)
    
# print out first 10 lists from master list
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

As you can see from the print statement, each transaction (row) from the initial DataFrame is now contained within a list, all making up the master list.

___
<br>
# Applying The Apriori Algorithm <a name="apriori-fit"></a>

In the code below we apply the apriori algorithm from the apyori library.

This algorithm allows us to specify the association rules that we want.  We set:

* A minimum *Support* of 0.003 to eliminate very rare product sets
* A minimum *Confidence* of 0.2
* A minimum *Lift* of 3 to ensure we're only focusing on product sets with strong relationships
* A minimum & maximum length of 2 meaning we're only focusing on product *pairs* rather than larger sets

```python

# apply the apriori algorthm and specify required parameters
apriori_rules = apriori(transactions_list,
                        min_support = 0.003,
                        min_confidence = 0.2,
                        min_lift = 3,
                        min_length = 2,
                        max_length = 2)

# convert the output to a list
apriori_rules = list(apriori_rules)

# print out the first element
apriori_rules[0]

RelationRecord(items=frozenset({'America White', 'American Rose'}), support=0.020745724698626296, ordered_statistics=[OrderedStatistic(items_base=frozenset({'American Rose'}), items_add=frozenset({'America White'}), confidence=0.5323741007194245, lift=3.997849299507762)])

```
<br>
The output from the algorithm is in the form of a generator. We covert this to a list as this is easier to manipulate & analyse.  

Based upon the parameters we set when applying the algorithm, we get 132 product pairs.  We print out the first element from the list to see what the output looks like, and while this contains all the key information we need - to make it easier to analyse (and more accessible & useable for stakeholders) - in the next code snippet, we extract the key elements and use list comprehension to re-work this data to exist as a Pandas DataFrame.

```python

# extract each piece of information
product1 = [list(rule[2][0][0])[0] for rule in apriori_rules]
product2 = [list(rule[2][0][1])[0] for rule in apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

# compile into a single dataframe
apriori_rules_df = pd.DataFrame({"product1" : product1,
                                 "product2" : product2,
                                 "support" : support,
                                 "confidence": confidence,
                                 "lift" : lift})

```
<br>
A sample of this data (the first 5 product pairs - not in any order) can be seen below:
<br>
<br>

| **product1** | **product2** | **support** | **confidence** | **lift** |
|---|---|---|---|---|
| American Rose | America White | 0.021 | 0.532 | 3.998 |
| America White | American White | 0.054 | 0.408 | 3.597 |
| Australian Rose | America White | 0.005 | 0.486 | 3.653 |
| Low Alcohol A.C | America White | 0.003 | 0.462 | 3.466 |
| American Rose | American Red | 0.016 | 0.403 | 3.575 |
| … | … | … | … | … |

<br>
In the DataFrame we have the two products in the pair, and then the three key metrics; Support, Confidence, and Lift. 

___
<br>
# Interpreting The Results <a name="apriori-results"></a>

<br>
#### Associated Products

Now we have our data in a useable format - let's look at the product pairs with the *strongest* relationships - we can do this by sorting our Lift column, in descending order.

```python

# sort pairs by descending Lift
apriori_rules_df.sort_values(by = "lift", ascending = False, inplace = True)

```

<br>
In the table below, we can see the ten highest product relationships, based upon Lift
<br>
<br>

| **product1** | **product2** | **support** | **confidence** | **lift** |
|---|---|---|---|---|
| Wine Gifts | Beer/Lager Gifts | 0.004 | 0.314 | 10.173 |
| Beer/Lager Gifts | Spirits & Fortified | 0.013 | 0.427 | 9.897 |
| Wine Gifts | Spirits & Fortified | 0.006 | 0.412 | 9.537 |
| Red Wine Bxes & 25Cl | White Boxes | 0.015 | 0.474 | 9.344 |
| French White Rhone | French Red | 0.003 | 0.480 | 8.691 |
| Small Sizeswhite Oth | Small Sizes White | 0.005 | 0.559 | 8.340 |
| Small Sizes Red | Small Sizes White | 0.025 | 0.486 | 7.258 |
| French White Loire | French White South | 0.004 | 0.349 | 6.763 |
| French White Rhone | French White 2 | 0.005 | 0.760 | 6.661 |
| Small Sizeswhite Oth | Small Sizes Red | 0.003 | 0.324 | 6.306 |
| Small Sizes Wht Othr | Small Sizes White | 0.003 | 0.414 | 6.176 |

<br>
Interestingly, the strongest relationship exists between two products labelled as "gifts" - this is useful information for the category managers as they may want to ensure that gift products are available in one section of the aisle, rather than existing in their respective product types.

We also see some strong relationships between French wines, and other French wines - which again is extremely useful for category managers who are thinking about the best way to lay out the products - having sections by country rather than necessarily by type might make it easier for customers to find what they are after.

Another interesting association is between products labelled "small".  At this point, we don't know exactly what that means - but it is certainly something to take back to the client as they may be able to make more sense of it, and turn it into an actionable insight!

<br>
#### Search Tool For Category Managers

With the data now stored as a DataFrame, we will also go back to the client with a proposal to build a simple "search" tool for Category Managers to use.

An example of how this might work would be to test a hypothesis around New Zealand wines.

The code below uses a string function to pull back all rows in the DataFrame where *product1* contains the words "New Zealand"

```python

# search based upon text
apriori_rules_df[apriori_rules_df["product1"].str.contains("New Zealand")]

```
<br>
The results of this search, in order of descending Lift are as follows:
<br>
<br>

| **product1** | **product2** | **support** | **confidence** | **lift** |
|---|---|---|---|---|
| New Zealand Red | Malt Whisky | 0.005326605 | 0.271428571 | 5.628986711 |
| New Zealand Red | Iberia White | 0.007289038 | 0.371428571 | 4.616326531 |
| New Zealand Red | New Zealand White | 0.012615643 | 0.642857143 | 4.613825812 |
| New Zealand Red | French White South | 0.004485562 | 0.228571429 | 4.431055901 |
| New Zealand Red | French White 2 | 0.009531819 | 0.485714286 | 4.256862057 |
| New Zealand Red | French Red | 0.004205214 | 0.214285714 | 3.879985497 |
| New Zealand Red | French Red South | 0.006447996 | 0.328571429 | 3.868033946 |
| New Zealand Red | South America | 0.010933558 | 0.557142857 | 3.799863425 |
| New Zealand Red | Other Red | 0.004485562 | 0.228571429 | 3.591692889 |
| New Zealand Red | Iberia | 0.012054948 | 0.614285714 | 3.528433402 |
| New Zealand Red | Champagne | 0.008690777 | 0.442857143 | 3.526052296 |
| New Zealand White | South America White | 0.049341183 | 0.354124748 | 3.423205902 |
| New Zealand Red | French Red 2 | 0.010092515 | 0.514285714 | 3.359811617 |
| New Zealand Red | South America White | 0.006728343 | 0.342857143 | 3.314285714 |
| New Zealand Red | Australia White | 0.007289038 | 0.371428571 | 3.215742025 |

<br>
There appears to be *some* relationship between New Zealand wines and other New Zealand wines, but what is also interesting is that New Zealand wines seem to be more associated with French & South American wines than they are with Australian Wines.

New Zealand & Australia are often grouped together, but in terms of wine this wouldn't make sense - perhaps because of the difference climates the wines are very different and thus it wouldn't make sense to group wines by geographical proximity, but by preference instead.  This is only a hypothesis for now - we will need to take this back to the client and get their category experts to help us interpret it!

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

As this was first & foremost an exploratory project, we will take back the results to the client Category Managers & discuss the results, our views on how these insights can be actioned best, and any considerations that need to be taken into account when interpreting.

From there we will recommend applying this same logic to all other categories, as well as potentially across the full-product range.

We will also propose the build of the "Keyword Search Engine" which will help Category Managers extract and utilise the insights held within the data.