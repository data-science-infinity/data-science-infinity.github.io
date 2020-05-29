---
layout: post
title: Data Science Post
image: "/posts/25.png"
tags: [Data Science, Machine Learning, Python]
---
Syntax highlighting is a feature that displays source code, in different colors and fonts according to the category of terms. This feature facilitates writing in a structured language such as a programming language or a markup language as both structures and syntax errors are visually distinct. Highlighting does not affect the meaning of the text itself; it is intended only for human readers.[^1]

```ruby
def abs()
    print(x)
```

# A/B Testing using Chi-Square Test for Independence
### This test would be used in cases where the response is binary, for example did purchase vs. did not purchase


## Import required packages


```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2
```

## Import Data


```python
campaign_data = pd.read_excel("grocery_database.xlsx", sheet_name = "campaign_data")
import matplotlib.pyplot as plt

plt.plot([1,2,3],[2,2,2])
```




    [<matplotlib.lines.Line2D at 0x1db010ce588>]




![png](output_4_1.png)


## Filter Data


```python
# only want to test Mailer 1 vs Mailer 2
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]

# only want mailer type and signup flag columns
# ensure that they're strings no integers
campaign_data = campaign_data[["mailer_type","signup_flag"]].applymap(str)
campaign_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mailer_type</th>
      <th>signup_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mailer1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mailer1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mailer2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mailer1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mailer2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>863</th>
      <td>Mailer2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>864</th>
      <td>Mailer1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>865</th>
      <td>Mailer2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>866</th>
      <td>Mailer1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>867</th>
      <td>Mailer2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>711 rows × 2 columns</p>
</div>



## Summarise data to get our observed frequencies


```python
observed_values = pd.crosstab(campaign_data["mailer_type"],campaign_data["signup_flag"]).values
observed_values
```




    array([[252, 123],
           [209, 127]], dtype=int64)



## Set acceptance criteria for chi-square statistic


```python
acceptance_criteria = 3.84 # equivalent to p-value of 0.05 for a 2x2 test
```

## Calculate expected frequencies & chi-square statistic


```python
# if dof = 1 then use correction = False (Yates correction)
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction=False)
chi2_statistic
```




    1.9414468614812481



## Print the outcome


```python
if chi2_statistic >= acceptance_criteria:
    print("We reject the null hypothesis, and conclude there is a relationship between the two variables")
else:
    print("We accept the null hypothesis, and conclude there is no relationship between the two variables")
```

    We accept the null hypothesis, and conclude there is no relationship between the two variables
    
