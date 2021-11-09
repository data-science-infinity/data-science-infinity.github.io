---
layout: post
title: A/B Test 
image: "/posts/mika_baumeister_unsplash.jpg"
tags: [Python, A/B Test]
---

In this project we are perfoming a so called A/B test to assess the sigh up rate of two groups using ***Chi-Square test for independence***. Summary for the business, for the data we have so far, we can see that it's not nessecary to spend money for the expensive mailing and keep the simpler and cheeper one.

---

#### IMPORT REQUIRED PACKAGES

```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2
```

#### IMPORT DATA

```python
campaign_data=pd.read_excel('Data/grocery_database.xlsx', sheet_name='campaign_data')
```

#### FILTER OUR DATA-drop the row with mailer group=Control as our task is to compare mailer1 and mailer2 groups

```python
campaign_data=campaign_data.loc[campaign_data['mailer_type']!='Control']
```

#### SUMMARISE TO GET OUR OBSERVED FREQUENCIES 

```python
observed_values=pd.crosstab(campaign_data['mailer_type'],campaign_data['signup_flag']).values
mailer1_signup_rate=123/(252+123)
mailer2_signup_rate=127/(209+127)
print(mailer1_signup_rate, mailer2_signup_rate)
```

#### SET HYPOTHESIS AND SET ACCEPTANCE CRITERIA

```python
null_hypothesis='There is no relationship between the mailer type and signup rate. They are independent.'
alternate_hypothesis='There is a relationship between the mailer type and signup rate. They are not independent.'

acceptance_criteria=0.05
```

#### CALCULATE EXPECTED FREQUENCIES AND CH-SQUARE STATISTIC

```python
chi2_statistic, p_value, dof, expected_values=chi2_contingency(observed_values, correction=False)
print(chi2_statistic, p_value)
```

#### FIND THE CRITICAL VALUE FOR OUR TEST 

```python
critical_value=chi2.ppf(1-acceptance_criteria, dof)
print(critical_value)
```

#### PRINT THE RESULTS (CHI-SQUARE STATISTIC)

```python
if chi2_statistic >= critical_value:
    print(f'As our chi-square statistic of {chi2_statistic} is higher than our critical value {critical_value}, we reject the null hypothesis and conclude that: {alternate_hypothesis}')
else:
    print(f'As our chi-square statistic of {chi2_statistic} is lower than our critical value {critical_value}, we retain the null hypothesis and conclude that: {null_hypothesis}')

OUTPUT:
As our chi-square statistic of 1.94 is lower than our critical value 3.84, we retain the null hypothesis and conclude that: There is no relationship between the mailer type and signup rate. They are independent.
```   

#### PRINT THE RESULTS (P_VALUE)

```python
if p_value <= acceptance_criteria:
    print(f'As our p-value of {p_value} is lower than our acceptance_criteria {acceptance_criteria}, we reject the null hypothesis and conclude that: {alternate_hypothesis}')
else:
    print(f'As our p-value of {p_value} is higher than our acceptance_criteria {acceptance_criteria}, we retain the null hypothesis and conclude that: {null_hypothesis}')
    
OUTPUT:
As our p-value of 0.16 is higher than our acceptance_criteria 0.05, we retain the null hypothesis and conclude that: There is no relationship between the mailer type and signup rate. They are independent.
```
#### OUR RESULTS

For the data we have so far, we can see that it's not nessecary to spend money for the expensive mailing and keep the simpler and cheeper one.

---
References

Andrew Jones, Data Science Infinity Course, https://www.data-science-infinity.com/

---

