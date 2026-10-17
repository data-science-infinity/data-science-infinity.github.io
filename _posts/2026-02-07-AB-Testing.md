---
layout: post
title: Assessing Campaign Performance
image: "/posts/Performance_Marketing_mod.jpg"
tags: [A/B Testing, Chi-Square, Stats]
---

A/B tests are used in business to describe a hypothesis test. An A/B Test is essentially a randomised experiment containing two groups, A & B that receive different experiences. Within an A/B test, we look to understand and measure the response of each group. Marketing campaign performance analysis measures the effectiveness of marketing initiatives against defined goals. 


---

Example of a Marketing Campaign Pilot:

 - Group 1 – customers received Mailer 1 (a basic cheaper version) to sign up to loyalty scheme
 - Group 2 – customers received Mailer 2 (a colourful more expensive version) to sign up to loyalty scheme
 - Group 3 = control group - received no Mailer at all (but can still sign up to loyalty scheme via main menu)

The marketing team is certain that customers who received a mailer (i.e. Groups 1 or 2) signed up to the company's loyalty scheme more than the control group but are unsure whether the quality of mailer they received made a significant difference. 

We can answer this question using hypothesis testing, in particular using the chi-square test for independence.

---



##### IMPORT REQUIRED PACKAGES
```
import pandas as pd
from scipy.stats import chi2_contingency as cc
from scipy.stats import chi2
```

##### IMPORT DATA (ensure the spreadsheet is located in the same directory as this Python Script File)
```
campaign_data = pd.read_excel('grocery_database.xlsx', sheet_name = 'campaign_data')
```



##### FILTER THE DATA (i.e take out all rows with CTL group using the .loc method)
```
campaign_data = campaign_data.loc[campaign_data['mailer_type'] != 'Control']
```



##### SUMMARISE TO GET OUR OBSERVED FREQUENCIES using .crosstab() method
```
observed_values = pd.crosstab(campaign_data['mailer_type'], campaign_data['signup_flag']).values
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"])
print(observed_values)

mailer1_signup_rate = 123 / (252 +123)
mailer2_signup_rate = 127 / (209 +127)
print(mailer1_signup_rate, mailer2_signup_rate)
```



##### STATE HYPOTHESES & SET ACCEPTANCE CRITERIA
```
null_hypothesis = 'there is no relationship between mailer type and signup rate. They are independednt.'
alternate_hypothesis = 'there is a relationship between mailer type and signup rate. They are NOT independednt.'
acceptance_criteria = 0.05
```



##### CALCULATE EXPECTED FREQUENCIES & CHI SQUARE STATISTIC
```
chi2_statistic, p_value, dof, expected_values = cc(observed_values, correction = False)
print(chi2_statistic, p_value)
```


##### FIND THE CRITICAL VALUE FOR THE TEST using the percentage point function
```
critical_value = chi2.ppf(1- acceptance_criteria, dof)
print(critical_value)
```


##### PRINT THE RESULTS/CONCLUSION (Chi Square Statistic)
```
if chi2_statistic >= critical_value:
    print(f'As our chi-suqre-statistic of {chi2_statistic} is HIGHER than our citical value of {critical_value}, we REJECT the null hypothesis and conclude that {alternate_hypothesis}')
else:
    print(f'As our chi-suqre-statistic of {chi2_statistic} is LOWER than our citical value of {critical_value}, we ACCEPT the null hypothesis and conclude that {null_hypothesis}')
 ```   


##### PRINT THE RESULTS/CONCLUSION (p-value)
```
if p_value <= acceptance_criteria:
    print(f'As our p-value of {p_value} is LOWER than our citical value of {acceptance_criteria}, we REJECT the null hypothesis and conclude that {alternate_hypothesis}')
else:
    print(f'As our p-value of {p_value} is HIGHER than our citical value of {acceptance_criteria}, we ACCEPT the null hypothesis and conclude that {null_hypothesis}')
```
---


#### Output in our console (after the code is run):
```
As our chi-suqre-statistic of 1.9414468614812481 is LOWER than our citical value of 3.841458820694124, we ACCEPT the null hypothesis and conclude that there is no relationship between mailer type and signup rate. They are independednt.
As our p-value of 0.16351152223398197 is HIGHER than our citical value of 0.05, we ACCEPT the null hypothesis and conclude that there is no relationship between mailer type and signup rate. They are independednt.
```
---

#### *Business Insight: The Marketing Team can safely utilise the basic (cheaper) mailer as a means to increase signups. Using the colourful more expensive mailer would result in unnecessary costs/expenses for the company.*
