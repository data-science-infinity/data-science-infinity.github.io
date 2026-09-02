---
layout: post
title: Customer Checkout UI Redesign A/B Test Analysis
image: "/posts/checkout_UI.jpg"
tags: [AB Testing, Hypothesis Testing, Z-Test, Shapiro-Wilk, Mann-Whitney U, Python]
---

Can a checkout redesign boost conversion rates without lowering average order values? In this case study, we evaluate a 30-day e-commerce A/B test using a triad of statistical hypothesis tests - combining a Two-Sample Z-Test for Proportions, Shapiro-Wilk normality testing, and a Mann-Whitney U Test—to deliver a data-backed rollout recommendation.

___

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Data Overview & Preparation](#data-overview)
- [02. Applying Z-Test for Proportions](#z-test-application)
- [03. Applying Shapiro-Wilk to Assess for Data Normality](#shapiro-wilk)
- [04. Applying Mann-Whitney U Test](#mann-whitney)
- [05. Analyzing The Results](#Z-test-results)
- [06. Discussion](#discussion)

___

# Project Overview  <a name="overview-main"></a>
<br>
### Context <a name="overview-context"></a>

An e-commerce retailer redesigned their customer checkout user interface (UI) in an effort to streamline the cart-to-purchase process and improve checkout conversion rates.

The campaign was run across a 30-day testing window where incoming site traffic was randomly split between two groups:
- The `Control Group` used the legacy UI to complete purchases
- The `Test Group` used the new redesigned UI to complete purchases

The product and marketing teams logged daily performance metrics including number of items added to carts, number of completed purchases, and the total spent by customers on a day-to-day basis.

**<u>Objective:</u>** Determine if the redesigned UI significantly improves purchase conversion rates without negatively impacting Average Order Value (AOV). 

<br>

### Actions <a name="overview-actions"></a>

To evaluate the performance of the UI redesign, as well as the impact on Average Order Value (AOV), we structured our analysis across three sequential hypothesis tests:

1. **Primary Metric Evaluation (Conversion Rate):**
   * <u>Objective:</u> Determine whether the new checkout UI drives a statistically significant lift in cart-to-purchase conversion rates.
   * <u>Method:</u> Run a Two-Sample Z-Test for Proportions with `statsmodels`.

2. Normality Diagnostic (Daily Average Order Value):
   * <u>Objective:</u> Evaluate whether daily Average Order Value (spent/purchased) meets the parametric assumption of a normal distribution.
   * <u>Method:</u> Run a Shapiro-Wilk Test on daily AOV for both Control and Test groups using `scipy.stats`.

3. Secondary Metric Evaluation (AOV Impact) 
   * <u>Objective:</u> Assess whether customer spending per completed order changed significantly between UI variants.
   * <u>Method:</u> Select the final two-sample test based on the Shapiro-Wilk diagnostic. If the data is normally distributed, run the Welch's *t*-Test. If the AOV data is non-parametric, run the Mann-Whitney U Test with. Use `scipy.stats`.

<br>

### Results & Discussion <a name="overview-results"></a>

Our 30-day A/B experiment confirmed that the redesigned checkout UI significantly improves purchase completion without harming average order value.

* Conversion Rate Lift: Cart-to-purchase conversion increased from 40.21% (Control) to 59.13% (Test) with a +47.03% relative improvement. Z-Test of Proportions confirmed this difference is statistically significant (p-value [0.0000] < significance level [0.05])  
* Average Order Value (AOV): AOV saw a nominal increase of +$0.51 ($102.14 vs. $102.65), but non-parametric testing confirmed this difference is not statistically significant (p-value [0.2717] > significance level [0.05]).

**<u>Business Recommendation:</u>** Proceed with the complete rollout the redesigned UI to all customers. With the split-testing routing infrastructure already in place, the engineering effort to fully deploy the UI is minimal and carries negligible risk. Continue with further metric testing as detailed in the *Discussion* section of this report.

___

# Data Overview & Preparation  <a name="data-overview"></a>
<br>
We'll start by importing our two groups of data, the `control` group who used the legacy UI and the `test` group who used the new UI for purchasing items in their carts. This data contains sales metrics on each unique day during the 30 day trial in August.

```python
import pandas as pd

control = pd.read_csv('control_group.csv', sep = ';')
test = pd.read_csv('test_group.csv', sep = ';')

control = control.dropna()
test = test.dropna()
```

We have the following columns of interest in both datasets:
* Date
* Spend [USD]
* \# of Add to Cart
* \# of Purchase

Before applying statistical analyses, let's simply aggregate the raw data to determine the conversion rates of items being put into carts to being purchased.

```python
ctrl_conversion_rate = round(control['# of Purchase'].sum() / control['# of Add to Cart'].sum() * 100, 3)
test_conversion_rate = round(test['# of Purchase'].sum() / test['# of Add to Cart'].sum() * 100, 3)

relative_lift = (test_conversion_rate - ctrl_conversion_rate)/ctrl_conversion_rate * 100

print(f'Control Group Conversion Rate = {ctrl_conversion_rate}%')
print(f'Test Group Conversion Rate = {test_conversion_rate}%')
print(f'Relative Lift = {relative_lift:.2f}%')

>> Control Group Conversion Rate = 40.215%
>> Test Group Conversion Rate = 59.128%
>> Relative Lift = 47.03%
```

At first glance, basic conversion rate data aggregation suggests the newly deployed UI improved checkout performance, driving a +47.03% relative lift in conversion rate (rising from 40.22% in the Control group to 59.13% in the Test group).

Looking at the average order value (AOV) data, we seemed to only have a $0.51 increase in revenue, which so far suggests that the new UI did not negatively impact sales.

```python
ctrl_AOV = round(control['Spend [USD]'].sum()/control['# of Purchase'].sum(), 2)
test_AOV = round(test['Spend [USD]'].sum()/test['# of Purchase'].sum(), 2)

print(f'Control AOV = ${ctrl_AOV}')
print(f'Test AOV = ${test_AOV}')

>> Control AOV = $4.41
>> Test AOV = $4.92
```

However, raw descriptive statistics alone cannot determine whether these gains are statistically meaningful or the result of random sampling noise. To establish whether the redesigned UI genuinely drives conversion rate improvements, we evaluate these proportions using a Two-Sample Z-Test for Proportions.

___

# Applying Z-Test for Proportions <a name="z-test-application"></a>

<br>

### State Hypotheses & Significance Level For Test

To kick off our Z-Test, we'll need to define our **Null Hypothesis**, our **Alternate Hypothesis**, and our **Significance Level**. For our significance level, we'll be using the commonly used value of 0.05 (or 5%), which will be carried through for all subsequent tests.

* **Null Hypothesis:** There is no significant relationship between the checkout UI version and the sales conversion rate. They are independent.
* **Alternate Hypothesis:** There is a relationship between the checkout UI version and the sales conversion rate. They are not independent.
* **Significance Level:** 0.05

<br>

## Calculating the P-Value

We want to look at if the new UI led to a significant increase in conversion rate, from items added to the cart to the purchase step.

```
* If p-value >= 0.05: Fail to reject the null hypothesis. UI version and conversion rate are statistically independent
* If p-value < 0.05: Reject the null hypothesis in favor of the alternate
```

To do this, we will `statsmodels.stats.proportion` library was used to import the `proportion_ztest` algorithm, to run our Z-Test. The results of this test will provide a p-value to be compared against our significance level.

<u>Inputs of the proportions_ztest:</u>  
--`count` represents the amount of successes for each dataset: it will be defined as the number of total purchases from each dataset.  
--`nobs (ie. Number of Observations)` will be the number of items added to carts.  
--`alternative`, looking if the % of signups is significantly higher, or *larger*

```python
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

purchases = [test["# of Purchase"].sum(), control["# of Purchase"].sum()]
carts = [test["# of Add to Cart"].sum(), control["# of Add to Cart"].sum()]

z_stat, p_val = proportions_ztest(
    count=purchases, nobs=carts, alternative="larger"
)

print(f"Z-statistic: {z_stat:.4f}")
print(f"p-value:     {p_val:.4f}")

>> Z-statistic: 47.1959
>> p-value:     0.0000
```

Our calculated p-value of 0.0000 is less than our set significance level of 0.05, which provides evidence to reject the null hypothesis in favor of the alternate. The 18.92% jump in cart-to-purchase conversion rate is statistically significant and virtually impossible to have happened by random chance!

___

# Applying Shapiro-Wilk to Assess for Data Normality <a name="shapiro-wilk"></a>

<br>

To assess the effect the UI version had on the Average Order Value (AOV), we first need to determine whether to run a parametric or a non-parametric test through another hypothesis test known as the Shapiro-Wilk test. 

* **Null Hypothesis:** The daily AOV data in both groups is normally distributed
* **Alternate Hypothesis:** The daily AOV data in both groups is not normally distributed
* **Significance Level:** 0.05

Parametric tests such as the standard two-sample *t*-test rely on the assumption of data normality to calculate standard errors and p-values; violating this assumption risks inflating Type I error rates. Running the Shapiro-Wilk test allows us to verify the data structure before selecting a model. 

## Calculating the P-Value

```
* If p-value >= 0.05: The daily AOV in both groups is normally distributed. Run Welch's *t*-test to assess statistical significance
* If p-value < 0.05: The daily AOV data is not normally distributed. Run the Mann Whitney U to assess statistical significance
```

We'll acquire the Shapiro-Wilk p-value outputs with the `scipy` library `stats` module.

```python
from scipy import stats

# Calculate AOV
control["AOV"] = control["Spend [USD]"] / control["# of Purchase"]
test["AOV"] = test["Spend [USD]"] / test["# of Purchase"]

# Run Shapiro-Wilk on AOV
stat_ctrl, p_ctrl = stats.shapiro(control["AOV"])
stat_test, p_test = stats.shapiro(test["AOV"])

print(f"Control AOV - W Stat: {stat_ctrl:.4f}, p-value: {p_ctrl:.4f}")
print(f"Test AOV - W Stat: {stat_test:.4f}, p-value: {p_test:.4f}")

>> Control AOV - W Stat: 0.9132, p-value: 0.0206
>> Test AOV - W Stat: 0.8966, p-value: 0.0069
```

The p-values returned from the Shapiro-Wilk test are less than the significance level of 0.05 for both groups, indicating that the AOV data is not normally distributed.

Consequently, we proceed with the non-parametric Mann-Whitney U test, which compares distribution ranks rather than sample means and requires no distributional assumptions.

___

# Applying Mann-Whitney U Test <a name="mann-whitney"></a>

We've now determined to run the Mann-Whitney U Test to assess our AOV between the control and test data.

* **Null Hypothesis:** There is no statistical difference in the distribution of daily AOV between the Control and Test groups
* **Alternate Hypothesis:** There is a statistically significant difference in the distribution of daily AOV between the Control and Test groups
* **Significance Level:** 0.05

## Calculating the P-Value

```
* If p-value >= 0.05: There is statistically no difference in daily AOV between groups. Fail to reject the null hypothesis. 
* If p-value < 0.05: There is a statistically significant difference in daily AOV between groups. Reject the null hypothesis.
```

With `scipy` `stats` already imported, we can run the `mannwhitneyu` algorithm on our data, then compare the returned p-value to our initial significance level.

**<u>Inputs of mannwhitneyu:</u>**  
--`control["AOV"]`  
--`test["AOV"]`  
--`alternative`: *two-sided* since we are looking at the difference between the control and test AOV.

```python
u_stat, u_pvalue = stats.mannwhitneyu(
    test["AOV"], control["AOV"], alternative="two-sided"
)

print(f"U-statistic: {u_stat:.4f}")
print(f"p-value: {u_pvalue:.4f}")

>> U-statistic: 508.0000
>> p-value: 0.2717
```

The returned p-value for the Mann-Whitney U test is greater than our significance level, thus, we fail to reject the null hypothesis.

Based on the 30-day trial, there is no statistically significant difference in AOV across the two groups.

___

# Analyzing The Results <a name="Z-test-results"></a>

Through the **Z-Test of Proportions**, we calculated:

```
p-value [0.0000] < significance level [0.05]
```

Thus, we reject the null hypothesis in favor of the alternate - indicating that there is a true relationship between the new UI and the increase in conversion rate, and that the observed relative lift was not due to chance.

<br>

Through the **Mann-Whitney U test** used to assess the statistical impact the new UI had on AOV, we determined the following:

```
p-value [0.2717] > significance level [0.05]
```

Thus, we fail to reject the null hypothesis - indicating that the new UI led to no statistically significant impact on AOV.

<br>

**<u>Conclusion:</u>** We can statistically conclude that the new UI led to more items purchased while not negatively impacting AOV. 

___

# Discussion <a name="discussion"></a>

Our 30-day A/B experiment confirms that the redesigned checkout UI delivered a statistically significant boost in conversion rate without degrading customer average order values.

Cart-to-purchase conversion increased from 40.21% (Control) to 59.13% (Test), representing a +47.03% relative lift in checkout efficiency. While raw AOV showed a slight increase of $0.51 cents per order, non-parametric testing confirmed this difference is not statistically significant with a p-value of 0.2717.

**<u>Business Impact:</u>** These statistical conclusions support the business decision to roll out the redesigned UI to all customers. With the split-testing routing infrastructure already in place, the engineering effort to fully deploy the UI is minimal and carries negligible risk.

**<u>Next Steps:</u>**  
* **Monitor Long-Term AOV Trends Post-Rollout:** Higher cart conversion in theory lays the groundwork for revenue growth over time. We recommend tracking AOV and total revenue across a 60–90 day post-launch window to evaluate whether increased purchase frequency translates into higher revenue.
* **Analyze Behavioral Flow Features for Website-wide Applicability:** Analyze the new UI features (such as simplified fields, button placements, color schemes, widgets, etc) to determine which design features drove the highest lift. Then, identify ways to apply these effective design choices to other areas of the website.
